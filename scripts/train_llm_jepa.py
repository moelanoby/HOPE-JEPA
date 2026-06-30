"""QLoRA finetuning of a HOPE-augmented HF LLM with the slot-JEPA (+ Reasoner)
auxiliary objective.

This is the GPU-box entrypoint: 4-bit QLoRA on a 7B model. It is NOT run on
this machine (no GPU here) -- correctness is covered by
`tests/test_llm_smoke.py` against a tiny CPU model. This script is ready to run
on a CUDA box with `bitsandbytes` installed.

What it does, per the approved plan:
  1. Load the base model with `BitsAndBytesConfig(load_in_4bit=True)` +
     `prepare_model_for_kbit_training`.
  2. Wrap base params with `peft.LoraConfig`; keep the new HOPE / slot-JEPA /
     Reasoner params FULLY trainable (they live on the wrapper, outside the
     quantized base, so they are full-precision by default and re-enabled
     explicitly after wrapping).
  3. Build `HopeLLM(cfg)` -- which splices HOPE layers and attaches the slot
     JEPA objective (+ Reasoner if enabled).
  4. Loop: set_global_step -> HopeLLM.forward (CE + slot-JEPA + SIGReg + div
     + optional predict-ahead) -> backward -> opt.step -> global_step++.

Usage:
    # FABLE.5 traces (auto-detects row_json -> parses JSON -> pulls "completion"):
    python scripts/train_llm_jepa.py --config config/llm_default.yaml \
        --dataset Crownelius/Complete-FABLE.5-traces-2M --output runs/fable_hope

    # Standard {"text": ...} dataset:
    python scripts/train_llm_jepa.py --config config/llm_default.yaml \
        --dataset tatsu-lab/alpaca --text_column instruction --output runs/alpaca
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

# Defragment the CUDA caching allocator before any CUDA context is created.
# QLoRA with gradient checkpointing fragments the 14-16 GiB budget heavily
# (the OOM traceback typically shows multi-GB "reserved but unallocated");
# expandable_segments lets freed blocks be returned across segment boundaries.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml

# Run from repo root so `hope_jepa` imports resolve (mirrors train_ssl.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hope_jepa.llm import HopeLLM, HopeLlmConfig, set_global_step
# Low-memory optimizers (LOMO/AdaLomo fuse their update into backward; LISA
# activates only a subset of decoder layers per step). Imported at module scope
# because the training loop isinstance-checks these types.
from hope_jepa.optim import LOMO, AdaLomo


# ---------------------------------------------------------------------------
def load_config(path: str) -> HopeLlmConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return HopeLlmConfig.from_dict(raw)


class _NullCtx:
    """A no-op context manager (used to disable AMP uniformly on CPU)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _oom_advice(args) -> str:
    """Actionable text shown when a CUDA OOM fires during the training step."""
    return (
        "\n" + "!" * 72 + "\n"
        "CUDA out of memory during the training step. The HOPE/Titans recurrence\n"
        "is the memory sink -- it keeps per-token activations through backward.\n"
        "Re-run with a smaller footprint, e.g. (in order of impact):\n"
        f"  --max_len {max(128, args.max_len // 2)}          # halve sequence length (biggest lever)\n"
        f"  --batch_size 1           # already 1? then only --max_len helps\n"
        "  --target_layers last:1    # fewer HOPE-swapped layers\n"
        "  --jepa_layers last:1      # fewer slot-JEPA heads (edit the config)\n"
        "  --optimizer adalomo       # fuses update into backward (no grad state)\n"
        "  --optimizer lisa --lisa_k 1   # freeze most base layers each step\n"
        "The auto-fit already shrank batch_size/max_len on a <=16GB GPU; pass the\n"
        "flags above explicitly to go smaller.\n" + "!" * 72
    )


def _split_lisa_params(model: HopeLLM):
    """Group the model's trainable params for LISA.

    Returns (always_active, layer_groups) where:
      * always_active  -- the NEW wrapper modules (HOPE / slot-JEPA / Reasoner)
        that must train EVERY step (they carry the new capabilities). These are
        everything trainable that is NOT a base-decoder-layer LoRA adapter.
      * layer_groups   -- one list of params PER base decoder layer, holding its
        LoRA adapter params. LISA activates only K of these per step.

    Grouping is by decoding the parameter NAME: peft LoRA params contain a
    `.layers.<i>.` segment under the base model (e.g.
    `base_model.model.model.layers.5.self_attn.q_proj.lora_A.default.weight`).
    Everything else trainable (jepa.*, reasoner.*, HopeDecoderLayer params...)
    is treated as always-on.
    """
    import re
    always_active, layer_groups = [], {}
    # Match the decoder layer index in a parameter path. The Llama-family base
    # exposes layers as `model.layers.<i>`; peft nests these under
    # `base_model.model.<...>`.
    layer_re = re.compile(r"layers\.(\d+)\.")

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = layer_re.search(name)
        if m:
            layer_groups.setdefault(int(m.group(1)), []).append(p)
        else:
            always_active.append(p)

    # Sort layer groups by index for determinism; drop any empty entries.
    groups = [layer_groups[i] for i in sorted(layer_groups) if layer_groups[i]]
    if not groups:
        # No per-layer params found (e.g. placement with no base LoRA targets):
        # fall back to treating everything as always-on so LISA still runs.
        always_active = [p for p in model.parameters() if p.requires_grad]
        groups = [[always_active.pop()]] if always_active else []
    return always_active, groups


def apply_qlora(model: HopeLLM, cfg: HopeLlmConfig):
    """Wrap the *base HF model's* params in LoRA adapters; keep all NEW
    (HOPE / slot-JEPA / Reasoner) modules fully trainable.

    Important: the new modules live on the `HopeLLM` wrapper, OUTSIDE the
    quantized base model. They are therefore already full-precision and not
    frozen by `prepare_model_for_kbit_training` (which only freezes the base).
    We must NOT put them in `modules_to_save` -- that peft option is for
    replacing *quantized base* modules with fp32 copies, and matching it by
    scraped leaf-name suffixes collides with `target_modules`, raising
    "No modules were targeted for adaptation". So we leave it at None and just
    LoRA the base attention projections.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING as TARGETS

    base = model.model  # the underlying HF CausalLM
    if cfg.training.qlora:
        # k-bit-safe: freeze base norms, enable grad-checkpointing-friendly
        # inputs. Does NOT touch the wrapper's new modules (jepa/reasoner).
        base = prepare_model_for_kbit_training(base)

    # Target the attention projections of the base model (q/k/v/o for Llama).
    # Auto-resolve the target module names from the model's architecture.
    arch = getattr(base.config, "architectures", ["LlamaForCausalLM"])[0]
    target_modules = TARGETS.get(arch, ["q_proj", "v_proj", "k_proj", "o_proj"])

    peft_cfg = LoraConfig(
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        # modules_to_save intentionally None: the new HOPE/JEPA/Reasoner
        # modules are not part of the quantized base, so they're already
        # trainable -- no peft copy needed.
    )
    # Wrap the base model in-place; the HopeLLM wrapper still owns it.
    model.model = get_peft_model(base, peft_cfg)

    # Explicitly (re)enable training on the new HOPE / slot-JEPA / Reasoner
    # modules. They live on the wrapper, outside the LoRA-wrapped base, but we
    # make sure here regardless of what prepare_model_for_kbit_training /
    # get_peft_model did to requires_grad upstream of them.
    from hope_jepa.llm.hope_block import HopeDecoderLayer, MACAttnAdapter
    n_unfrozen = 0
    for name, mod in model.named_modules():
        if isinstance(mod, (HopeDecoderLayer, MACAttnAdapter)) or \
           name.startswith("jepa.") or name.startswith("reasoner."):
            for p in mod.parameters(recurse=True):
                if not p.requires_grad:
                    p.requires_grad_(True)
                    n_unfrozen += 1
    return model


# ---------------------------------------------------------------------------
def _extract_text(ex: dict, text_column: str | None,
                  json_field: str | None) -> str:
    """Pull a plain-text string out of one dataset row.

    Three cases:
      1. `text_column` is given (or auto-detected) and the row's value is a
         plain string -> return it. This covers {"text": ...} and the FABLE.5
         dataset's {"row_json": "<json string>"} when you WANT the raw json.
      2. The column holds a JSON string (FABLE.5 `row_json`): parse it and pull
         `json_field` (default: try "completion", then "message.content", then
         "content"). Returns "" if the field is missing or the row is a
         non-text operation (enqueue/dequeue/etc.).
      3. Nothing usable -> "" (the caller skips it).
    """
    import json as _json

    col = text_column
    if col is None or col not in ex:
        # auto-pick: prefer an explicit text column, else a chat column.
        for cand in ("text", "conversations", "messages", "row_json",
                     "content", "prompt", "completion"):
            if cand in ex:
                col = cand
                break
    if col is None or col not in ex:
        return ""
    val = ex[col]

    # --- Chat-format datasets (sharegpt/openassistant style): a list of turns
    # like [{"from":"human","value":"..."},{"from":"gpt","value":"..."}] or
    # [{"role":"user","content":"..."},...]. Flatten to a single string we can
    # train the LM on. This is the most common reason rows were silently
    # skipped before (no text/content/completion column exists). ---
    if isinstance(val, list) and val and isinstance(val[0], dict):
        return _flatten_turns(val)

    if not isinstance(val, str):
        # Some datasets store dicts natively (not as a json string).
        if isinstance(val, (list, dict)):
            val = _json.dumps(val)
        else:
            val = str(val)

    # If the column is a JSON string and the user wants a nested field, parse.
    looks_json = val.lstrip().startswith(("{", "["))
    if looks_json and (json_field is not None or col == "row_json"):
        try:
            obj = _json.loads(val)
        except (ValueError, TypeError):
            return val  # not actually json; return as-is
        fields = [json_field] if json_field else \
                 ["completion", "content", "text", "answer"]
        for f in fields:
            v = _dig_field(obj, f)
            if isinstance(v, str) and v.strip():
                return v
        # message.content style (list of messages)
        if isinstance(obj, dict) and "message" in obj:
            mc = _dig_field(obj["message"], "content")
            if isinstance(mc, str) and mc.strip():
                return mc
        return ""   # row exists but has no usable text (e.g. enqueue/dequeue)
    return val


def _dig_field(obj, field: str):
    """Get obj[field] or obj[field][.sub...] supporting dotted paths and the
    common 'content' nested under message/choices."""
    cur = obj
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def _flatten_turns(turns) -> str:
    """Flatten a list of chat turns into a single training string.

    Handles the two common schemas:
      * sharegpt:  [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
      * openai:    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    The role/value keys are tried in order. Turns with empty text are dropped.
    We join with newlines so the model sees one contiguous sequence.
    """
    parts = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        # Role: "from" (sharegpt) or "role" (openai).
        role = t.get("from") or t.get("role") or ""
        # Text: "value" (sharegpt) or "content" (openai).
        text = t.get("value")
        if text is None:
            text = t.get("content")
        if text is None:
            # message.content nested style.
            msg = t.get("message")
            if isinstance(msg, dict):
                text = msg.get("content")
        if isinstance(text, list):
            # content can itself be a list of {text: ...} blocks.
            text = "\n".join(
                (b.get("text") if isinstance(b, dict) else str(b)) or ""
                for b in text
            )
        if not isinstance(text, str) or not text.strip():
            continue
        parts.append(f"{role}: {text}" if role else text)
    return "\n".join(parts).strip()


def iterate_batches(tokenizer, dataset, max_len: int, batch_size: int, device,
                    text_column: str | None = None,
                    json_field: str | None = None,
                    max_rows: int = 0):
    """Lazily yield (input_ids, attention_mask, labels) batches.

    LAZY: rows are streamed (no materializing 2M strings into memory). For each
    row we extract a plain-text string via `_extract_text` and skip empties
    (FABLE.5 has many non-text operation rows). When a full `batch_size` of
    non-empty texts is collected, tokenize+pad and yield.

    Args:
        text_column: column to read (None => auto-detect 'text'/'row_json'/...).
        json_field:  if the column is a JSON string, pull this nested field
                     (None => try completion/content/text/answer).
        max_rows:    stop after scanning this many rows (0 = no cap).
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
        else tokenizer.eos_token_id
    buf = []
    scanned = 0
    for ex in dataset:
        scanned += 1
        if max_rows and scanned > max_rows:
            break
        txt = _extract_text(ex, text_column, json_field)
        if not txt or not txt.strip():
            continue
        buf.append(txt)
        if len(buf) >= batch_size:
            yield _encode_batch(tokenizer, buf, max_len, pad_id, device)
            buf = []
    if buf:   # final partial batch
        yield _encode_batch(tokenizer, buf, max_len, pad_id, device)


def _encode_batch(tokenizer, texts, max_len, pad_id, device):
    import torch
    enc = tokenizer(texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=max_len)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    labels = input_ids.clone()
    labels[input_ids == pad_id] = -100
    return (input_ids.to(device), attn.to(device), labels.to(device))


class PrefetchBatches:
    """Pre-tokenize batches on a background CPU thread so the GPU never waits
    on the tokenizer.

    Wraps `iterate_batches`: a worker thread runs the (CPU-bound) text scan +
    tokenize loop and pushes finished batches onto a bounded queue. The main
    (GPU) thread pulls batches off the queue, so tokenization overlaps with the
    forward/backward compute. All `.to(device)` H2D copies happen on the worker
    thread here -- they are cheap relative to tokenize and keep the main thread
    purely on GPU work. The default `queue.maxsize=prefetch` bounds memory.

    On any worker exception the error is re-raised on the main thread at the
    next `next()` call (so failures are not silently swallowed). With
    `prefetch=0` the caller should use `iterate_batches` directly (synchronous).
    """

    def __init__(self, iterator_factory, prefetch: int, device):
        from queue import Queue
        from threading import Thread
        self._device = device
        self._sentinel = object()
        self._q: Queue = Queue(maxsize=max(1, prefetch))
        self._exc: list = []
        self._thread = Thread(target=self._run, args=(iterator_factory,), daemon=True)
        self._thread.start()

    def _run(self, iterator_factory):
        try:
            for batch in iterator_factory():
                self._q.put(batch)
        except BaseException as e:  # surface on the main thread
            self._exc.append(e)
        finally:
            self._q.put(self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is self._sentinel:
            if self._exc:
                raise self._exc[0]
            raise StopIteration
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/llm_default.yaml")
    ap.add_argument("--dataset", required=True,
                    help="HF dataset id or path to a jsonl/parquet")
    ap.add_argument("--text_column", default=None,
                    help="Column holding the text (auto-detects 'text'/'row_json'/"
                         "'content'/'prompt'/'completion' if not set)")
    ap.add_argument("--json_field", default=None,
                    help="Nested field to extract when the column is a JSON string "
                         "(auto-tries 'completion'/'content'/'text'/'answer' for "
                         "row_json-style datasets like FABLE.5)")
    ap.add_argument("--output", default="runs/hope_jepa")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="Number of gradient accumulation steps (default: 1)")
    ap.add_argument("--optimizer", default="adamw",
                    choices=["adamw", "adamw_8bit", "paged_adamw_8bit", "paged_adamw_32bit",
                             "lomo", "adalomo", "lisa"],
                    help="Optimizer: adamw (default) | 8-bit/paged bnb variants | "
                         "lomo (fused SGD, global clip, 2 backward passes) | "
                         "adalomo (fused momentum, per-tensor clip, 1 pass -- "
                         "recommended low-memory) | "
                         "lisa (Layerwise Importance Sampled AdamW: activate only "
                         "--lisa_k decoder layers per step).")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="cap on total steps (0 = no cap)")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True,
                    help="Enable gradient checkpointing (default: True)")
    ap.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing",
                    help="Disable gradient checkpointing")
    # --- Single-GPU speedups -------------------------------------------------
    ap.add_argument("--amp", action="store_true", default=True,
                    help="bf16 autocast for the custom (HOPE/JEPA/SIGReg/CE) compute "
                         "(default: True; the 4-bit base already runs low-precision, "
                         "but the added heads run fp32 without this). Auto-disabled on CPU.")
    ap.add_argument("--no_amp", action="store_false", dest="amp",
                    help="Disable bf16 autocast (run custom compute in fp32)")
    ap.add_argument("--compile", action="store_true", default=False,
                    help="torch.compile the model (default: OFF). Experimental with "
                         "4-bit QLoRA + the Titans recurrence; falls back to eager if "
                    "graph capture fails. Enable only if you have time to debug.")
    ap.add_argument("--prefetch", type=int, default=4,
                    help="Number of batches to pre-tokenize on a CPU thread so the GPU "
                         "is not starved by the tokenizer (0 = disable; default: 4)")
    ap.add_argument("--diag_every", type=int, default=50,
                    help="Only run the expensive eff_rank SVD + sparsity diagnostic "
                         "every N optimizer steps (default: 50). It feeds only the log "
                         "line but costs an O(d^3) matmul + GPU sync each step.")
    # --- LISA (Layerwise Importance Sampled AdamW) knobs ----------------------
    ap.add_argument("--lisa_k", type=int, default=2,
                    help="LISA: number of base decoder layers activated per step "
                         "(the rest are frozen, so backward is pruned into them). "
                         "Only used with --optimizer lisa. Paper default: 2.")
    ap.add_argument("--lisa_refresh_every", type=int, default=50,
                    help="LISA: re-rank layers by accumulated grad-norm and rebuild "
                         "the sampling distribution every N steps (default: 50). "
                         "Only used with --optimizer lisa.")
    args = ap.parse_args()

    # --- Auto-fit memory knobs to the GPU (only if the user did NOT pass them). -
    # A 3B + 152K-vocab model under QLoRA fits in ~16GB at batch_size=1, max_len
    # 512; the script's hard defaults (batch_size=4, max_len=1024) were sized for
    # a 24GB+ card and OOMs a T4/A10 on the first backward. We detect a small GPU
    # and shrink the defaults, but only for args the user left at their default
    # (an explicit flag always wins). Detected by comparing to the argparse
    # default -- crude but reliable for these knobs.
    DEFAULTS = {"batch_size": 4, "max_len": 1024, "prefetch": 4}
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb <= 17.0:                       # T4/P100 (16GB) and smaller
            # The biggest single-GPU OOM levers: sequence length and batch size.
            if args.max_len == DEFAULTS["max_len"]:
                args.max_len = 512
                print(f"[autofit] {vram_gb:.1f}GB GPU: --max_len 1024 -> 512 "
                      f"(pass --max_len to override)", flush=True)
            if args.batch_size == DEFAULTS["batch_size"]:
                args.batch_size = 1
                print(f"[autofit] {vram_gb:.1f}GB GPU: --batch_size 4 -> 1 "
                      f"(pass --batch_size to override)", flush=True)
            # Prefetch 4 batches of [B,T] CPU tensors is fine, but on a tiny card
            # keep the CPU-side queue small.
            if args.prefetch == DEFAULTS["prefetch"]:
                args.prefetch = 2
        # Also strongly recommend the paged 8-bit optimizer on small cards: it
        # offloads optimizer state to CPU on OOM spikes instead of crashing.

    cfg = load_config(args.config)

    # --- Model ---
    model = HopeLLM(cfg)                       # loads + splices HOPE layers
    model = apply_qlora(model, cfg)            # 4-bit QLoRA on base, new params trainable
    # Throttle the per-step eff_rank SVD + sparsity diagnostic. It feeds only
    # the log line but costs an O(d^3) matmul + GPU sync every step otherwise.
    model.jepa.diag_every = max(1, args.diag_every)

    if args.gradient_checkpointing:
        # Try to pass gradient_checkpointing_kwargs to avoid the PyTorch 2.9 warning
        try:
            model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            try:
                model.model.gradient_checkpointing_enable()
            except Exception:
                pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    # --- Mixed precision -----------------------------------------------------
    # The 4-bit base already runs low-precision, but the spliced HOPE/Titans,
    # slot-JEPA, SIGReg and chunked-CE compute all run fp32 unless wrapped.
    # bf16 autocast engages the T4 tensor cores for those ops (~1.5-2x) with no
    # accuracy caveat: CE cross-entropy and SIGReg covariance already upcast to
    # fp32 inside their kernels. Auto-disabled on CPU (no benefit / no bf16 path).
    use_amp = args.amp and device == "cuda"
    amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp
               else _NullCtx())

    # --- Optional torch.compile (experimental) --------------------------------
    # The Titans recurrence is a long Python loop and 4-bit QLoRA has known
    # compile-compatibility quirks, so this is OFF by default and falls back to
    # eager if graph capture throws.
    if args.compile and device == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[compile] torch.compile enabled (mode=reduce-overhead)", flush=True)
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[compile] torch.compile unavailable, falling back to eager: {e}",
                  flush=True)

    # --- Data ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train") \
        if not os.path.exists(args.dataset) \
        else load_dataset("json", data_files=args.dataset, split="train")

    # --- Optimizer (LoRA + new params) ---
    # `trainable` is the always-on trainable set (LoRA adapters + the new
    # HOPE/JEPA/Reasoner wrapper modules). For AdamW-family optimizers this is
    # the full param list; for LISA it is split into (always-on wrapper params)
    # + (per-decoder-layer LoRA groups) so only K layers activate per step.
    trainable = [p for p in model.parameters() if p.requires_grad]

    # Detect whether the optimizer fuses its update into backward (LOMO/AdaLomo).
    # If so, gradient clipping / zero_grad are handled inside the fused hooks and
    # must NOT be called explicitly (they'd double-update or error). Fused
    # optimizers update params IN PLACE during backward, so they are INCOMPATIBLE
    # with gradient accumulation (you cannot accumulate grads across micro-batches
    # when each backward already applied its update).
    fused_opt = args.optimizer in ("lomo", "adalomo")
    if fused_opt and args.grad_accum > 1:
        raise SystemExit(
            f"--optimizer {args.optimizer} fuses the update into backward and so "
            f"cannot be combined with --grad_accum > 1 (each backward already "
            f"updates the params). Use --grad_accum 1, or pick a non-fused "
            f"optimizer (adamw / paged_adamw_8bit / lisa).")
    lisa_opt = args.optimizer == "lisa"
    opt = None
    lisa = None   # the LISA optimizer (for per-step layer sampling) if lisa_opt

    if lisa_opt:
        from hope_jepa.optim import LISA
        always_active, layer_groups = _split_lisa_params(model)
        # `trainable` for LISA = always-on params only (the layer groups are
        # toggled per step and must not be in the "always" clip set).
        trainable = always_active
        lisa = LISA(
            always_active_params=always_active,
            layer_groups=layer_groups,
            lr=cfg.training.lr,
            k=min(args.lisa_k, len(layer_groups)),
            refresh_every=args.lisa_refresh_every,
            weight_decay=0.0,
        )
        # Mark the LoRA adapter params as LISA-toggleable (frozen base stays off).
        lora_params = [p for g in layer_groups for p in g]
        lisa.mark_trainable(lora_params)
        opt = lisa
    elif args.optimizer == "lomo":
        opt = LOMO(trainable, lr=cfg.training.lr, clip_grad=1.0)
    elif args.optimizer == "adalomo":
        opt = AdaLomo(trainable, lr=cfg.training.lr, momentum=0.9, clip_grad=1.0)
    elif args.optimizer == "adamw":
        opt = torch.optim.AdamW(trainable, lr=cfg.training.lr, weight_decay=0.0)
    elif args.optimizer == "adamw_8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable, lr=cfg.training.lr, weight_decay=0.0)
    elif args.optimizer == "paged_adamw_8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(trainable, lr=cfg.training.lr, weight_decay=0.0)
    elif args.optimizer == "paged_adamw_32bit":
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW32bit(trainable, lr=cfg.training.lr, weight_decay=0.0)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Before the first step, LISA must freeze its inactive layers so the first
    # forward prunes backward into them.
    os.makedirs(args.output, exist_ok=True)
    step = 0
    accum_steps = args.grad_accum
    opt.zero_grad()
    if lisa is not None:
        lisa.sample_layers(step)

    # --- Batch iterator: prefetch on a CPU thread if requested, else sync. ----
    # `--prefetch` (default 4) keeps the GPU fed while the tokenizer runs. The
    # worker builds batches; the GPU thread consumes them. prefetch=0 -> the old
    # synchronous `iterate_batches` (no thread overhead).
    def _make_iter():
        return iterate_batches(
            tokenizer, ds, args.max_len, args.batch_size, device,
            text_column=args.text_column, json_field=args.json_field,
        )

    if args.prefetch and args.prefetch > 0:
        batch_iter = PrefetchBatches(_make_iter, args.prefetch, device)
    else:
        batch_iter = _make_iter()

    def _log(epoch, s, out, suffix=""):
        """One coalesced log line. `out.ce_loss` / `out.loss` are pulled to CPU
        in a single stack so we issue at most one GPU->CPU sync for the scalars
        that weren't already `.item()`-ed inside the diag dict."""
        d = out.jepa_diag
        scalars = torch.stack([
            out.loss.detach().reshape(()),
            out.ce_loss.detach().reshape(()),
        ]).cpu()
        print(f"[ep{epoch} s{s}] loss={float(scalars[0]):.4f} "
              f"ce={float(scalars[1]):.4f} jepa={d['jepa']:.4f} "
              f"sigreg={d['sigreg']:.4f} div={d['slot_div']:.4f} "
              f"sparse={d['slot_sparsity']:.3f} effrank={d['eff_rank']:.1f}"
              f"{suffix}", flush=True)

    for epoch in range(args.epochs):
        batch_idx = 0
        out = None
        for input_ids, attn_mask, labels in batch_iter:
            # LISA: pick this step's active layers BEFORE forward so backward is
            # pruned into frozen layers. Skip on accumulation sub-steps (the
            # active set is fixed for the whole accumulated step).
            if lisa is not None and batch_idx % accum_steps == 0:
                lisa.sample_layers(step)

            set_global_step(model, step)

            try:
                with amp_ctx:
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                                labels=labels)

                loss = out.loss / accum_steps
                if fused_opt and isinstance(opt, AdaLomo):
                    # AdaLomo: single backward applies the fused update.
                    loss.backward()
                elif fused_opt and isinstance(opt, LOMO):
                    # LOMO needs the GLOBAL grad norm, which is only known after
                    # backward -- but the update must happen DURING backward. So:
                    # (1) a DRY backward (retain_graph=True so the graph survives)
                    #     whose hooks only accumulate the global norm; then
                    # (2) zero grads, arm the live hooks, and a LIVE backward that
                    #     clips each grad by the measured norm and applies the SGD
                    #     step in place, freeing each grad as it goes.
                    opt.zero_grad()
                    loss.backward(retain_graph=True)   # dry: hooks only measure norm
                    opt.begin_fused_update()           # finalize norm, arm live hooks
                    loss.backward()                    # live: hooks apply + free grads
                else:
                    loss.backward()
            except torch.cuda.OutOfMemoryError:
                # Clear whatever survived so the traceback/advice prints cleanly,
                # then give actionable knobs instead of a bare stack trace.
                try:
                    opt.zero_grad(set_to_none=True)
                except Exception:
                    pass
                torch.cuda.empty_cache()
                print(_oom_advice(args), flush=True)
                raise SystemExit(1)

            batch_idx += 1
            if batch_idx % accum_steps == 0:
                if not fused_opt:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                if not fused_opt:
                    opt.zero_grad()
                step += 1
                _log(epoch, step, out)

            if args.max_steps and step >= args.max_steps:
                break

        # End of epoch: step remaining gradients
        if batch_idx % accum_steps != 0 and out is not None:
            if not fused_opt:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            if not fused_opt:
                opt.zero_grad()
            step += 1
            _log(epoch, step, out, suffix=" (epoch end)")

        if args.max_steps and step >= args.max_steps:
            break

    # Safety net: if NO optimizer steps ran, the dataset yielded nothing usable.
    # This is almost always a text-column/schema mismatch (e.g. a sharegpt
    # `conversations` dataset with no `text`/`content` column). Fail loudly
    # instead of silently saving an untrained checkpoint.
    if step == 0:
        print("\n" + "!" * 72, flush=True)
        print("ERROR: 0 optimizer steps ran -- the model was NOT trained, so", flush=True)
        print("nothing useful was saved. The dataset yielded no usable batches.", flush=True)
        print("!" * 72, flush=True)
        print("Likely cause: the rows don't have a text column the loader "
              "recognizes.\n", flush=True)
        # Peek at the first row to show the user what schema it actually has.
        try:
            sample = next(iter(ds))
            print(f"First row keys: {list(sample.keys())}", flush=True)
            for k, v in sample.items():
                vt = type(v).__name__
                prev = (str(v)[:120] + "...") if isinstance(v, str) else \
                       (f"<{vt} of len {len(v)}>" if isinstance(v, (list, dict)) else str(v))
                print(f"  {k} ({vt}): {prev}", flush=True)
            print("\nFix: pass the right --text_column, e.g.", flush=True)
            print("  --text_column conversations   # sharegpt chat format", flush=True)
            print("  --text_column text            # plain-text column", flush=True)
            print("or --json_field for a nested field inside a JSON-string column.",
                  flush=True)
        except Exception as e:
            print(f"(could not peek at dataset: {e})", flush=True)
        print("\n" + "!" * 72, flush=True)
        return

    # Save adapters + the new (non-quantized) HOPE/JEPA modules.
    print(f"\nTrained {step} optimizer step(s).", flush=True)
    model.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
