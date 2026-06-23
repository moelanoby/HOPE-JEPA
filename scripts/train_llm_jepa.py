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

import torch
import yaml

# Run from repo root so `hope_jepa` imports resolve (mirrors train_ssl.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hope_jepa.llm import HopeLLM, HopeLlmConfig, set_global_step


# ---------------------------------------------------------------------------
def load_config(path: str) -> HopeLlmConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return HopeLlmConfig.from_dict(raw)


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
        # auto-pick: prefer an explicit text column, else row_json.
        for cand in ("text", "row_json", "content", "prompt", "completion"):
            if cand in ex:
                col = cand
                break
    if col is None or col not in ex:
        return ""
    val = ex[col]
    if not isinstance(val, str):
        # Some datasets store lists/dicts natively (not as a json string).
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
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="cap on total steps (0 = no cap)")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True,
                    help="Enable gradient checkpointing (default: True)")
    ap.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing",
                    help="Disable gradient checkpointing")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # --- Model ---
    model = HopeLLM(cfg)                       # loads + splices HOPE layers
    model = apply_qlora(model, cfg)            # 4-bit QLoRA on base, new params trainable
    
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
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.training.lr, weight_decay=0.0)

    os.makedirs(args.output, exist_ok=True)
    step = 0
    accum_steps = args.grad_accum
    opt.zero_grad()
    
    for epoch in range(args.epochs):
        batch_idx = 0
        out = None
        for input_ids, attn_mask, labels in iterate_batches(
                tokenizer, ds, args.max_len, args.batch_size, device,
                text_column=args.text_column,
                json_field=args.json_field):
            set_global_step(model, step)
            
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                        labels=labels)
            
            loss = out.loss / accum_steps
            loss.backward()
            
            batch_idx += 1
            if batch_idx % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                
                d = out.jepa_diag
                print(f"[ep{epoch} s{step}] loss={out.loss.item():.4f} "
                      f"ce={out.ce_loss.item():.4f} jepa={d['jepa']:.4f} "
                      f"sigreg={d['sigreg']:.4f} div={d['slot_div']:.4f} "
                      f"sparse={d['slot_sparsity']:.3f} effrank={d['eff_rank']:.1f}",
                      flush=True)
            
            if args.max_steps and step >= args.max_steps:
                break
                
        # End of epoch: step remaining gradients
        if batch_idx % accum_steps != 0 and out is not None:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            
            d = out.jepa_diag
            print(f"[ep{epoch} s{step}] loss={out.loss.item():.4f} "
                  f"ce={out.ce_loss.item():.4f} jepa={d['jepa']:.4f} "
                  f"sigreg={d['sigreg']:.4f} div={d['slot_div']:.4f} "
                  f"sparse={d['slot_sparsity']:.3f} effrank={d['eff_rank']:.1f} (epoch end)",
                  flush=True)
                  
        if args.max_steps and step >= args.max_steps:
            break

    # Save adapters + the new (non-quantized) HOPE/JEPA modules.
    model.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
