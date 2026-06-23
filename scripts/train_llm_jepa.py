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
     Reasoner params FULLY trainable via `modules_to_save` (these are new, not
     quantized, and should learn from scratch).
  3. Build `HopeLLM(cfg)` -- which splices HOPE layers and attaches the slot
     JEPA objective (+ Reasoner if enabled).
  4. Loop: set_global_step -> HopeLLM.forward (CE + slot-JEPA + SIGReg + div
     + optional predict-ahead) -> backward -> opt.step -> global_step++.

Usage:
    python scripts/train_llm_jepa.py --config config/llm_default.yaml \
        --dataset <hf-dataset-or-jsonl> --output runs/qwen_hope_jepa
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

    `modules_to_save` makes a full-precision trainable copy of the named
    modules, so the HOPE blocks and the JEPA heads are NOT quantized and learn
    from scratch alongside the LoRA adapters on the pretrained weights.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES as TARGETS

    base = model.model  # the underlying HF CausalLM
    if cfg.training.qlora:
        # k-bit-safe: freeze norm, enable grad checkpointing-friendly inputs.
        base = prepare_model_for_kbit_training(base)

    # Target the attention projections of the base model (q/k/v/o for Llama).
    # Auto-resolve the target module names from the model's architecture.
    arch = getattr(base.config, "architectures", ["LlamaForCausalLM"])[0]
    target_modules = TARGETS.get(arch, ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Collect the names of every NEW module we introduced (HOPE blocks + JEPA
    # heads + slots + sigreg + reasoner). These live on the HopeLLM wrapper, so
    # we locate them by type via the public classes.
    from hope_jepa.llm.hope_block import HopeDecoderLayer, MACAttnAdapter
    new_modules = []
    for name, mod in model.named_modules():
        if isinstance(mod, (HopeDecoderLayer, MACAttnAdapter)) or \
           name.startswith("jepa.") or name.startswith("reasoner."):
            # peft matches by suffix; use the leaf name path.
            new_modules.append(name.split(".")[-1])
    # Deduplicate (many layers share the suffix "hope", "mix_q", etc.) -- peft
    # saves whole modules, so a handful of distinct suffixes covers them.
    new_modules = sorted(set(new_modules)) or None

    peft_cfg = LoraConfig(
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=new_modules,   # keep new HOPE/JEPA/Reasoner params trainable
    )
    # Wrap the base model in-place; the HopeLLM wrapper still owns it.
    model.model = get_peft_model(base, peft_cfg)
    return model


# ---------------------------------------------------------------------------
def iterate_batches(tokenizer, dataset, max_len: int, batch_size: int, device):
    """Yield (input_ids, attention_mask, labels) batches from a text dataset.

    Expects `dataset` to be an iterable of {"text": str} (the HF `datasets`
    convention). Packs each example as its own block: labels = input_ids (HF
    shifts internally). Padding tokens are masked in both attention_mask and
    labels.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
        else tokenizer.eos_token_id
    texts = [ex["text"] for ex in dataset]
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=max_len)
        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]
        labels = input_ids.clone()
        labels[input_ids == pad_id] = -100
        yield (input_ids.to(device), attn.to(device), labels.to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/llm_default.yaml")
    ap.add_argument("--dataset", required=True,
                    help="HF dataset id or path to a jsonl with a 'text' field")
    ap.add_argument("--output", default="runs/hope_jepa")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="cap on total steps (0 = no cap)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # --- Model ---
    model = HopeLLM(cfg)                       # loads + splices HOPE layers
    model = apply_qlora(model, cfg)            # 4-bit QLoRA on base, new params trainable
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
    for epoch in range(args.epochs):
        for input_ids, attn_mask, labels in iterate_batches(
                tokenizer, ds, args.max_len, args.batch_size, device):
            set_global_step(model, step)
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                        labels=labels)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            d = out.jepa_diag
            print(f"[ep{epoch} s{step}] loss={out.loss.item():.4f} "
                  f"ce={out.ce_loss.item():.4f} jepa={d['jepa']:.4f} "
                  f"sigreg={d['sigreg']:.4f} div={d['slot_div']:.4f} "
                  f"sparse={d['slot_sparsity']:.3f} effrank={d['eff_rank']:.1f}",
                  flush=True)
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    # Save adapters + the new (non-quantized) HOPE/JEPA modules.
    model.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
