"""EGGROLL self-play RL entrypoint (the SECOND training phase).

Loads the HOPE+JEPA center model theta (from `model.hope_config`, the same
config `scripts/train_llm_jepa.py` uses) and evolves it via EGGROLL self-play:
3 injectors make bugs in real repo snapshots, 12 fixers (6 bare + 6 tool-augmented)
race to fix them, validated against the repo's real test suite. The center is
updated by rank-shaped EGGROLL evolution strategies -- no gradients, no critic.

This mirrors the structure of `scripts/train_llm_jepa.py`: argparse, a YAML
config, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set before CUDA init,
and a model built from an llm_default-style config. On CPU it builds a tiny
model (no download) so the path is smoke-testable.

Usage:
    # Point at one or many repos (local path or git URL) and run 50 generations:
    python scripts/train_eggroll.py --config config/eggroll_default.yaml \
        --repos ./my_repo --output runs/eggroll/run1

    # Multiple repos:
    python scripts/train_eggroll.py --config config/eggroll_default.yaml \
        --repos ./repo_a ./repo_b https://github.com/owner/repo_c.git

    # CPU smoke (tiny model, a couple of generations, no GPU):
    python scripts/train_eggroll.py --config config/eggroll_default.yaml \
        --repos ./sample_repo --generations 2 --cpu

WARNING: agents emit code (patches + tools) that is executed inside a sandboxed
cwd (no network env, CPU/time capped). That is defense-in-depth, NOT a hard
boundary -- only point --repos at trusted code, and ideally run inside a
container.
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml

# Same fragmentation guard as train_llm_jepa.py -- set before CUDA init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Run from repo root so `hope_jepa` imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hope_jepa.llm import HopeLLM, HopeLlmConfig
from hope_jepa.rl import EggrollConfig, EggrollTrainer


def parse_args():
    ap = argparse.ArgumentParser(description="EGGROLL self-play RL phase")
    ap.add_argument("--config", default="config/eggroll_default.yaml")
    ap.add_argument("--repos", nargs="*", default=None,
                    help="Override the repos list (1..N local paths or git URLs)")
    ap.add_argument("--output", default="runs/eggroll")
    ap.add_argument("--generations", type=int, default=None,
                    help="Override train.generations")
    ap.add_argument("--max_turns", type=int, default=None,
                    help="Override train.max_turns (fixer action budget)")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU + tiny model (smoke path, no download)")
    ap.add_argument("--no_tools", action="store_true",
                    help="Disable tool authoring (all 12 fixers become bare)")
    return ap.parse_args()


def load_config(path: str) -> EggrollConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return EggrollConfig.from_dict(raw)


def build_center_model(cfg: EggrollConfig, cpu: bool):
    """Build the center HopeLLM theta from `cfg.model.hope_config`.

    On GPU we honor the model's quantize setting (4bit QLoRA). On CPU we build a
    tiny model from a LlamaConfig (no download) so the path runs without a GPU.
    """
    hope_yaml = cfg.model.hope_config
    with open(hope_yaml) as f:
        hope_dict = yaml.safe_load(f)
    hope_cfg = HopeLlmConfig.from_dict(hope_dict)

    if cpu:
        # Tiny CPU model: build directly from a LlamaConfig, no from_pretrained.
        from transformers import LlamaConfig, LlamaForCausalLM
        lc = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=4, max_position_embeddings=128)
        base = LlamaForCausalLM(lc)
        hope_cfg.model_id = "tiny-eggroll"
        model = HopeLLM(hope_cfg, model=base)
        # Make everything trainable (no QLoRA on CPU smoke).
        for p in model.parameters():
            p.requires_grad_(True)
        return model

    # GPU path: full build (HOPE splice + optional 4-bit QLoRA).
    # Reuse the same apply_qlora path as train_llm_jepa for consistency.
    model = HopeLLM(hope_cfg)
    if hope_cfg.training.qlora and cfg.model.quantize == "4bit":
        from scripts.train_llm_jepa import apply_qlora  # noqa: WPS433
        model = apply_qlora(model, hope_cfg)
    return model


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides.
    if args.repos is not None:
        cfg.repos.repos = args.repos
    if args.generations is not None:
        cfg.train.generations = args.generations
    if args.max_turns is not None:
        cfg.train.max_turns = args.max_turns
    if args.no_tools:
        cfg.roles.tool_fixers = 0
        cfg.roles.bare_fixers = cfg.roles.num_fixers

    if not cfg.repos.repos:
        sys.exit("error: no repos configured. Pass --repos PATH or set repos: "
                 "in the config. EGGROLL needs at least one repo as the arena.")

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"== EGGROLL self-play RL ==")
    print(f"config: {args.config}  device: {device}")
    print(f"repos: {cfg.repos.repos}")
    print(f"roles: {cfg.roles.num_injectors} injectors vs "
          f"{cfg.roles.num_fixers} fixers "
          f"({cfg.roles.tool_fixers} tool, {cfg.roles.bare_fixers} bare)")
    print(f"eggroll: rank={cfg.eggroll.rank} sigma={cfg.eggroll.sigma} "
          f"lr={cfg.eggroll.lr} utility={cfg.eggroll.utility_shape}")
    print(f"generations: {cfg.train.generations}  max_turns: {cfg.train.max_turns}")

    center_model = build_center_model(cfg, cpu=(device == "cpu"))
    center_model.to(device)

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            yaml.safe_load(open(cfg.model.hope_config))["model_id"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:                      # noqa: BLE001 - CPU smoke has no tokenizer
        tokenizer = None

    trainer = EggrollTrainer(cfg, center_model, tokenizer, device)
    history = trainer.train(output_dir=args.output)

    print(f"\nEGGROLL done. {len(history)} generations. "
          f"Final best fitness={history[-1].best_fitness:.4f} "
          f"({history[-1].best_role}). Center saved to {args.output}")


if __name__ == "__main__":
    main()
