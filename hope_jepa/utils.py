"""Small shared utilities: config loading, seeding, lr scheduling, device."""

from __future__ import annotations

import math
import os
import random

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int = 0) -> float:
    """Cosine schedule with linear warmup. Returns the lr multiplier behavior as
    a concrete lr value given `base_lr`."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def make_run_dir(cfg: dict) -> str:
    out = cfg.get("out_dir", "./runs/default")
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    return out
