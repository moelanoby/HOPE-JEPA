"""Linear probe evaluation: freeze the SSL backbone, train a single linear
classifier on its pooled embedding, report top-1 accuracy.

This is the canonical JEPA evaluation protocol: a good SSL representation needs
only a linear head to reach high accuracy. Comparing SIGReg-on vs SIGReg-off
(ablation) gives the headline result -- SIGReg prevents collapse, so the
SIGReg-on backbone should probe much better.

Usage:
  python linear_probe.py --config config/tiny.yaml \
      --checkpoint runs/tiny/checkpoints/ssl_final.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hope_jepa import HopeJepaModel
from hope_jepa.data import build_probe_loaders, num_classes
from hope_jepa.utils import (
    cosine_lr, count_parameters, load_config, make_run_dir,
    resolve_device, seed_everything,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="SSL checkpoint to load.")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, probe, loader, device):
    model.eval()
    probe.eval()
    correct, total = 0, 0
    for img, label in loader:
        img = img.to(device)
        label = label.to(device)
        feats = model.encode(img)
        logits = probe(feats)
        pred = logits.argmax(dim=-1)
        correct += (pred == label).sum().item()
        total += label.numel()
    return correct / max(total, 1)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 0))
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = make_run_dir(cfg)

    print(f"== Linear-probe evaluation ==")
    print(f"checkpoint: {args.checkpoint}")

    train_loader, val_loader = build_probe_loaders(cfg)
    model = HopeJepaModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    # Freeze the entire backbone.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    d = cfg["model"]["d_model"]
    n_cls = num_classes(cfg["data"]["dataset"])
    probe = nn.Linear(d, n_cls).to(device)
    print(f"probe params: {count_parameters(probe):,}  (backbone frozen)")

    s = cfg["probe"]
    total_steps = max(1, len(train_loader) * s["epochs"])
    opt = torch.optim.AdamW(probe.parameters(), lr=s["lr"],
                            weight_decay=s.get("weight_decay", 0.0))
    lossfn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(s["epochs"]):
        probe.train()
        for step, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)
            for g in opt.param_groups:
                g["lr"] = cosine_lr(epoch * len(train_loader) + step,
                                    total_steps, s["lr"])
            opt.zero_grad()
            with torch.no_grad():
                feats = model.encode(img)
            logits = probe(feats)
            loss = lossfn(logits, label)
            loss.backward()
            opt.step()

        acc = evaluate(model, probe, val_loader, device)
        best_acc = max(best_acc, acc)
        if epoch % 5 == 0 or epoch == s["epochs"] - 1:
            print(f"  epoch {epoch:3d}  val_acc={acc:.4f}  best={best_acc:.4f}")

    print(f"\nLinear-probe best top-1 accuracy: {best_acc*100:.2f}%")
    print(f"(random baseline = {100/n_cls:.2f}% for {n_cls}-way)")
    return best_acc


if __name__ == "__main__":
    main()
