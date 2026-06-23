"""SSL pretraining for HOPE-JEPA-SIGReg.

Trains the backbone with the per-layer JEPA objective + SIGReg, using NO labels.
Logs JEPA loss, the SIGReg penalty, and the effective rank of the embedding
covariance (the collapse monitor). Saves checkpoints to out_dir/checkpoints/.

Usage:
  python train_ssl.py --config config/tiny.yaml
  python train_ssl.py --config config/tiny.yaml --no-sigreg   # ablation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hope_jepa import HopeJepaModel
from hope_jepa.data import build_ssl_loaders
from hope_jepa.losses import ssl_loss
from hope_jepa.utils import (
    count_parameters, cosine_lr, load_config, make_run_dir,
    resolve_device, seed_everything,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--no-sigreg", action="store_true",
                   help="Disable SIGReg (ablation: expect collapse).")
    p.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.no_sigreg:
        cfg["sigreg"]["enabled"] = False
        cfg["run_name"] = cfg.get("run_name", "run") + "_nosigreg"

    seed_everything(cfg.get("seed", 0))
    device = resolve_device(cfg.get("device", "auto"))
    out_dir = make_run_dir(cfg)
    print(f"== HOPE-JEPA-SIGReg SSL pretraining ==")
    print(f"config: {args.config}  (SIGReg enabled: {cfg['sigreg']['enabled']})")
    print(f"device: {device}  out_dir: {out_dir}")

    train_loader, _ = build_ssl_loaders(cfg)
    model = HopeJepaModel(cfg).to(device)
    print(f"trainable params: {count_parameters(model):,}")

    s = cfg["ssl"]
    total_steps = max(1, len(train_loader) * s["epochs"])
    warmup_steps = len(train_loader) * s.get("warmup_epochs", 0)
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"],
                            weight_decay=s.get("weight_decay", 0.0))

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    log_path = os.path.join(out_dir, "ssl_log.jsonl")
    log_every = cfg.get("log_every", 100)
    sig_w = float(cfg["sigreg"]["weight"])
    sigreg_on = bool(cfg["sigreg"]["enabled"])
    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, s["epochs"]):
        for view1, view2, _ in train_loader:
            view1 = view1.to(device, non_blocking=True)
            # JEPA is single-view here (context & target drawn from the same
            # view's patch embeddings); the second view is reserved for a
            # future multi-view extension. We train on view1 to keep the
            # objective unambiguous.
            for g in opt.param_groups:
                g["lr"] = cosine_lr(global_step, total_steps, s["lr"], warmup_steps)

            opt.zero_grad()
            out = model(view1, global_step=global_step)
            loss, diag = ssl_loss(out, sigreg_on, sig_w, model.sigreg,
                                  slots=model.slots,
                                  slot_div_weight=model.slot_div_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if global_step % log_every == 0:
                elapsed = time.time() - t0
                print(f"  ep {epoch} step {global_step:5d}/{total_steps}  "
                      f"jepa={diag['jepa']:.4f} sigreg={diag['sigreg']:.4f} "
                      f"div={diag['slot_div']:.4f} sparsity={diag['slot_sparsity']:.2f} "
                      f"eff_rank={diag['eff_rank']:.2f} lr={g['lr']:.2e} "
                      f"({elapsed:.0f}s)")
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": global_step, **diag}) + "\n")
            global_step += 1

        # checkpoint each epoch
        ckpt_path = os.path.join(out_dir, "checkpoints", f"ssl_epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                    "epoch": epoch, "global_step": global_step, "cfg": cfg},
                   ckpt_path)
        print(f"  [epoch {epoch} done] saved {ckpt_path}")

    final = os.path.join(out_dir, "checkpoints", "ssl_final.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg}, final)
    print(f"\nSSL pretraining complete. Final checkpoint: {final}")
    print("Next: run the linear probe ->  python linear_probe.py "
          f"--config {args.config} --checkpoint {final}")


if __name__ == "__main__":
    main()
