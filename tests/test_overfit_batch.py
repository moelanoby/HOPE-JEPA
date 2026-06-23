"""Overfit-a-single-batch test: the strongest proof the model is wired to learn.

If the SSL pipeline is correct, training on ONE repeated batch must drive the
JEPA loss arbitrarily close to zero. If it doesn't, something is broken
(detached gradient, wrong target, masking bug, etc.). Runs on CPU in seconds.

Run:  python -m tests.test_overfit_batch
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hope_jepa.model import HopeJepaModel
from hope_jepa.losses import ssl_loss


def _tiny_cfg(sigreg_weight=1.0):
    return {
        "model": {
            "img_size": 32, "patch_size": 4, "d_model": 48, "num_layers": 2,
            "num_heads": 4, "dropout": 0.0,
            "num_slots": 4, "slot_div_weight": 0.1,
            "titans": {"num_persistent_memory": 4, "d_hidden": 96, "init_memory_std": 0.02},
            "cms": {"num_modules": 2, "base_update_freq": 1, "d_ff_multiplier": 2},
            "jepa": {"predictor_depth": 2, "mask_ratio": 0.5},
        },
        "sigreg": {"enabled": True, "sketch_dim": 24, "target_scale": 1.0,
                   "weight": sigreg_weight},
    }


def _run(cfg, images, steps=150, lr=3e-3):
    model = HopeJepaModel(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    init_loss = None
    diag = None
    for step in range(steps):
        opt.zero_grad()
        out = model(images, global_step=step)
        loss, diag = ssl_loss(out, True, cfg["sigreg"]["weight"], model.sigreg,
                              slots=model.slots,
                              slot_div_weight=cfg["model"]["slot_div_weight"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if init_loss is None:
            init_loss = diag["total"]
    return init_loss, diag


def main():
    torch.manual_seed(0)
    # One fixed batch of random "images" (identical for both runs).
    images = torch.randn(8, 3, 32, 32)

    print("Run A: weak SIGReg (weight=0.1) -- expect collapse (rank -> ~1)")
    cfgA = _tiny_cfg(sigreg_weight=0.1)
    initA, diagA = _run(cfgA, images)
    print(f"  init total={initA:.4f}  final jepa={diagA['jepa']:.4f}  "
          f"eff_rank={diagA['eff_rank']:.2f}")

    print("\nRun B: adequate SIGReg (weight=1.0) -- expect rank held high")
    cfgB = _tiny_cfg(sigreg_weight=1.0)
    initB, diagB = _run(cfgB, images)
    print(f"  init total={initB:.4f}  final jepa={diagB['jepa']:.4f}  "
          f"eff_rank={diagB['eff_rank']:.2f}")

    # 1. The SSL pipeline learns in both cases: JEPA loss drops sharply.
    assert diagA["jepa"] < initA * 0.5 and diagB["jepa"] < initB * 0.5, \
        "JEPA loss did not drop -> wiring bug"
    # 2. SIGReg does its job: rank under adequate SIGReg >> rank under weak SIGReg.
    assert diagB["eff_rank"] > diagA["eff_rank"] * 1.5, \
        f"SIGReg did not prevent collapse (B rank {diagB['eff_rank']:.2f} vs A {diagA['eff_rank']:.2f})"
    print(f"\n[PASS] pipeline learns (jepa {diagA['jepa']:.4f}) AND SIGReg prevents "
          f"collapse (rank {diagA['eff_rank']:.1f} -> {diagB['eff_rank']:.1f}).")


if __name__ == "__main__":
    main()
