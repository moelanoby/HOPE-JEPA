"""End-to-end smoke test: a couple of SSL steps then a couple of linear-probe
steps on a tiny synthetic dataset. No CIFAR download, runs on CPU in seconds.
Proves the whole training+eval path is free of NaN/inf and dimension mismatches.

Run:  python -m tests.test_smoke
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from hope_jepa.model import HopeJepaModel
from hope_jepa.losses import ssl_loss


def _tiny_cfg():
    return {
        "model": {
            "img_size": 32, "patch_size": 4, "d_model": 32, "num_layers": 2,
            "num_heads": 2, "dropout": 0.0,
            "titans": {"num_persistent_memory": 4, "d_hidden": 64, "init_memory_std": 0.02},
            "cms": {"num_modules": 2, "base_update_freq": 1, "d_ff_multiplier": 2},
            "jepa": {"predictor_depth": 1, "mask_ratio": 0.6},
        },
        "sigreg": {"enabled": True, "sketch_dim": 16, "target_scale": 1.0, "weight": 1.0},
    }


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")
    cfg = _tiny_cfg()
    model = HopeJepaModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- 3 SSL steps ---
    for step in range(3):
        img = torch.randn(4, 3, 32, 32)
        opt.zero_grad()
        out = model(img, global_step=step)
        loss, diag = ssl_loss(out, True, cfg["sigreg"]["weight"], model.sigreg)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss).all(), "non-finite SSL loss"
        print(f"  ssl step {step}: total={diag['total']:.4f} jepa={diag['jepa']:.4f} "
              f"sigreg={diag['sigreg']:.4f} eff_rank={diag['eff_rank']:.2f}")

    # --- linear probe: freeze backbone, train a linear head on synthetic labels ---
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    probe = nn.Linear(cfg["model"]["d_model"], 10).to(device)
    popt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    lossfn = nn.CrossEntropyLoss()
    for step in range(3):
        img = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        with torch.no_grad():
            feats = model.encode(img)
        logits = probe(feats)
        loss = lossfn(logits, labels)
        popt.zero_grad()
        loss.backward()
        popt.step()
        assert torch.isfinite(loss).all(), "non-finite probe loss"
        print(f"  probe step {step}: ce={loss.item():.4f}")

    print("\n[PASS] smoke test: SSL + linear-probe path is finite and dimensionally correct.")


if __name__ == "__main__":
    main()
