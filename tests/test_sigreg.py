"""Focused SIGReg behavior test.

Verifies:
  1. The penalty is ~0 on a perfectly isotropic unit-covariance batch.
  2. The penalty is large on a rank-deficient (collapsed) batch.
  3. Gradient descent on SIGReg alone spreads a collapsed batch's effective
     rank upward -- the core mechanism that prevents JEPA collapse.

Run:  python -m tests.test_sigreg
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hope_jepa.sigreg import SIGReg


def main():
    torch.manual_seed(0)
    d = 32
    sr = SIGReg(d, sketch_dim=d, target_scale=1.0)  # full covariance for a clean test

    # 1. Isotropic: draw from N(0, I) with enough samples -> cov ~ I -> low penalty.
    iso = torch.randn(4096, d, requires_grad=True)
    p_iso = sr(iso).item()
    print(f"  isotropic penalty = {p_iso:.4f} (expect small)")

    # 2. Collapsed: all rows equal -> zero variance off the mean -> huge penalty.
    base = torch.randn(1, d)
    coll = base.expand(4096, -1).clone().requires_grad_(True)
    p_coll = sr(coll).item()
    print(f"  collapsed penalty = {p_coll:.4f} (expect >> isotropic)")
    assert p_coll > p_iso, "collapsed should penalize more than isotropic"

    # 3. Optimize a collapsed batch toward isotropy using SIGReg alone.
    x = torch.randn(512, d)
    x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
    # collapse it: project onto a random low-rank subspace
    V = torch.linalg.qr(torch.randn(d, 2))[0][:, :2]
    x = (x @ V) @ V.t()           # rank-2
    x = x.clone().requires_grad_(True)
    er0 = sr.effective_rank(x.detach())
    opt = torch.optim.Adam([x], lr=5e-2)
    for _ in range(200):
        opt.zero_grad()
        loss = sr(x)
        loss.backward()
        opt.step()
    er1 = sr.effective_rank(x.detach())
    print(f"  effective rank: collapsed={er0:.2f} -> after SIGReg opt={er1:.2f}")
    assert er1 > er0 * 1.5, "SIGReg optimization did not increase effective rank"
    print(f"  [PASS] SIGReg pushes effective rank from {er0:.2f} to {er1:.2f}")


if __name__ == "__main__":
    main()
