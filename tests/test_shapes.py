"""Shape + gradient-connectivity tests for every module.

Run:  python -m tests.test_shapes
These run on CPU in a couple of seconds and verify:
  - patchify produces the expected token count,
  - the Titans memory recurrence is gradient-connected through time (no detached
    state that would block BPTT),
  - a HOPE layer preserves shape,
  - JEPA predictor + layer loss produce correct shapes and a scalar loss,
  - the full model forward returns the advertised dict.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hope_jepa.data import patchify, random_mask
from hope_jepa.titans import NeuralLongTermMemory, MACMixer
from hope_jepa.hope import HopeLayer
from hope_jepa.jepa import JEPAPredictor, jepa_layer_loss
from hope_jepa.sigreg import SIGReg
from hope_jepa.model import HopeJepaModel


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


def test_patchify():
    img = torch.randn(2, 3, 32, 32)
    p = patchify(img, 4)
    assert p.shape == (2, 64, 48), p.shape  # 64 patches, 3*4*4 = 48 dims
    print("  [ok] patchify ->", tuple(p.shape))


def test_mask():
    m = random_mask(3, 64, 0.5, torch.device("cpu"))
    assert m.shape == (3, 64) and m.dtype == torch.bool
    frac = m.float().mean().item()
    assert 0.4 < frac < 0.6, frac
    print(f"  [ok] random_mask mask ratio ~{frac:.2f}")


def test_titans_grad():
    """Verify BPTT gradients reach the initial memory parameter."""
    mem = NeuralLongTermMemory(32, 64)
    tokens = torch.randn(2, 8, 32, requires_grad=True)
    retrieval, final_M = mem(tokens)
    assert retrieval.shape == (2, 8, 32)
    assert final_M.shape == (2, 32, 32)
    retrieval.sum().backward()
    assert mem.M0.grad is not None, "BPTT did not reach the memory init parameter"
    assert tokens.grad is not None and tokens.grad.abs().sum() > 0
    print("  [ok] Titans BPTT grad reaches M0 and input")


def test_mac_mixer():
    mixer = MACMixer(32, 64, num_persistent_memory=4)
    x = torch.randn(2, 64, 32, requires_grad=True)
    y = mixer(x)
    assert y.shape == x.shape, y.shape
    y.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    print("  [ok] MACMixer preserves shape and is differentiable")


def test_hope_layer():
    layer = HopeLayer(32, 64, 4, 2, 1, 2, dropout=0.0)
    x = torch.randn(2, 64, 32, requires_grad=True)
    y = layer(x, global_step=0)
    assert y.shape == x.shape, y.shape
    y.sum().backward()
    assert x.grad is not None
    # also exercise a non-zero step so off-cadence NLMs activate
    _ = layer(x, global_step=1)
    print("  [ok] HopeLayer shape ok at steps 0 and 1")


def test_jepa():
    d = 32
    pred = JEPAPredictor(d, num_heads=2, depth=1)
    z = torch.randn(2, 64, d)
    mask = random_mask(2, 64, 0.6, torch.device("cpu"))
    loss = jepa_layer_loss(z, mask, pred)
    assert loss.ndim == 0, "layer loss must be scalar"
    loss.backward()
    # predictor params should have grad
    g = sum(p.grad.abs().sum().item() for p in pred.parameters() if p.grad is not None)
    assert g > 0, "no grads reached the predictor"
    print(f"  [ok] JEPA layer loss = {loss.item():.4f}, grads flow")


def test_sigreg_extremes():
    sr = SIGReg(32, sketch_dim=16, target_scale=1.0)
    # collapsed: all rows identical -> rank-deficient covariance -> large penalty
    base = torch.randn(1, 32)
    collapsed = base.expand(512, -1)
    # isotropic-ish: iid gaussian rows with unit variance
    isotropic = torch.randn(512, 32)
    pc = sr(collapsed).item()
    pi = sr(isotropic).item()
    print(f"  [ok] SIGReg collapsed={pc:.3f} isotropic={pi:.3f} (collapsed should be >= isotropic)")
    assert pc >= pi, "collapsed penalty should be >= isotropic penalty"
    er = sr.effective_rank(isotropic)
    assert er > 5, f"effective rank of isotropic batch too low: {er}"
    print(f"  [ok] effective_rank(isotropic) = {er:.2f}")


def test_full_model():
    cfg = _tiny_cfg()
    model = HopeJepaModel(cfg)
    img = torch.randn(2, 3, 32, 32)
    out = model(img, global_step=0)
    assert "layer_embeddings" in out and len(out["layer_embeddings"]) == 2
    assert out["layer_embeddings"][0].shape == (2, 65, 32)  # 64 patches + CLS
    assert out["pooled"].shape == (2, 32)
    assert out["mask"].shape == (2, 64)
    assert len(out["jepa_losses"]) == 2
    print("  [ok] full model forward returns expected dict")


def main():
    tests = [
        ("patchify", test_patchify),
        ("mask", test_mask),
        ("titans_grad", test_titans_grad),
        ("mac_mixer", test_mac_mixer),
        ("hope_layer", test_hope_layer),
        ("jepa", test_jepa),
        ("sigreg_extremes", test_sigreg_extremes),
        ("full_model", test_full_model),
    ]
    print("Running shape/grad tests...")
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            raise
    print("\nAll shape/grad tests passed.")


if __name__ == "__main__":
    main()
