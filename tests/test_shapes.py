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
from hope_jepa.entmax import entmax15
from hope_jepa.slots import (
    SlotJEPAPredictor, SlotReadout, jepa_slot_layer_loss, slot_divergence_loss,
)


def _tiny_cfg():
    return {
        "model": {
            "img_size": 32, "patch_size": 4, "d_model": 32, "num_layers": 2,
            "num_heads": 2, "dropout": 0.0,
            "num_slots": 4, "slot_div_weight": 0.1,
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

    # Test with variable mask lengths (different Nt per batch item)
    mask_var = torch.zeros(2, 64, dtype=torch.bool)
    mask_var[0, :10] = True
    mask_var[1, :15] = True
    pred.zero_grad()
    loss_var = jepa_layer_loss(z, mask_var, pred)
    assert loss_var.ndim == 0
    loss_var.backward()
    g_var = sum(p.grad.abs().sum().item() for p in pred.parameters() if p.grad is not None)
    assert g_var > 0
    print(f"  [ok] JEPA layer loss = {loss.item():.4f} (var={loss_var.item():.4f}), grads flow")



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


def test_entmax():
    """entmax15 is on the simplex, can be exactly 0, and is differentiable."""
    # Peaked logits -> one slot gets ~1, the rest should be exactly 0.
    peaked = torch.tensor([[10.0, 0.0, -5.0, 0.0]])
    w = entmax15(peaked, dim=-1)
    s = w.sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4), "entmax must sum to 1"
    n_zeros = int((w == 0).sum().item())
    assert n_zeros >= 2, f"entmax should produce exact zeros on peaked logits, got {w}"
    # Uniform logits -> uniform output (no zeros).
    u = entmax15(torch.zeros(1, 4), dim=-1)
    assert torch.allclose(u, torch.full((1, 4), 0.25), atol=1e-3), u
    # Gradient: use a multi-element-support case + a non-degenerate objective.
    # (Note: d/dz of sum(p) is structurally 0 since p is on the simplex, and a
    # single-element support is pinned at weight 1 -> zero grad. So we pick a
    # case with >=2 support elements and a weighted objective.)
    logits = torch.tensor([[1.5, 1.0, -1.0, -1.0]], requires_grad=True)
    w = entmax15(logits, dim=-1)
    coeff = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    (w * coeff).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum().item() > 0, "no grad"
    # Off-support entries (weight==0) must receive exactly zero gradient.
    off = (w == 0)
    if off.any():
        assert logits.grad[off].abs().max().item() < 1e-7, \
            f"off-support grad must be zero, got {logits.grad[off]}"
    print(f"  [ok] entmax15: simplex, exact zeros (n={n_zeros}), uniform->spread, grads flow")


def test_slots():
    """Slot JEPA predictor, layer loss, divergence, and readout are correct."""
    d, K = 32, 4
    slots = torch.randn(K, d, requires_grad=True)
    pred = SlotJEPAPredictor(d, num_heads=2, num_slots=K, depth=1)
    z = torch.randn(2, 64, d)
    mask = random_mask(2, 64, 0.6, torch.device("cpu"))
    loss, w = jepa_slot_layer_loss(z, mask, pred, slots)
    assert loss.ndim == 0, "slot layer loss must be scalar"
    assert w.dim() == 3 and w.shape[-1] == K, w.shape           # [n, Nt, K]
    loss.backward()
    # Gradients must reach both the slot heads and the shared slot embeddings.
    head_grad = sum(p.grad.abs().sum().item() for p in pred.parameters() if p.grad is not None)
    assert head_grad > 0, "no grads reached the slot predictor"
    assert slots.grad is not None and slots.grad.abs().sum().item() > 0, \
        "no grads reached the shared slots through mixing"

    # Test with variable mask lengths (different Nt per batch item)
    mask_var = torch.zeros(2, 64, dtype=torch.bool)
    mask_var[0, :10] = True
    mask_var[1, :15] = True
    pred.zero_grad()
    if slots.grad is not None:
        slots.grad.zero_()
        slots.grad = None
    loss_var, w_var = jepa_slot_layer_loss(z, mask_var, pred, slots)
    assert loss_var.ndim == 0
    assert w_var.dim() == 3 and w_var.shape[-1] == K, w_var.shape
    loss_var.backward()
    head_grad_var = sum(p.grad.abs().sum().item() for p in pred.parameters() if p.grad is not None)
    assert head_grad_var > 0
    assert slots.grad is not None and slots.grad.abs().sum().item() > 0


    # Divergence: correlated slots -> high, orthogonal slots -> ~0.
    base = torch.randn(1, d)
    correlated = base.expand(4, -1)
    dc = slot_divergence_loss(correlated).item()
    orth = torch.linalg.qr(torch.randn(d, 4))[0].t()      # 4 orthonormal rows
    do = slot_divergence_loss(orth).item()
    assert dc > 0.5 and do < 1e-4, f"divergence correlated={dc:.3f} orthogonal={do:.3f}"

    # Readout: [B, N+1, d] + slots -> [B, d], differentiable.
    readout = SlotReadout(d, K)
    z_full = torch.randn(2, 65, d, requires_grad=True)
    pooled = readout(z_full, slots)
    assert pooled.shape == (2, d), pooled.shape
    pooled.sum().backward()
    assert z_full.grad is not None and z_full.grad.abs().sum().item() > 0, "readout not differentiable"
    print(f"  [ok] slot predictor/loss/divergence/readout: div corr={dc:.2f} orth={do:.2e}")


def test_full_model():
    cfg = _tiny_cfg()
    model = HopeJepaModel(cfg)
    img = torch.randn(2, 3, 32, 32)
    out = model(img, global_step=0)
    assert "layer_embeddings" in out and len(out["layer_embeddings"]) == 2
    assert out["layer_embeddings"][0].shape == (2, 65, 32)  # 64 patches + CLS
    assert out["pooled"].shape == (2, 32)                  # slot-assembled readout
    assert out["mask"].shape == (2, 64)
    assert len(out["jepa_losses"]) == 2
    assert "slot_weights" in out and len(out["slot_weights"]) == 2
    sw = out["slot_weights"][0]
    assert sw.dim() == 3 and sw.shape[-1] == 4             # [n, Nt, K] per layer
    # Sparsity is genuinely non-trivial at init (entmax drops slots); this is
    # the core "can choose 0 for something" property the slot system provides.
    assert (sw == 0).float().mean().item() > 0.0, "no slots dropped at init"
    print("  [ok] full model forward returns expected dict "
          f"(sparsity={(sw==0).float().mean():.2f})")


def main():
    torch.manual_seed(42)
    tests = [
        ("patchify", test_patchify),
        ("mask", test_mask),
        ("titans_grad", test_titans_grad),
        ("mac_mixer", test_mac_mixer),
        ("hope_layer", test_hope_layer),
        ("jepa", test_jepa),
        ("entmax", test_entmax),
        ("slots", test_slots),
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
