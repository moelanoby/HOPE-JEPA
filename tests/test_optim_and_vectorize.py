"""Tests for the low-memory optimizers (LOMO / AdaLomo / LISA) and the
vectorized JEPA compute path.

These are TORCH-ONLY (no transformers / GPU needed) so they run anywhere
torch is installed. They are the correctness gate for the throughput work:

  * `_masked_random_mask`        -- per-row target COUNT matches the old
                                    per-example semantics exactly.
  * `jepa_slot_layer_loss`       -- the new BATCHED loss is numerically
                                    equivalent to the old per-example loop
                                    (same loss + same gradients, to fp tolerance).
  * `LOMO` / `AdaLomo`           -- params update during backward, grads are
                                    freed, clipping is applied.
  * `LISA`                       -- only the active layer subset updates; the
                                    frozen subset keeps its weights and gets no
                                    optimizer state.

Run with:
    python -m tests.test_optim_and_vectorize
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hope_jepa.slots import SlotJEPAPredictor, jepa_slot_layer_loss
from hope_jepa.llm.jepa_llm import _masked_random_mask
from hope_jepa.optim import LOMO, AdaLomo, LISA


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _tiny_predictor(d=16, K=3):
    return SlotJEPAPredictor(d_model=d, num_heads=2, num_slots=K, depth=1)


# ---------------------------------------------------------------------------
# 1. _masked_random_mask: per-row target count matches the old semantics.
# ---------------------------------------------------------------------------
def test_masked_random_mask_counts():
    """Each row must select exactly max(1, min(n_i-1, round(n_i*ratio))) valid
    positions, and only ever valid positions; rows with <2 valid get none."""
    torch.manual_seed(1)
    B, T, ratio = 5, 20, 0.4
    # Variable-length valid regions (simulate padding).
    valid = torch.tensor([
        [1]*20,            # n=20 -> round(8.0)=8 targets
        [1]*15 + [0]*5,    # n=15 -> round(6.0)=6
        [1]*10 + [0]*10,   # n=10 -> round(4.0)=4
        [1]*2 + [0]*18,    # n=2  -> min(1, 1)=1
        [0]*20,            # n=0  -> 0
    ], dtype=torch.bool)
    mask = _masked_random_mask(B, T, ratio, valid, device="cpu")

    # Expected per-row counts.
    n_valid = valid.sum(1)
    expected = []
    for nv in n_valid.tolist():
        if nv < 2:
            expected.append(0)
        else:
            nm = max(1, min(nv - 1, round(nv * ratio)))
            expected.append(nm)

    counts = mask.sum(1).tolist()
    assert counts == expected, f"counts {counts} != expected {expected}"

    # Targets must lie within valid positions only.
    assert not (mask & ~valid).any(), "a target was placed on a PAD position"
    print(f"[ok] _masked_random_mask counts per row: {counts} == {expected}")


def test_masked_random_mask_distribution():
    """Over many draws, every valid position in a row is chosen ~uniformly."""
    B, T, ratio = 1, 40, 0.5   # n=40, n_mask=20 each draw
    valid = torch.ones(B, T, dtype=torch.bool)
    hits = torch.zeros(T)
    n_draws = 2000
    for _ in range(n_draws):
        m = _masked_random_mask(B, T, ratio, valid, device="cpu")
        hits += m.sum(0)
    # Each position is selected in `ratio` (=0.5) of draws on average, since
    # every draw picks 20 of the 40 positions uniformly. freq = hits / n_draws.
    freq = hits / n_draws
    # Every position should be picked ~50% of the time (loose bounds).
    assert freq.min() > 0.40 and freq.max() < 0.60, \
        f"sampling not uniform: min={freq.min():.3f} max={freq.max():.3f}"
    print(f"[ok] _masked_random_mask uniform: freq in "
          f"[{freq.min():.3f}, {freq.max():.3f}]")


# ---------------------------------------------------------------------------
# 2. jepa_slot_layer_loss: batched == per-example loop (the real correctness gate).
# ---------------------------------------------------------------------------
def _per_example_loss(z, mask, predictor, slots):
    """A faithful re-implementation of the OLD per-example `jepa_slot_layer_loss`
    (before vectorization) so we can check the new batched version against it."""
    B, N, d = z.shape
    preds_list, targets_list, weights_list = [], [], []
    for i in range(B):
        m = mask[i]
        tpos = torch.nonzero(m, as_tuple=False).squeeze(-1)
        cpos = torch.nonzero(~m, as_tuple=False).squeeze(-1)
        if tpos.numel() == 0 or cpos.numel() == 0:
            continue
        zc = z[i:i + 1, cpos, :]
        pred, w = predictor(zc, cpos.unsqueeze(0), tpos.unsqueeze(0), slots)
        # Match the production code: JEPA targets are stop-gradient (standard
        # JEPA practice + a major memory win). Keep the reference identical so
        # the test stays a valid batching-equivalence check.
        tgt = z[i:i + 1, tpos, :].detach()
        preds_list.append(pred)
        targets_list.append(tgt)
        weights_list.append(w)
    if not preds_list:
        return z.new_zeros(())
    if all(p.shape == preds_list[0].shape for p in preds_list):
        preds = torch.cat(preds_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
    else:
        preds = torch.cat([p.reshape(-1, d) for p in preds_list], dim=0)
        targets = torch.cat([t.reshape(-1, d) for t in targets_list], dim=0)
    return nn.functional.mse_loss(preds, targets)


def test_jepa_loss_matches_per_example():
    """The batched loss == the old per-example loss (same weights), to fp tol."""
    torch.manual_seed(42)
    d, K, B, N = 16, 3, 4, 14
    pred = _tiny_predictor(d, K)
    slots = nn.Parameter(torch.randn(K, d) * 0.02)
    z = torch.randn(B, N, d)

    # Variable masks per row (different #targets each) -- exercises padding.
    torch.manual_seed(7)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for i in range(B):
        n_pick = torch.randint(2, 7, ()).item()
        idx = torch.randperm(N)[:n_pick]
        mask[i, idx] = True

    loss_batched, _ = jepa_slot_layer_loss(z, mask, pred, slots)
    loss_perex = _per_example_loss(z, mask, pred, slots)
    diff = float((loss_batched.detach() - loss_perex.detach()).abs())
    assert diff < 1e-5, f"batched {loss_batched:.6f} != per-example {loss_perex:.6f} (diff {diff:.2e})"
    print(f"[ok] batched JEPA loss matches per-example: "
          f"{float(loss_batched):.6f} vs {float(loss_perex):.6f} (diff {diff:.2e})")


def test_jepa_loss_grad_matches_per_example():
    """Gradients of the batched loss match the per-example gradients."""
    torch.manual_seed(42)
    d, K, B, N = 16, 3, 4, 14
    slots = nn.Parameter(torch.randn(K, d) * 0.02)
    z = torch.randn(B, N, d)
    torch.manual_seed(7)
    mask = torch.zeros(B, N, dtype=torch.bool)
    for i in range(B):
        n_pick = torch.randint(2, 7, ()).item()
        mask[i, torch.randperm(N)[:n_pick]] = True

    # --- batched ---
    pred_b = _tiny_predictor(d, K)
    pred_b.load_state_dict(_tiny_predictor(d, K).state_dict())  # distinct fresh
    pred_b2 = _tiny_predictor(d, K); pred_b2.load_state_dict(pred_b.state_dict())
    slots_b = nn.Parameter(slots.detach().clone())
    loss_b, _ = jepa_slot_layer_loss(z, mask, pred_b2, slots_b)
    loss_b.backward()
    g_b = pred_b2.mask_token.grad.clone()

    # --- per-example (clone identical weights) ---
    pred_p = _tiny_predictor(d, K); pred_p.load_state_dict(pred_b.state_dict())
    slots_p = nn.Parameter(slots.detach().clone())
    loss_p = _per_example_loss(z, mask, pred_p, slots_p)
    loss_p.backward()
    g_p = pred_p.mask_token.grad.clone()

    gd = float((g_b - g_p).abs().max())
    assert gd < 1e-5, f"mask_token grad differs by {gd:.2e}"
    # Slot grads too.
    sd = float((slots_b.grad - slots_p.grad).abs().max())
    assert sd < 1e-5, f"slots grad differs by {sd:.2e}"
    print(f"[ok] batched JEPA grads match per-example: "
          f"mask_token d={gd:.2e}, slots d={sd:.2e}")


def test_jepa_loss_empty_rows():
    """Rows with <2 valid tokens are skipped without error."""
    torch.manual_seed(0)
    d, K = 16, 3
    pred = _tiny_predictor(d, K)
    slots = nn.Parameter(torch.randn(K, d) * 0.02)
    B, N = 3, 10
    z = torch.randn(B, N, d)
    mask = torch.zeros(B, N, dtype=torch.bool)
    # Row 0: all-target (no context) -> skipped; row 1: one target; row 2: normal
    mask[0] = True
    mask[1, 0] = True
    mask[2, [0, 2, 4, 6]] = True
    loss, w = jepa_slot_layer_loss(z, mask, pred, slots)
    assert torch.isfinite(loss), "loss not finite with edge-case masks"
    loss.backward()
    print(f"[ok] edge-case masks (empty/1-target rows) handled: loss={float(loss):.4f}")


# ---------------------------------------------------------------------------
# 3. LOMO: fused SGD update with global clip, grads freed.
# ---------------------------------------------------------------------------
def test_lomo_updates_and_clips():
    """LOMO: a 2-pass backward updates params by the clipped gradient, and frees
    the grad afterward."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([3.0, -4.0, 0.0]))   # ||g|| will be large
    opt = LOMO([p], lr=0.1, clip_grad=1.0)
    loss = (p * p).sum()       # dL/dp = 2p = [6, -8, 0], global norm = 10
    # Dry pass: measure global norm. retain_graph=True so the graph survives the
    # live pass. Do NOT zero_grad between dry and live (that would wipe the
    # measured norm) -- zero_grad is the "start of a fresh step" signal.
    opt.zero_grad(); loss.backward(retain_graph=True)
    opt.begin_fused_update()
    # Live pass: apply clipped update (hooks fire the fused step), then step()
    # clears the leftover grad.
    loss.backward()
    opt.step()
    before = torch.tensor([3.0, -4.0, 0.0])
    # clip = 1.0, global norm = 10 -> scale = 0.1; grad_used = [0.6, -0.8, 0]
    # p_new = p - lr*grad_used = [3-0.06, -4+0.08, 0]
    expected = before - 0.1 * torch.tensor([0.6, -0.8, 0.0])
    assert torch.allclose(p.data, expected, atol=1e-6), \
        f"LOMO update wrong: {p.data} != {expected}"
    # Global grad norm was clipped (the unclipped step would have moved more).
    moved = (before - p.data).norm()
    assert moved < 1.0, f"global clip failed: moved {moved:.3f} > clip*lr"
    print(f"[ok] LOMO clipped fused update: p={p.data.tolist()}, moved={moved:.3f}")


def test_lomo_frees_grad():
    """LOMO step() frees the grad (the low-memory point): autograd re-materializes
    it after the live hook, so step() clears it."""
    p = nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = LOMO([p], lr=0.1, clip_grad=1.0)
    loss = (p * p).sum()
    opt.zero_grad(); loss.backward(retain_graph=True)
    opt.begin_fused_update()
    loss.backward()
    opt.step()
    assert p.grad is None, "LOMO did not free the grad"
    print("[ok] LOMO freed p.grad after step()")


# ---------------------------------------------------------------------------
# 4. AdaLomo: single-pass momentum + per-tensor clip.
# ---------------------------------------------------------------------------
def test_adalomo_updates_single_pass():
    """AdaLomo: a SINGLE backward updates the param (no dry pass needed)."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([3.0, -4.0]))
    opt = AdaLomo([p], lr=0.1, momentum=0.0, clip_grad=10.0)
    # momentum=0 + clip high => behaves like plain SGD step with grad 2p.
    before = p.data.clone()
    loss = (p * p).sum()           # grad = [6, -8], norm=10 < clip 10 (no clip)
    opt.zero_grad(); loss.backward()
    # update happens inside backward; opt.step is a no-op.
    expected = before - 0.1 * torch.tensor([6.0, -8.0])
    assert torch.allclose(p.data, expected, atol=1e-6), \
        f"AdaLomo update wrong: {p.data} != {expected}"
    print(f"[ok] AdaLomo single-pass update: p={p.data.tolist()}")


def test_adalomo_momentum_buffer():
    """AdaLomo keeps an fp32 momentum buffer across steps."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.tensor([1.0]))
    opt = AdaLomo([p], lr=0.0, momentum=0.9, clip_grad=1e9)  # lr 0: no move
    loss = (p * 5).sum()           # grad = 5
    opt.zero_grad(); loss.backward()
    buf1 = opt.state[p]["momentum_buffer"].clone()
    # (1-m)*g = 0.1*5 = 0.5
    assert torch.allclose(buf1, torch.tensor([0.5]), atol=1e-6), f"buf1={buf1}"
    loss = (p * 5).sum()
    opt.zero_grad(); loss.backward()
    buf2 = opt.state[p]["momentum_buffer"].clone()
    # buf2 = 0.9*0.5 + 0.1*5 = 0.45+0.5 = 0.95
    assert torch.allclose(buf2, torch.tensor([0.95]), atol=1e-6), f"buf2={buf2}"
    print(f"[ok] AdaLomo momentum buffer: 0.5 -> 0.95")


# ---------------------------------------------------------------------------
# 5. LISA: only the active subset updates; frozen subset untouched.
# ---------------------------------------------------------------------------
def test_lisa_freezes_inactive():
    """LISA.sample_layers() must freeze all but K layer groups, and the always-on
    wrapper params must stay trainable."""
    torch.manual_seed(0)
    # 2 always-on wrapper params + 4 decoder-layer groups (1 param each).
    wrap = [nn.Parameter(torch.randn(4)), nn.Parameter(torch.randn(4))]
    groups = [[nn.Parameter(torch.randn(4))] for _ in range(4)]
    opt = LISA(wrap, groups, lr=0.1, k=2, refresh_every=1000, bias_importance=False)
    opt.mark_trainable([p for g in groups for p in g])
    opt.sample_layers(0)

    # Wrapper params always trainable.
    assert all(w.requires_grad for w in wrap), "wrapper params frozen"
    # Exactly k=2 layer groups active.
    active = sum(1 for g in groups for p in g if p.requires_grad)
    assert active == 2, f"LISA activated {active} != 2 layer params"
    print(f"[ok] LISA froze {4-2}/4 layer groups, kept 2 wrappers active")


def test_lisa_only_updates_active():
    """After a step, frozen layer params keep their values; active ones move."""
    torch.manual_seed(0)
    wrap = [nn.Parameter(torch.zeros(2))]
    groups = [[nn.Parameter(torch.zeros(2))] for _ in range(3)]
    opt = LISA(wrap, groups, lr=0.1, k=1, refresh_every=1000, bias_importance=False)
    opt.mark_trainable([p for g in groups for p in g])
    opt.sample_layers(0)

    # Which group is active?
    active_idx = None
    for gi, g in enumerate(groups):
        if g[0].requires_grad:
            active_idx = gi

    # Give each param a grad; step.
    for w in wrap:
        w.grad = torch.ones(2)
    for g in groups:
        for p in g:
            p.grad = torch.ones(2)   # frozen ones get grad too, but won't update
    opt.step()

    # Active group + wrapper updated; frozen groups untouched.
    assert torch.allclose(groups[active_idx][0].data,
                          torch.full((2,), -0.1), atol=1e-6), "active layer didn't update"
    assert torch.allclose(wrap[0].data, torch.full((2,), -0.1), atol=1e-6), \
        "wrapper didn't update"
    for gi, g in enumerate(groups):
        if gi == active_idx:
            continue
        assert torch.allclose(g[0].data, torch.zeros(2)), \
            f"frozen layer {gi} was modified"
    # State only on active + wrapper (frozen groups have no AdamW state).
    n_with_state = sum(1 for g in groups for p in g if p in opt.state)
    assert n_with_state == 1, f"optimizer state on {n_with_state} layers, expected 1"
    print(f"[ok] LISA updated only active layer #{active_idx} + wrapper; "
          f"2 layers untouched, no state on frozen layers")


def test_lisa_covers_all_layers():
    """Over many steps with uniform sampling, every layer group gets activated
    at least once (no permanently-starved layer)."""
    torch.manual_seed(0)
    wrap = [nn.Parameter(torch.zeros(2))]
    groups = [[nn.Parameter(torch.zeros(2))] for _ in range(5)]
    opt = LISA(wrap, groups, lr=0.1, k=1, refresh_every=1000, bias_importance=False)
    opt.mark_trainable([p for g in groups for p in g])
    seen = set()
    for s in range(200):
        opt.sample_layers(s)
        for gi, g in enumerate(groups):
            if g[0].requires_grad:
                seen.add(gi)
    assert seen == set(range(5)), f"LISA never activated layers: {set(range(5)) - seen}"
    print(f"[ok] LISA activated all 5 layers over 200 steps")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 64)
    print("Optimizer + vectorization tests (torch-only, no transformers)")
    print("=" * 64)
    test_masked_random_mask_counts()
    test_masked_random_mask_distribution()
    test_jepa_loss_matches_per_example()
    test_jepa_loss_grad_matches_per_example()
    test_jepa_loss_empty_rows()
    test_lomo_updates_and_clips()
    test_lomo_frees_grad()
    test_adalomo_updates_single_pass()
    test_adalomo_momentum_buffer()
    test_lisa_freezes_inactive()
    test_lisa_only_updates_active()
    test_lisa_covers_all_layers()
    print("=" * 64)
    print("ALL TESTS PASSED")
    print("=" * 64)
