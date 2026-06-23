"""CPU smoke test for the HOPE-into-LLM integration.

Builds a TINY Llama (2 layers, hidden 64, vocab 256) *from a config* -- no
weights download, no GPU -- and verifies the whole pipeline end to end:

  1. all 3 placement modes (replace / insert / swap_attention) forward cleanly
     and produce the right shapes,
  2. the slot-JEPA + SIGReg + slot-divergence aux loss is finite and
     backprop-able,
  3. `set_global_step` actually changes the CMS cadence (the staggered update
     fires differently at step 0 vs step 1),
  4. the JEPA-Reasoner rollout produces the right shapes and its predict-ahead
     loss is finite,
  5. a full HopeLLM forward + backward + optimizer step runs without error.

Runs in seconds on CPU. Run with:
    python -m tests.test_llm_smoke
"""

from __future__ import annotations

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaForCausalLM

from hope_jepa.llm import (
    HopeLLM, HopeLlmConfig, build_hope_llm, install_hope_layers,
    set_global_step, get_global_step,
)
from hope_jepa.llm.hope_block import HopeDecoderLayer, MACAttnAdapter


# ---------------------------------------------------------------------------
def _tiny_config(**overrides) -> HopeLlmConfig:
    cfg = HopeLlmConfig()
    cfg.model_id = "tiny-llama"
    cfg.placement = "swap_attention"
    cfg.target_layers = "last:2"
    cfg.hope.d_hidden = 0
    cfg.jepa.enabled = True
    cfg.jepa.layers = "last:2"
    cfg.jepa.num_slots = 4
    cfg.jepa.num_heads = 4
    cfg.jepa.predictor_depth = 1
    cfg.jepa.mask_ratio = 0.4
    cfg.jepa.weight = 1.0
    cfg.sigreg.enabled = True
    cfg.sigreg.sketch_dim = 16
    cfg.sigreg.weight = 1.0
    cfg.slot_div_weight = 0.1
    cfg.reasoner.enabled = False
    cfg.reasoner.steps = 3
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _tiny_model() -> LlamaForCausalLM:
    # 2 layers, hidden 64, vocab 256 -- instantiates from config, no download.
    lc = LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        hidden_act="silu", max_position_embeddings=128,
    )
    return LlamaForCausalLM(lc)


def _rand_batch(B=4, T=16, device="cpu"):
    input_ids = torch.randint(0, 256, (B, T), device=device)
    attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
    labels = input_ids.clone()
    # Make the last quarter of each row padding to exercise the valid-only mask.
    attention_mask[:, -(T // 4):] = 0
    labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
def test_placement_modes():
    """All 3 placement modes forward and produce [B, T, hidden] logits."""
    B, T = 4, 16
    input_ids, attention_mask, _ = _rand_batch(B, T)
    for mode in ("replace", "insert", "swap_attention"):
        cfg = _tiny_config(placement=mode)
        model = _tiny_model()
        install_hope_layers(model, cfg)

        # Sanity: the surgery actually happened.
        if mode == "replace":
            assert isinstance(model.model.layers[-1], HopeDecoderLayer), \
                f"{mode}: last layer not a HopeDecoderLayer"
        elif mode == "swap_attention":
            assert isinstance(model.model.layers[-1].self_attn, MACAttnAdapter), \
                f"{mode}: last layer's self_attn not a MACAttnAdapter"
        elif mode == "insert":
            # Insertion lengthens the stack; check at least one extra
            # HopeDecoderLayer got spliced in (n was 2, now > 2).
            n_hope = sum(1 for l in model.model.layers
                         if isinstance(l, HopeDecoderLayer))
            assert n_hope >= 1 and len(model.model.layers) > 2, \
                f"{mode}: no HopeDecoderLayer inserted"

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        assert out.logits.shape == (B, T, 256), \
            f"{mode}: logits {out.logits.shape} != ({B}, {T}, 256)"
    print(f"[ok] placement modes (replace/insert/swap_attention) forward cleanly")


def test_global_step_threading():
    """set_global_step updates the CMS cadence observable."""
    cfg = _tiny_config(placement="replace")
    model = build_hope_llm(cfg, model=_tiny_model())
    set_global_step(model, 0)
    assert get_global_step() == 0
    # Walk to the HopeDecoderLayer and check its CMS sees a different active
    # set at step 0 vs step 1 (module k is active iff step % (base*2^k) == 0).
    hope_layer = model.model.layers[-1].hope
    cms = hope_layer.cms
    active_0 = [cms._is_active(k, 0) for k in range(len(cms.modules_list))]
    set_global_step(model, 1)
    active_1 = [cms._is_active(k, get_global_step())
                for k in range(len(cms.modules_list))]
    assert active_0 != active_1, "global_step did not change the CMS cadence"
    print(f"[ok] set_global_step threads the cadence: step0 active={active_0} "
          f"step1 active={active_1}")


def test_slot_jepa_aux_loss():
    """Slot-JEPA + SIGReg + slot_div aux loss is finite and backprops."""
    cfg = _tiny_config()
    model = HopeLLM(cfg, model=_tiny_model())
    input_ids, attention_mask, labels = _rand_batch()
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert torch.isfinite(out.loss), "total loss not finite"
    assert torch.isfinite(out.ce_loss), "CE loss not finite"
    assert out.jepa_diag["jepa"] >= 0, "JEPA loss negative"
    assert out.jepa_diag["sigreg"] >= 0, "SIGReg negative"
    assert out.jepa_diag["slot_div"] >= 0, "slot_div negative"
    out.loss.backward()
    # The slot param got a gradient.
    assert model.jepa.slots.grad is not None, "slots received no gradient"
    assert torch.isfinite(model.jepa.slots.grad).all(), "slot grad not finite"
    print(f"[ok] aux loss finite + backprops: "
          f"jepa={out.jepa_diag['jepa']:.4f} "
          f"sigreg={out.jepa_diag['sigreg']:.4f} "
          f"div={out.jepa_diag['slot_div']:.4f} "
          f"sparsity={out.jepa_diag['slot_sparsity']:.3f} "
          f"eff_rank={out.jepa_diag['eff_rank']:.2f}")


def test_reasoner():
    """The JEPA-Reasoner rollout + predict-ahead loss works."""
    from hope_jepa.llm.config import ReasonerCfg
    cfg = _tiny_config()
    cfg.reasoner = ReasonerCfg(enabled=True, steps=3)
    model = HopeLLM(cfg, model=_tiny_model())
    input_ids, attention_mask, labels = _rand_batch()
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert torch.isfinite(out.reasoner_loss), "reasoner loss not finite"
    # Rollout shape check.
    h0 = model.model.model.embed_tokens(input_ids).mean(dim=1)  # [B, h]
    h_R, trail = model.reasoner.rollout(h0)
    assert h_R.shape == h0.shape, "rollout changed shape"
    assert trail.shape[0] == cfg.reasoner.steps + 1, "trail length wrong"
    out.loss.backward()
    print(f"[ok] reasoner rollout (R={cfg.reasoner.steps}) + predict-ahead "
          f"loss={out.reasoner_loss.item():.4f}")


def test_full_train_step():
    """A complete HopeLLM forward + backward + optimizer step runs."""
    cfg = _tiny_config()
    model = HopeLLM(cfg, model=_tiny_model())
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    input_ids, attention_mask, labels = _rand_batch()

    set_global_step(model, 0)
    out0 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    opt.zero_grad()
    out0.loss.backward()
    opt.step()
    set_global_step(model, 1)

    out1 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # (We don't assert loss decreased -- one step on random data is noise; we
    # only assert the whole loop runs and produces finite values.)
    assert torch.isfinite(out0.loss) and torch.isfinite(out1.loss)
    print(f"[ok] full train step runs: loss0={out0.loss.item():.4f} "
          f"loss1={out1.loss.item():.4f}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("HOPE-into-LLM CPU smoke test (tiny 2-layer Llama, no download)")
    print("=" * 60)
    test_placement_modes()
    test_global_step_threading()
    test_slot_jepa_aux_loss()
    test_reasoner()
    test_full_train_step()
    print("=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
