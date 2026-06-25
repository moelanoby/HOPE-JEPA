"""`HopeLLM`: one object that owns the HF model + slot-JEPA + (optional) Reasoner.

This is the training-loop-facing entrypoint. Its `forward(input_ids,
attention_mask, labels)` runs the (HOPE-augmented) HF model with
`output_hidden_states=True`, computes the standard next-token CE, adds the
slot-JEPA + SIGReg + slot-divergence auxiliary loss, and (if enabled) the
Reasoner's predict-ahead loss -- returning a single scalar plus a diagnostics
dict.

The HF model remains fully usable as an HF model (its `generate()` still
works; see the `hope_block` docstring re: KV-cache recomputation). This wrapper
just adds the JEPA objectives on top.

Usage (see scripts/train_llm_jepa.py for the full QLoRA loop):

    cfg = HopeLlmConfig.from_dict(yaml.safe_load(open(...)))
    model = HopeLLM(cfg)              # builds + splices HOPE layers
    out = model(input_ids=..., attention_mask=..., labels=...)
    out.loss.backward(); opt.step()
    set_global_step(model, step + 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .config import HopeLlmConfig
from .jepa_llm import SlotJEPAForLLM
from .reasoner import JepaReasoner
from .surgery import build_hope_llm


def _chunked_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    chunk_positions: int = 512,
) -> torch.Tensor:
    """Next-token CE that never materializes the full fp32 `[B, T, V]` tensor.

    HF's built-in `ForCausalLMLoss` upcasts the whole `[B, T, V]` logits block
    to float32 and runs `log_softmax` over the *entire* vocab in one call. For a
    large-vocab model (Qwen2.5: V=152,064) at batch 4 x len 1024 that is a multi-
    GB transient -- the classic "loss-function" CUDA OOM even when the model
    weights themselves fit. This helper replicates HF's loss *exactly* (shift by
    one, ignore_index=-100) but flattens to `[N, V]` and processes `chunk_positions`
    rows at a time, so the fp32 peak is `chunk x V x 4` bytes instead of
    `B*T*V*4`. Default 512 rows ~ 0.3 GiB per pass for Qwen2.5's vocab.

    Args:
        logits:          [B, T, V] from the HF model (the LM head already applied).
        labels:          [B, T] (unshifted; we shift internally, matching HF).
        ignore_index:    label value to skip (default -100 / pad).
        chunk_positions: how many flattened positions to reduce per pass.
    Returns:
        scalar mean loss over non-ignored positions.
    """
    # Standard causal shift: predict token t+1 from position t.
    shift_logits = logits[..., :-1, :].reshape(-1, logits.size(-1))   # [N, V]
    shift_labels = labels[..., 1:].reshape(-1)                         # [N]
    N = shift_labels.size(0)

    total = shift_logits.new_zeros((), dtype=torch.float32)
    count = shift_labels.new_zeros((), dtype=torch.float32)
    for s in range(0, N, chunk_positions):
        e = min(s + chunk_positions, N)
        lg = shift_logits[s:e].float()                # [chunk, V] fp32, bounded
        tgt = shift_labels[s:e]                       # [chunk]
        loss = nn.functional.cross_entropy(lg, tgt, ignore_index=ignore_index,
                                           reduction="sum")
        n_valid = (tgt != ignore_index).sum()
        total = total + loss
        count = count + n_valid

    if count.item() == 0:
        return shift_logits.new_zeros(())
    return total / count


@dataclass
class HopeOutput:
    """Return type of `HopeLLM.forward`.

    Note: the HF model's vocab logits are intentionally NOT returned. With a
    large-vocab LM head (Qwen2.5: V=152,064) the `[B, T, V]` tensor is multi-GB
    and pinning it on the autograd graph through `.backward()` is a needless
    OOM risk -- the loss already captures everything needed for training. Call
    `model.model(...)` directly if you need logits at inference time.
    """
    loss: torch.Tensor            # total (CE + aux)
    ce_loss: torch.Tensor         # next-token CE alone
    jepa_diag: dict               # slot-JEPA + SIGReg + div diagnostics
    reasoner_loss: torch.Tensor   # predict-ahead MSE (0 if reasoner off)


class HopeLLM(nn.Module):
    """Wrapper: HF CausalLM (HOPE-augmented) + SlotJEPAForLLM + JepaReasoner.

    Args:
        cfg:    the `HopeLlmConfig`.
        model:  optional pre-built HF model (smoke-test path; skips
                `from_pretrained`). If None, loaded via `build_hope_llm`.
    """

    def __init__(self, cfg: HopeLlmConfig, model: Optional[nn.Module] = None):
        super().__init__()
        self.cfg = cfg
        # Splice HOPE layers into the HF model.
        self.model = build_hope_llm(cfg, model=model)
        hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_hidden_layers

        # Slot-JEPA aux objective on the (now HOPE-augmented) hidden states.
        self.jepa = SlotJEPAForLLM(cfg, hidden_size, num_layers)

        # Optional Reasoner: shares the slot parameter with the JEPA head and
        # reuses the model's lm_head as the Talker.
        self.reasoner = None
        if cfg.reasoner.enabled:
            lm_head = getattr(self.model, "lm_head", None)
            if lm_head is None:
                raise AttributeError(
                    "Reasoner needs `model.lm_head` but the base model has none. "
                    "Disable cfg.reasoner.enabled or use a *ForCausalLM model."
                )
            self.reasoner = JepaReasoner(cfg, hidden_size, self.jepa.slots, lm_head)

    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_reasoner: Optional[bool] = None) -> HopeOutput:
        """Run the HOPE-augmented HF model + JEPA objectives.

        Args:
            input_ids:      [B, T].
            attention_mask: [B, T] (1 = real token).
            labels:         [B, T] for next-token CE (HF shifts internally).
            use_reasoner:   override cfg.reasoner.enabled for this call.
        Returns:
            `HopeOutput` with the total loss ready for `.backward()`.
        """
        do_reasoner = (use_reasoner if use_reasoner is not None
                       else self.reasoner is not None)

        # Run the HF model WITHOUT labels: with a large-vocab LM head, HF's
        # built-in ForCausalLMLoss upcasts the full [B, T, V] logits to fp32 and
        # log-softmaxes them in one shot -- a multi-GB transient that OOMs even
        # when the (quantized) weights fit. We pull logits + hidden states and
        # compute CE ourselves via `_chunked_causal_lm_loss` (same value,
        # bounded fp32 peak). Pass use_cache=False so hidden_states is complete.
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits
        hidden_states = out.hidden_states    # tuple, len = num_layers + 1
        if labels is not None:
            ce_loss = _chunked_causal_lm_loss(logits, labels)
        else:
            ce_loss = input_ids.new_zeros(())
        # Drop the logits reference so the full [B, T, V] tensor can be freed
        # before backward -- keeping it on the graph is a needless OOM risk.
        # `ce_loss` already pulled everything it needs out of it.
        del logits

        # Slot-JEPA + SIGReg + slot-divergence on chosen hidden layers.
        jepa_loss, jepa_diag = self.jepa.compute_loss(
            hidden_states, attention_mask=attention_mask,
        )

        reasoner_loss = input_ids.new_zeros(())
        if do_reasoner and self.reasoner is not None and labels is not None:
            # Reasoner: rollout from the last real token's hidden state, then
            # predict-ahead against the answer-token hidden states.
            # h0 = final layer hidden at the last real position per example.
            B, T, h = hidden_states[-1].shape
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1) - 1      # [B]
            else:
                lengths = torch.full((B,), T - 1, device=input_ids.device)
            h_last = hidden_states[-1][torch.arange(B, device=input_ids.device),
                                       lengths]              # [B, h]
            h_R, trail = self.reasoner.rollout(h_last)       # [B,h], [R+1,B,h]
            # Build the forward answer-token hidden trail [T_ans, B, h]:
            # the per-position final-layer hidden states after the start point.
            # We use the full sequence's hidden states transposed; the
            # predict-ahead loss takes the first `steps` of them.
            ans_h = hidden_states[-1].transpose(0, 1)        # [T, B, h]
            reasoner_loss = self.reasoner.predict_ahead_loss(trail, ans_h)
            jepa_diag["reasoner"] = float(reasoner_loss.detach().item())

        total = ce_loss + jepa_loss + reasoner_loss
        return HopeOutput(
            loss=total, ce_loss=ce_loss, jepa_diag=jepa_diag,
            reasoner_loss=reasoner_loss,
        )

    # ------------------------------------------------------------------
    # Delegate generation to the underlying HF model (it recomputes through
    # HOPE blocks -- correct, not fast).
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
