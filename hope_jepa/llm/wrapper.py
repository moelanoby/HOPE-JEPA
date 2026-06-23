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


@dataclass
class HopeOutput:
    """Return type of `HopeLLM.forward`."""
    loss: torch.Tensor            # total (CE + aux)
    ce_loss: torch.Tensor         # next-token CE alone
    jepa_diag: dict               # slot-JEPA + SIGReg + div diagnostics
    reasoner_loss: torch.Tensor   # predict-ahead MSE (0 if reasoner off)
    logits: torch.Tensor          # the HF model's vocab logits


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

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        ce_loss = out.loss if out.loss is not None else input_ids.new_zeros(())
        logits = out.logits
        hidden_states = out.hidden_states    # tuple, len = num_layers + 1

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
            reasoner_loss=reasoner_loss, logits=logits,
        )

    # ------------------------------------------------------------------
    # Delegate generation to the underlying HF model (it recomputes through
    # HOPE blocks -- correct, not fast).
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
