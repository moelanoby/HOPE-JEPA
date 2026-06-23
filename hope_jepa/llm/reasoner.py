"""JEPA-Reasoner: latent-space reasoning decoupled from token generation.

Following the JEPA-Reasoner paradigm (arXiv:2512.19171), we split the model's
job into a *Reasoner* (think in latent space over R steps) and a *Talker*
(emit tokens from the refined latent). Here the Reasoner is built from your
slot system: K diverging slots each predict the next latent state from the
current one, the per-step predictions are sparse-mixed by entmax-1.5 (a slot
can be fully dropped for a step), and we roll this out for R steps starting
from the model's final hidden state h0. The Talker is a light projection that
feeds the refined latent into the model's existing `lm_head`.

The supervision is shared with the slot-JEPA objective (no separate, redundant
loss): the Reasoner's per-step predictions are supervised to *predict ahead*
the model's own forward hidden states over the next R tokens -- i.e. the slot
predictor doing exactly what it does in the image model, but rolled out over
reasoning steps rather than image patches.

`JepaReasoner` is optional (cfg.reasoner.enabled). When off, the model is pure
(HOPE stack + slot-JEPA aux). When on, `rollout(h0)` returns the refined
latent; the Talker (`to_logits`) turns it into vocab logits.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..entmax import entmax15
from ..slots import SlotJEPAPredictor
from .config import HopeLlmConfig


def _standardize_last_dim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Zero-mean, unit-var along the last dim (reused from slots.py semantics).

    Standardizing the slot-mix logits across the slot axis decouples the
    entmax sparsity from the raw dot-product magnitude (which depends on init).
    """
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True)
    return (x - mu) / (sd + eps)


class JepaReasoner(nn.Module):
    """Latent reasoning rollout over the slot system + a Talker head.

    The Reasoner is a *recurrence* over R steps:
        h_0 = model's final hidden state (the "prompt summary")
        for r in 0..R-1:
            P_k = slot_head_k(h_r)            # K per-slot next-latent predictions
            w   = entmax15( beta * <Qh_r, Kslot> )   # sparse slot mix (can drop)
            h_{r+1} = sum_k w_k * P_k         # refined latent
        h_R = refined latent -> Talker -> logits

    Each slot head is an `nn.Linear(h, h)` (one per slot), mirroring the
    per-slot linear heads in `SlotJEPAPredictor`. The mix uses the shared
    `slots` embeddings as keys (same unifying object as everywhere else).

    Args:
        cfg:        the `HopeLlmConfig`.
        hidden_size: base model hidden dim.
        slots:      the shared [K, h] slot parameter (shared with `SlotJEPAForLLM`,
                    so the roles the JEPA head predicts with are the roles the
                    Reasoner thinks with -- passed in by the wrapper).
        lm_head:    the base model's existing `lm_head` (the Talker reuses it).
    """

    def __init__(self, cfg: HopeLlmConfig, hidden_size: int,
                 slots: nn.Parameter, lm_head: nn.Module):
        super().__init__()
        rcfg = cfg.reasoner
        self.hidden_size = hidden_size
        self.steps = int(rcfg.steps)
        self.num_slots = slots.shape[0]

        # Shared slots (registered as a reference to the *same* parameter the
        # SlotJEPAForLLM module owns, so it is one object used two ways).
        self.slots_ref = slots  # not re-registered: caller's module owns it.

        # Per-slot next-latent predictors. Linear (h -> h), one per slot, as in
        # SlotJEPAPredictor.slot_heads.
        self.slot_heads = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(self.num_slots)]
        )
        # Mixing-attention projections (same construction as SlotJEPAPredictor).
        self.mix_q = nn.Linear(hidden_size, hidden_size)
        self.mix_k = nn.Linear(hidden_size, hidden_size)
        self.raw_inv_temp = nn.Parameter(torch.tensor(2.0))   # softplus -> ~2.4

        # Talker: project the refined latent back into the model's hidden space,
        # then reuse the (frozen-or-LoRA'd) lm_head for vocab logits.
        talker_h = int(hidden_size * rcfg.talker_dim_mult)
        self.talker_norm = nn.LayerNorm(hidden_size)
        self.talker_proj = (nn.Identity() if talker_h == hidden_size
                            else nn.Linear(hidden_size, talker_h))
        self.lm_head = lm_head

        # Layer norm on the input latent so the recurrence is stable over R.
        self.in_norm = nn.LayerNorm(hidden_size)

    # ------------------------------------------------------------------
    def rollout(self, h0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the R-step latent reasoning rollout.

        Args:
            h0: [B, h] starting latent (e.g. the model's final-token hidden).
        Returns:
            h_R:   [B, h] refined latent.
            trail: [R+1, B, h] the latent at each step (h0..hR), for the
                   predict-ahead supervision.
        """
        K, d = self.num_slots, self.hidden_size
        h = self.in_norm(h0)
        trail = [h]
        qk_slots = self.mix_k(self.slots_ref).unsqueeze(0)   # [1, K, d]
        beta = F.softplus(self.raw_inv_temp)                  # > 0, sparse init
        for _ in range(self.steps):
            slot_preds = torch.stack(
                [head(h) for head in self.slot_heads], dim=1
            )                                                 # [B, K, h]
            q = self.mix_q(h).unsqueeze(1)                   # [B, 1, d]
            logits = torch.matmul(q, qk_slots.transpose(1, 2)).squeeze(1) / (d ** 0.5)
            logits = _standardize_last_dim(logits)           # [B, K], over slots
            w = entmax15(logits * beta, dim=-1)              # [B, K], sparse simplex
            h = (w.unsqueeze(-1) * slot_preds).sum(dim=1)    # [B, h]
            trail.append(h)
        return h, torch.stack(trail, dim=0)                  # [B, h], [R+1, B, h]

    # ------------------------------------------------------------------
    def to_logits(self, h_R: torch.Tensor) -> torch.Tensor:
        """Talker: refined latent -> vocab logits via the reused lm_head."""
        return self.lm_head(self.talker_proj(self.talker_norm(h_R)))

    # ------------------------------------------------------------------
    def predict_ahead_loss(self, trail: torch.Tensor,
                           answer_hiddens: torch.Tensor) -> torch.Tensor:
        """Supervise the Reasoner rollout with predict-ahead.

        Each Reasoner step r predicts the *r-th ahead* answer-token hidden
        state. `trail` is [R+1, B, h] (h0..hR); we compare trail[r+1] (the
        prediction made at step r) against answer_hiddens[r] for
        r in 0..min(R, T_ans)-1. MSE, as in the slot-JEPA head.

        Args:
            trail:           [R+1, B, h] from `rollout`.
            answer_hiddens:  [T_ans, B, h] forward hidden states of the answer
                             tokens (the model's own embeddings, LeJEPA-style --
                             no EMA teacher, no stop-grad).
        Returns:
            scalar MSE, or 0 if no overlapping steps.
        """
        R = self.steps
        T = answer_hiddens.shape[0]
        steps = min(R, T)
        if steps <= 0:
            return trail.new_zeros(())
        preds = trail[1:steps + 1]                # [steps, B, h]
        targets = answer_hiddens[:steps]           # [steps, B, h]
        return F.mse_loss(preds, targets)
