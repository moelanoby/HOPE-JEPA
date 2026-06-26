"""Slot-JEPA auxiliary objective on an HF LLM's hidden states.

Adapts the image-SSL slot-JEPA head (`hope_jepa.slots.SlotJEPAPredictor`),
SIGReg (`hope_jepa.sigreg.SIGReg`) and slot divergence
(`hope_jepa.slots.slot_divergence_loss`) -- unchanged -- to operate on the
hidden states of an HF `*ForCausalLM` model, masking BPE *token* positions
rather than image patches.

This is the "JEPA goes in" half: an *auxiliary* loss added to the standard
next-token CE during (Q)LoRA finetuning:

    L = CE_next_token
      + lambda_jepa * mean_l( slotJEPA_l )        # per-layer slot prediction
      + lambda_sig  * SIGReg( masked hidden states )
      + lambda_div  * slot_divergence( slots )

The mechanics mirror the image model: at each chosen hidden layer, K diverging
slots predict the masked-token embeddings from the context tokens, and the
per-target predictions are sparse-mixed by entmax-1.5. The difference is only
*where the targets come from* -- the LLM's own `[B, T, h]` hidden states at
those layers, with `random_mask` (token-generic) picking the targets.

Unlike the image model we DROP the CLS convention: an LLM has no CLS, and the
JEPA target is just the layer's hidden state at a masked BPE position.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..data import random_mask
from ..sigreg import SIGReg
from ..slots import SlotJEPAPredictor, jepa_slot_layer_loss, slot_divergence_loss
from .config import HopeLlmConfig, parse_layer_spec


class SlotJEPAForLLM(nn.Module):
    """Holds the slot-JEPA heads + shared slots + SIGReg for an HF LLM.

    Lives *beside* the HF model (not inside it): `compute_loss` takes the list
    of per-layer hidden states the model emits with `output_hidden_states=True`
    and returns the auxiliary loss + diagnostics. The wrapper model owns both
    the HF model and this module (see `JepaReasonerLlm` / the train script).

    Args:
        cfg:        the `HopeLlmConfig`.
        hidden_size: the base model's hidden dim.
        num_layers: the base model's num_hidden_layers (to resolve `jepa.layers`).
    """

    def __init__(self, cfg: HopeLlmConfig, hidden_size: int, num_layers: int):
        super().__init__()
        self.cfg_jepa = cfg.jepa
        self.cfg_sig = cfg.sigreg
        self.slot_div_weight = float(cfg.slot_div_weight)
        self.hidden_size = hidden_size
        self.layer_idxs = parse_layer_spec(cfg.jepa.layers, num_layers)

        if not cfg.jepa.enabled:
            # Still keep slots as an (unused) param buffer so state-dict shapes
            # are stable when toggling the objective on/off.
            self.slots = nn.Parameter(torch.zeros(cfg.jepa.num_slots, hidden_size))
            self.heads = nn.ModuleList()
            self.sigreg = SIGReg(hidden_size, cfg.sigreg.sketch_dim,
                                 cfg.sigreg.target_scale)
            return

        K = cfg.jepa.num_slots
        # Shared slot embeddings [K, h]: the mix keys (JEPA), gather queries
        # (readout, if used), and the divergence target. Same unifying object
        # as in the image model.
        self.slots = nn.Parameter(torch.randn(K, hidden_size) * 0.02)

        # Per-chosen-layer slot-JEPA predictor heads.
        self.heads = nn.ModuleList([
            SlotJEPAPredictor(
                d_model=hidden_size,
                num_heads=cfg.jepa.num_heads,
                num_slots=K,
                depth=cfg.jepa.predictor_depth,
                dropout=0.0,
            )
            for _ in self.layer_idxs
        ])
        self.sigreg = SIGReg(hidden_size, cfg.sigreg.sketch_dim,
                             cfg.sigreg.target_scale)

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        hidden_states: List[torch.Tensor],   # tuple from model(...,output_hidden_states=True)
        attention_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute the slot-JEPA + SIGReg + slot-divergence auxiliary loss.

        Args:
            hidden_states: the HF model's `output_hidden_states` tuple. Index 0
                is the embedding output; index l+1 is the output of layer l.
                We read the entries corresponding to `self.layer_idxs`.
            attention_mask: [B, T] (1 = real token). If given, JEPA targets are
                only sampled among real (non-pad) positions.
            generator: optional torch.Generator for reproducible masking.
        Returns:
            (loss, diagnostics).
        """
        device = hidden_states[0].device
        zero = hidden_states[0].new_zeros(())

        if not self.cfg_jepa.enabled or len(self.heads) == 0:
            return zero, {"jepa": 0.0, "sigreg": 0.0, "slot_div": 0.0,
                          "slot_sparsity": 0.0, "total": 0.0, "eff_rank": -1.0}

        # hidden_states[l+1] is the output of decoder layer l.
        chosen = [hidden_states[l + 1] for l in self.layer_idxs]   # list of [B, T, h]
        B, T, h = chosen[0].shape

        # Build a single token mask [B, T] over REAL positions only, reused
        # across layers (consistent targets across depth, as in the image model
        # where one mask serves all layers).
        valid = (attention_mask.bool()
                 if attention_mask is not None
                 else torch.ones(B, T, dtype=torch.bool, device=device))
        mask = _masked_random_mask(B, T, self.cfg_jepa.mask_ratio, valid, device,
                                   generator)              # True == target

        jepa_losses, slot_weights = [], []
        for z, head in zip(chosen, self.heads):
            loss_l, w_l = jepa_slot_layer_loss(z, mask, head, self.slots)
            jepa_losses.append(loss_l)
            slot_weights.append(w_l.detach())

        mean_jepa = torch.stack(jepa_losses).mean()

        # SIGReg over the concatenated masked hidden states across chosen
        # layers (regularizes every chosen layer's representation).
        masked_states = []
        for z in chosen:
            for i in range(B):
                tpos = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
                if tpos.numel() > 0:
                    masked_states.append(z[i, tpos, :])
        sigreg_val = zero
        flat = None
        scale = None
        if self.cfg_sig.enabled and masked_states:
            flat = torch.cat(masked_states, dim=0)        # [n, h]
            # A pretrained LLM's residual-stream hidden states carry a large,
            # fixed magnitude (per-dim std ~ O(10-100)). SIGReg's
            # ||Cov - sigma^2 I|| penalty assumes unit-scale embeddings; on raw
            # hidden states it blows up to ~ magnitude^4 (a ~7M transient on a
            # 3B model) and its gradient then crushes the representation toward
            # unit variance -- fighting and damaging the pretrained weights (CE
            # rebounds, effective rank collapses). SIGReg's actual job is to
            # prevent collapse / enforce isotropy, i.e. the *shape* of the
            # covariance, not its absolute scale. Normalize by the detached
            # batch RMS so the penalty is scale-invariant: Cov(z/s) = Cov(z)/s^2
            # shares the same (collapse-revealing) shape regardless of magnitude,
            # and detaching `s` means the gradient only reshapes the covariance
            # instead of shrinking the whole residual stream.
            scale = flat.pow(2).mean().sqrt().clamp_min(1e-6).detach()
            sigreg_val = self.sigreg(flat / scale)

        div_val = (slot_divergence_loss(self.slots)
                   if self.slot_div_weight > 0 else zero)

        total = (self.cfg_jepa.weight * mean_jepa
                 + (self.cfg_sig.weight * sigreg_val if self.cfg_sig.enabled else 0.0)
                 + self.slot_div_weight * div_val)

        with torch.no_grad():
            n_total = sum(w.numel() for w in slot_weights)
            n_zeros = sum(int((w == 0).sum().item()) for w in slot_weights)
            slot_sparsity = (n_zeros / n_total) if n_total > 0 else 0.0
            if flat is not None:
                # Rank of the same normalized states SIGReg penalizes, so the
                # diagnostic reflects the collapse signal the regularizer sees.
                normed = flat / scale if scale is not None else flat
                eff_rank = self.sigreg.effective_rank(normed)
            else:
                eff_rank = -1.0

        diag = {
            "jepa": float(mean_jepa.detach().item()),
            "sigreg": float(sigreg_val.detach().item()),
            "slot_div": float(div_val.detach().item()),
            "slot_sparsity": float(slot_sparsity),
            "total": float(total.detach().item()),
            "eff_rank": float(eff_rank),
        }
        return total, diag


def _masked_random_mask(B: int, T: int, mask_ratio: float,
                        valid: torch.Tensor, device,
                        generator=None) -> torch.Tensor:
    """Like `random_mask` but only ever marks *valid* (non-pad) positions.

    For each example we count its valid positions n_i, sample ~mask_ratio*n_i
    of them, and mark exactly those. At least 1 context and 1 target are kept
    whenever n_i >= 2; rows with <2 valid tokens get an all-False mask (those
    examples are skipped by `jepa_slot_layer_loss`).
    """
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for i in range(B):
        valid_idx = torch.nonzero(valid[i], as_tuple=False).squeeze(-1)
        n_i = int(valid_idx.numel())
        if n_i < 2:
            continue
        n_mask = max(1, min(n_i - 1, int(round(n_i * mask_ratio))))
        perm = torch.randperm(n_i, generator=generator, device=device)[:n_mask]
        mask[i, valid_idx[perm]] = True
    return mask
