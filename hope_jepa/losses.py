"""Aggregate SSL loss + collapse diagnostics for HOPE-JEPA-SIGReg (slot variant).

The SSL objective is:

    L = mean_l( jepa_slot_loss^l )
      + lambda_sig    * SIGReg( pooled_layer_embeds )
      + lambda_div    * slot_divergence( slots )

where:
  - SIGReg is computed over the concatenation of all per-layer patch embeddings
    (so it regularizes every layer's representation, not just the last),
  - slot_divergence decorrelates the shared slot embeddings so each slot
    specializes on a *different* aspect of the data.

Diagnostics returned:
  - effective rank of the embedding covariance (high = healthy, ~1 = collapsed),
  - mean per-layer slot-JEPA loss,
  - the SIGReg penalty value,
  - the slot divergence value,
  - slot sparsity: mean fraction of zero mixing weights across layers (how often
    the entmax attention fully drops a slot for a target).
"""

from __future__ import annotations

import torch

from .slots import slot_divergence_loss


def ssl_loss(model_out: dict, sigreg_enabled: bool, sigreg_weight: float,
             sigreg_module, slots: torch.Tensor | None = None,
             slot_div_weight: float = 0.0) -> tuple[torch.Tensor, dict]:
    """Compute the total SSL loss and a diagnostics dict.

    Args:
        model_out:       dict returned by `HopeJepaModel.forward`.
        sigreg_enabled:  whether to add the SIGReg term.
        sigreg_weight:   lambda_sig.
        sigreg_module:   the SIGReg nn.Module (penalty + effective rank).
        slots:           shared slot embeddings [K, d]; if given and weight>0,
                         the divergence term is added.
        slot_div_weight: lambda_div on the slot divergence term.
    Returns:
        (total_loss, diagnostics)
    """
    jepa_losses = model_out["jepa_losses"]
    mean_jepa = torch.stack(jepa_losses).mean()

    # SIGReg over all layer patch embeddings concatenated.
    layer_emb = model_out["layer_embeddings"]           # list of [B, N+1, d]
    all_patches = torch.cat([e[:, 1:, :] for e in layer_emb], dim=0)  # [L*B, N, d]
    flat = all_patches.reshape(-1, all_patches.shape[-1])            # [L*B*N, d]

    sigreg_val = flat.new_zeros(())
    div_val = flat.new_zeros(())
    total = mean_jepa
    if sigreg_enabled:
        sigreg_val = sigreg_module(flat)
        total = total + sigreg_weight * sigreg_val
    if slots is not None and slot_div_weight > 0.0:
        div_val = slot_divergence_loss(slots)
        total = total + slot_div_weight * div_val

    with torch.no_grad():
        eff_rank = sigreg_module.effective_rank(flat) if sigreg_enabled else -1.0
        # Slot sparsity: mean fraction of exactly-zero entmax weights across all
        # layers and all (example, target) pairs. Each entry of slot_weights is
        # [n, Nt, K] (ragged over the batch); a 0 means that slot was fully
        # dropped for that (example, target). High = slots often dropped (sparse
        # specialization); ~0 = all slots always active.
        slot_w = model_out.get("slot_weights", [])
        if slot_w:
            n_total = 0.0
            n_zeros = 0.0
            for w in slot_w:
                if w.numel() == 0:
                    continue
                n_total += w.numel()
                n_zeros += int((w == 0).sum().item())
            slot_sparsity = (n_zeros / n_total) if n_total > 0 else 0.0
        else:
            slot_sparsity = 0.0

    diag = {
        "jepa": float(mean_jepa.detach().item()),
        "sigreg": float(sigreg_val.detach().item()),
        "slot_div": float(div_val.detach().item()),
        "slot_sparsity": float(slot_sparsity),
        "total": float(total.detach().item()),
        "eff_rank": float(eff_rank),
    }
    return total, diag
