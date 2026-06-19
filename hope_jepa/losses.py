"""Aggregate SSL loss + collapse diagnostics for HOPE-JEPA-SIGReg.

The SSL objective is:

    L = mean_l( jepa_layer_loss^l ) + lambda_sig * SIGReg( pooled_layer_embeds )

where SIGReg is computed over the concatenation of all per-layer patch
embeddings (so it regularizes every layer's representation, not just the last).

We also return diagnostics that make SIGReg's role visible:
  - effective rank of the embedding covariance (high = healthy, ~1 = collapsed),
  - mean per-layer JEPA loss,
  - the SIGReg penalty value.
"""

from __future__ import annotations

import torch


def ssl_loss(model_out: dict, sigreg_enabled: bool, sigreg_weight: float,
             sigreg_module) -> tuple[torch.Tensor, dict]:
    """Compute the total SSL loss and a diagnostics dict.

    Args:
        model_out:     dict returned by `HopeJepaModel.forward`.
        sigreg_enabled: whether to add the SIGReg term.
        sigreg_weight:  lambda_sig.
        sigreg_module:  the SIGReg nn.Module (for computing both penalty and
                        effective rank).
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
    if sigreg_enabled:
        sigreg_val = sigreg_module(flat)
        total = mean_jepa + sigreg_weight * sigreg_val
    else:
        total = mean_jepa

    with torch.no_grad():
        eff_rank = sigreg_module.effective_rank(flat) if sigreg_enabled else -1.0

    diag = {
        "jepa": float(mean_jepa.detach().item()),
        "sigreg": float(sigreg_val.detach().item()),
        "total": float(total.detach().item()),
        "eff_rank": float(eff_rank),
    }
    return total, diag
