"""Per-layer JEPA predictive head (NON-SLOT BASELINE / ablation).

This is the original single-predictor JEPA head, kept for ablation against the
slot-based predictor in `slots.py`. The model in `model.py` uses the slot
variant (`SlotJEPAPredictor`); this module is not wired into the live model but
is retained so the slot system can be A/B-tested by swapping heads.

At each HOPE layer we have a token sequence z^l in [B, N, d] (the layer's
output embedding of the *same* patch positions). JEPA predicts, *in latent
space*, the embeddings of a random subset of masked ("target") patches from the
remaining "context" patches via a small transformer predictor.

Consistent with LeJEPA, the targets are the network's own layer embeddings
(no EMA teacher, no stop-gradient on the target branch). Representation collapse
-- the obvious failure mode of self-targeting prediction -- is handled by
SIGReg (see `sigreg.py`), not by architectural tricks.

The sum over layers of `layer_loss` is the deep/hierarchical JEPA objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAPredictor(nn.Module):
    """A small transformer that maps context tokens + mask-position queries to
    predicted target embeddings.

    The predictor receives the context (visible) tokens plus learned query
    embeddings placed at the masked positions; it outputs a prediction for the
    masked positions, compared to the layer's own embeddings there.
    """

    def __init__(self, d_model: int, num_heads: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=False, activation="gelu",
        )
        self.predictor = nn.TransformerEncoder(layer, num_layers=depth)
        # Learned positional embedding so the predictor knows *which* positions
        # to predict (reuses the model's position indices).
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        # mask-token embedding used for target positions in the predictor input.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, z_context: torch.Tensor, context_pos: torch.Tensor,
                target_pos: torch.Tensor) -> torch.Tensor:
        """Predict target embeddings from context embeddings.

        Args:
            z_context: [B, Nc, d]  embeddings of context tokens.
            context_pos: [B, Nc]   position indices (0..N-1) of those tokens.
            target_pos:  [B, Nt]   position indices to predict.
        Returns:
            preds: [B, Nt, d] predicted embeddings at the target positions.
        """
        B, Nc, d = z_context.shape
        Nt = target_pos.shape[1]

        mask_tok = self.mask_token.expand(B, Nt, -1)
        # Sequence: [context tokens | mask tokens], each with positional embedding.
        seq = torch.cat([z_context, mask_tok], dim=1)
        positions = torch.cat([context_pos, target_pos], dim=1)
        seq = seq + self.pos_embed[:, : seq.shape[1]]
        # (positional info already added; positions tensor itself is used only to
        # index outputs below, kept for clarity.)
        out = self.predictor(seq)
        preds = out[:, Nc:]                      # take the mask-token outputs
        return preds


def jepa_layer_loss(z: torch.Tensor, mask: torch.Tensor,
                    predictor: JEPAPredictor) -> torch.Tensor:
    """Compute the JEPA prediction loss for a single layer.

    Args:
        z:         [B, N, d] layer embeddings (target source).
        mask:      [B, N]    boolean; True == target/masked position.
        predictor: the layer's predictor head.
    Returns:
        scalar mean squared error between predicted and true target embeddings.
    """
    B, N, d = z.shape
    device = z.device
    # Build per-example index lists for context and target positions.
    # (Masks are variable-length so we pad; here N is small (64) so a Python
    # loop over the batch is cheap and keeps the code clear.)
    preds_list, targets_list = [], []
    for i in range(B):
        m = mask[i]
        tpos = torch.nonzero(m, as_tuple=False).squeeze(-1)
        cpos = torch.nonzero(~m, as_tuple=False).squeeze(-1)
        if tpos.numel() == 0 or cpos.numel() == 0:
            continue
        cpos_b = cpos.unsqueeze(0)               # [1, Nc]
        tpos_b = tpos.unsqueeze(0)               # [1, Nt]
        zc = z[i:i + 1, cpos, :]                 # [1, Nc, d]
        pred = predictor(zc, cpos_b, tpos_b)     # [1, Nt, d]
        tgt = z[i:i + 1, tpos, :]                # [1, Nt, d]
        preds_list.append(pred)
        targets_list.append(tgt)

    if not preds_list:
        return z.new_zeros(())
    shapes = [p.shape for p in preds_list]
    if all(s == shapes[0] for s in shapes):
        preds = torch.cat(preds_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
    else:
        preds = torch.cat([p.view(-1, d) for p in preds_list], dim=0)
        targets = torch.cat([t.view(-1, d) for t in targets_list], dim=0)
    return F.mse_loss(preds, targets)
