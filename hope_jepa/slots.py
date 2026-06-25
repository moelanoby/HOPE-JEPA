"""Slot-based JEPA predictors, sparse slot-mixing, and slot-assembled readout.

This is the "slot system" layered on top of the HOPE-JEPA-SIGReg backbone.
Three ideas, jointly:

  1. Diverging slots. Instead of one JEPA predictor we have K *slot* queries
     (`slots` in [K, d], shared across layers, also shared with the readout and
     the divergence regularizer). Each slot is pushed to specialize on a
     *different* aspect of the data by `slot_divergence_loss` -- it decorrelates
     the slot embeddings so they spread out in representation space.

  2. Sparse-mix attention. Each slot produces its own per-target prediction
     `P_k(t)`. The final prediction is a weighted SUM over slots (not a "pick
     one" routing). The weights come from an entmax-1.5 attention between a
     per-target context summary and the slot embeddings, which -- unlike softmax
     -- can output *exact zeros*, so a slot can be fully dropped for a given
     target. "They all get mixed in the end" -- but the mix can exclude any
     slot.

  3. Slot-assembled readout. The same slot set gathers per-slot summaries of the
     patch tokens (standard softmax cross-attention), then the image-level mix
     is again a sparse entmax combination of those summaries, driven by the CLS
     token. This is the "full picture" reassembled from the divergent slots,
     and it is what the linear probe consumes.

The shared `slots` tensor is what makes (1), (2), and (3) two views of one
decomposition: the same K roles that JEPA predicts with are the roles the
encoder gathers and recombines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .entmax import entmax15


def _standardize_last_dim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Zero-mean, unit-variance along the last dim.

    Used before entmax so the slot-mix sparsity depends only on the learnable
    inverse temperature, not on the (init-dependent) raw logit magnitude.
    """
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True)
    return (x - mu) / (sd + eps)


# ---------------------------------------------------------------------------
# Slot JEPA predictor
# ---------------------------------------------------------------------------
class SlotJEPAPredictor(nn.Module):
    """A slot-based JEPA predictive head.

    Mirrors `JEPAPredictor`'s transformer-over-[context|mask-tokens] core, but
    instead of emitting one prediction per masked position it emits K per-slot
    predictions and mixes them with sparse entmax attention.

    Args:
        d_model:   token embedding dim.
        num_heads: heads in the predictor transformer.
        num_slots: K, the number of slots.
        depth:     predictor transformer depth.
        dropout:   predictor dropout.
    """

    def __init__(self, d_model: int, num_heads: int, num_slots: int,
                 depth: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=False, activation="gelu",
        )
        self.predictor = nn.TransformerEncoder(layer, num_layers=depth)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # One linear "slot head" per slot: maps the per-target context summary
        # to a slot-specialized prediction. ModuleList (not a single [d,K,d])
        # so each slot's role is an independent linear map.
        self.slot_heads = nn.ModuleList([nn.Linear(d_model, d_model)
                                         for _ in range(num_slots)])

        # Mixing-attention projections: logits_k = <Wq . c_t, Wk . slot_k> / sqrt(d).
        # We then STANDARDIZE the logits across the slot axis (per target) and
        # scale by a LEARNABLE inverse temperature beta before entmax. Why
        # standardize: the raw dot product's magnitude depends on a pile of
        # init-dependent factors (slot std, transformer output scale, 1/sqrt(d))
        # and in practice starts ~3 orders of magnitude too small for entmax to
        # drop any slot. Standardizing removes that dependence entirely -- the
        # sparsity is then a function only of beta, which is stable and learnable.
        # beta = softplus(raw) is strictly positive; init 2.0 -> beta ~2.4 ->
        # ~30% of slots dropped at init, i.e. the "can choose 0 for something"
        # property is non-degenerate from step 0.
        self.mix_q = nn.Linear(d_model, d_model)
        self.mix_k = nn.Linear(d_model, d_model)
        self.raw_inv_temp = nn.Parameter(torch.tensor(2.0))    # softplus -> ~2.4, sparse init

    def forward(self, z_context: torch.Tensor, context_pos: torch.Tensor,
                target_pos: torch.Tensor, slots: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict target embeddings via K slots, sparse-mixed.

        Args:
            z_context:  [B, Nc, d]  embeddings of context tokens.
            context_pos:[B, Nc]     (unused in computation; kept for API parity
                        with the non-slot predictor / future position features).
            target_pos: [B, Nt]     position indices to predict.
            slots:      [K, d]      shared slot embeddings (mix keys).
        Returns:
            preds:   [B, Nt, d]   sparse slot-mixed predictions.
            weights: [B, Nt, K]   entmax weights used per target (sum to 1 over
                                  K; entries can be exactly 0).
        """
        B, Nc, d = z_context.shape
        Nt = target_pos.shape[1]
        K = self.num_slots

        mask_tok = self.mask_token.expand(B, Nt, -1)
        seq = torch.cat([z_context, mask_tok], dim=1)         # [B, Nc+Nt, d]
        seq = seq + self.pos_embed[:, : seq.shape[1]]
        out = self.predictor(seq)
        c = out[:, Nc:]                                       # [B, Nt, d] per-target summary

        # Per-slot predictions, accumulated into the mix lazily to avoid forming
        # a [B, Nt, K, d] tensor. For cifar100 (B=512, Nt~42, K=8, d=384) that
        # would be ~1.3GB; the loop keeps peak memory at [B, Nt, d].
        slot_preds = [head_k(c) for head_k in self.slot_heads]  # list of [B, Nt, d]

        # Sparse mixing weights over slots, per target. Standardize the logits
        # across the slot axis (per target) so sparsity is decoupled from the
        # raw magnitude (see __init__), then scale by a learnable inverse
        # temperature and apply entmax-1.5.
        q = self.mix_q(c)                                     # [B, Nt, d]
        ks = self.mix_k(slots).unsqueeze(0)                   # [1, K, d]
        logits = torch.matmul(q, ks.transpose(1, 2)) / (d ** 0.5)  # [B, Nt, K]
        logits = _standardize_last_dim(logits)                # per-target, over K
        beta = F.softplus(self.raw_inv_temp)                  # > 0; init ~2.4 (sparse)
        weights = entmax15(logits * beta, dim=-1)              # [B, Nt, K], on simplex

        preds = torch.zeros_like(c)                           # [B, Nt, d]
        for k in range(K):
            w_k = weights[:, :, k].unsqueeze(-1)              # [B, Nt, 1]
            preds = preds + w_k * slot_preds[k]
        return preds, weights


def jepa_slot_layer_loss(z: torch.Tensor, mask: torch.Tensor,
                         predictor: SlotJEPAPredictor, slots: torch.Tensor
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Slot-JEPA prediction loss for a single layer (MSE) + the mixing weights.

    Mirrors `jepa_layer_loss` (per-example Python loop over the batch, since
    masks are variable-length and N is small).

    Returns:
        (mse_loss, weights) where weights is the full per-target entmax weight
        tensor [n_examples, Nt, K] across the layer (before any averaging), so
        the caller can compute the genuine zero-fraction sparsity. Entries are
        exactly 0 wherever entmax dropped a slot for a target.
    """
    B, N, d = z.shape
    preds_list, targets_list, weights_list = [], [], []
    for i in range(B):
        m = mask[i]
        tpos = torch.nonzero(m, as_tuple=False).squeeze(-1)
        cpos = torch.nonzero(~m, as_tuple=False).squeeze(-1)
        if tpos.numel() == 0 or cpos.numel() == 0:
            continue
        cpos_b = cpos.unsqueeze(0)
        tpos_b = tpos.unsqueeze(0)
        zc = z[i:i + 1, cpos, :]
        pred, w = predictor(zc, cpos_b, tpos_b, slots)        # [1, Nt, d], [1, Nt, K]
        tgt = z[i:i + 1, tpos, :]
        preds_list.append(pred)
        targets_list.append(tgt)
        weights_list.append(w)

    if not preds_list:
        zero = z.new_zeros(())
        empty_w = z.new_zeros(0, 0, predictor.num_slots)
        return zero, empty_w
    shapes = [p.shape for p in preds_list]
    if all(s == shapes[0] for s in shapes):
        preds = torch.cat(preds_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        weights = torch.cat(weights_list, dim=0)
    else:
        preds = torch.cat([p.view(-1, d) for p in preds_list], dim=0)
        targets = torch.cat([t.view(-1, d) for t in targets_list], dim=0)
        weights = torch.cat([w.view(-1, predictor.num_slots) for w in weights_list], dim=0).unsqueeze(0)
    loss = F.mse_loss(preds, targets)
    return loss, weights


# ---------------------------------------------------------------------------
# Slot divergence
# ---------------------------------------------------------------------------
def slot_divergence_loss(slots: torch.Tensor) -> torch.Tensor:
    """Penalize similarity between slot embeddings.

    Uses the mean off-diagonal *squared* cosine similarity of the
    (row-normalized) slots. Scale-invariant and non-negative, bounded in [0, 1]:
    0 when slots are mutually orthogonal, 1 when they are all identical.

        s_hat = slots / ||slots||        (per-row L2 normalize)
        G     = s_hat @ s_hat^T          ([K, K]; entries are cosines, diag = 1)
        L_div = (sum(G*G) - K) / (K*(K-1))   (mean off-diagonal cos^2; >= 0)

    We penalize the *squared* cosine so opposite-direction slots (negatively
    correlated) are not pushed further apart artificially -- they are already
    decorrelated -- and the penalty stays non-negative (raw-cosine sums can go
    negative, which would be a reward, not a penalty).
    """
    K = slots.shape[0]
    if K <= 1:
        return slots.new_zeros(())
    s = F.normalize(slots, dim=-1)                          # [K, d]
    G = s @ s.t()                                           # [K, K] cosines
    G2 = G * G                                              # cos^2; diag still 1
    off = G2.sum() - K                                      # sum of off-diagonal cos^2
    return off / (K * (K - 1))


# ---------------------------------------------------------------------------
# Slot-assembled readout (encoder "full picture" reassembly)
# ---------------------------------------------------------------------------
class SlotReadout(nn.Module):
    """Assemble the pooled image embedding from sparse slot summaries.

    Two steps:
      (a) Gather: each slot queries the patch tokens with standard softmax
          cross-attention to get a per-slot summary h_k in [B, d]. This is the
          "focus" step -- what does each slot care about across the image.
      (b) Mix: the image-level mix query is the CLS token (position 0). It
          scores the K summaries, an entmax-1.5 picks amounts (can be exactly 0
          per slot), and the pooled embedding is the weighted sum of the
          summaries. This is the "full picture" reassembled from the divergent
          slots, sparsely.

    Args:
        d_model:   token embedding dim.
        num_slots: K (kept for symmetry; the slot set itself lives on the model).
    """

    def __init__(self, d_model: int, num_slots: int):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        # Gather step: slots = queries, patches = keys/values.
        self.gather_q = nn.Linear(d_model, d_model)
        self.gather_k = nn.Linear(d_model, d_model)
        self.gather_v = nn.Linear(d_model, d_model)
        self.gather_norm = nn.LayerNorm(d_model)
        # Mix step: CLS = query, slot summaries = keys/values. As in the slot
        # JEPA predictor, the logits are standardized across the slot axis and
        # scaled by a learnable inverse temperature so entmax can drop slots
        # regardless of the raw dot-product magnitude.
        self.mix_q = nn.Linear(d_model, d_model)
        self.mix_k = nn.Linear(d_model, d_model)
        self.raw_inv_temp = nn.Parameter(torch.tensor(2.0))    # softplus -> ~2.4, sparse init

    def forward(self, z_full: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        """z_full: [B, N+1, d] (CLS at 0), slots: [K, d] -> pooled [B, d]."""
        B, L, d = z_full.shape
        K = slots.shape[0]

        patches = z_full[:, 1:, :]                          # [B, N, d] (drop CLS)
        # --- (a) Gather per-slot summaries via softmax cross-attention. ---
        qs = self.gather_q(slots).unsqueeze(0).expand(B, -1, -1)  # [B, K, d]
        kk = self.gather_k(patches)                               # [B, N, d]
        vv = self.gather_v(patches)                               # [B, N, d]
        attn = torch.matmul(qs, kk.transpose(1, 2)) / (d ** 0.5)  # [B, K, N]
        h = torch.matmul(attn.softmax(dim=-1), vv)                # [B, K, d]
        h = self.gather_norm(h)

        # --- (b) Sparse entmax mix driven by the CLS token. Standardize across
        # slots, scale by a learnable inverse temperature, then entmax-1.5. ---
        cls = z_full[:, 0, :]                                # [B, d]
        qm = self.mix_q(cls).unsqueeze(1)                    # [B, 1, d]
        km = self.mix_k(h)                                   # [B, K, d]
        logits = torch.matmul(qm, km.transpose(1, 2)).squeeze(1) / (d ** 0.5)  # [B, K]
        logits = _standardize_last_dim(logits)               # per-image, over K
        beta = F.softplus(self.raw_inv_temp)                 # > 0; init ~2.4 (sparse)
        w = entmax15(logits * beta, dim=-1)                  # [B, K], on simplex
        pooled = (w.unsqueeze(-1) * h).sum(dim=1)            # [B, d]
        return pooled
