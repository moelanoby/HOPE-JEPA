"""The full HOPE-JEPA-SIGReg model (slot-based JEPA + slot-assembled readout).

Pipeline:
  image --patchify--> patch tokens --[L x HOPE layer]--> layer embeddings
                                                                  |
              per-layer SLOT JEPA head predicts masked positions via K diverging
              slots, sparse-mixed by entmax-1.5 attention
                                                                  |
                          SIGReg regularizes the embedding covariance;
                          slot_divergence decorrelates the shared slots
                                                                  |
              pooled embedding = slot-assembled readout (sparse entmax mix of
              per-slot summaries driven by the CLS token) -- the "full picture"

The shared `slots` parameter ([K, d]) is the unifying object: it is (a) the
mix-key for the JEPA slot heads, (b) the gather-query for the readout, and
(c) the decorrelation target. One decomposition, used in three ways.

`forward` returns everything the loss / training loop needs: the list of
per-layer embeddings (for SIGReg), per-layer JEPA losses, the slot mixing
weights (for the sparsity diagnostic), and the slot-assembled pooled embedding
used only by the downstream linear probe.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .hope import HopeLayer
from .sigreg import SIGReg
from .slots import SlotJEPAPredictor, SlotReadout


class HopeJepaModel(nn.Module):
    """Patch embedding + stack of HOPE layers + slot JEPA predictors + SIGReg.

    Slot system (see `slots.py`): K shared slot embeddings drive both the
    per-layer slot-JEPA predictive heads (diverging per-slot predictions
    sparse-mixed by entmax-1.5) and the slot-assembled readout that produces the
    pooled embedding for the linear probe.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]

        self.img_size = m["img_size"]
        self.patch_size = m["patch_size"]
        self.d_model = m["d_model"]
        self.num_layers = m["num_layers"]
        self.num_slots = int(m.get("num_slots", 4))
        self.slot_div_weight = float(m.get("slot_div_weight", 0.1))
        assert self.img_size % self.patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.in_dim = 3 * self.patch_size * self.patch_size

        # Patch embedding: flattened patch pixels -> d_model.
        self.patch_embed = nn.Linear(self.in_dim, self.d_model)
        # Positional embedding + a CLS token. CLS now drives the slot-mix in the
        # readout rather than *being* the pooled output (see SlotReadout).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.emb_norm = nn.LayerNorm(self.d_model)
        self.drop = nn.Dropout(m.get("dropout", 0.0))

        # Backbone: a stack of HOPE layers.
        t = m["titans"]
        c = m["cms"]
        self.hope_layers = nn.ModuleList([
            HopeLayer(
                d_model=self.d_model,
                d_hidden=t["d_hidden"],
                num_persistent_memory=t["num_persistent_memory"],
                cms_num_modules=c["num_modules"],
                cms_base_update_freq=c["base_update_freq"],
                cms_d_ff_multiplier=c["d_ff_multiplier"],
                dropout=m.get("dropout", 0.0),
            )
            for _ in range(self.num_layers)
        ])

        # Shared slot embeddings: the unifying object across the JEPA slot heads
        # (mix keys), the readout (gather queries), and the divergence penalty.
        self.slots = nn.Parameter(torch.randn(self.num_slots, self.d_model) * 0.02)

        # Per-layer slot JEPA predictors.
        j = m["jepa"]
        self.jepa_heads = nn.ModuleList([
            SlotJEPAPredictor(self.d_model, m["num_heads"], self.num_slots,
                              j["predictor_depth"], m.get("dropout", 0.0))
            for _ in range(self.num_layers)
        ])

        # Slot-assembled readout -> pooled embedding for the linear probe.
        self.slot_readout = SlotReadout(self.d_model, self.num_slots)

        # SIGReg regularizer (applied to the pooled layer embeddings).
        s = cfg["sigreg"]
        self.sigreg_enabled = bool(s["enabled"])
        self.sigreg_weight = float(s["weight"])
        self.sigreg = SIGReg(self.d_model, s["sketch_dim"], s["target_scale"])

        self.mask_ratio = float(j["mask_ratio"])

    # ------------------------------------------------------------------
    def embed_patches(self, images: torch.Tensor) -> torch.Tensor:
        """images [B, C, H, W] -> token sequence [B, N+1, d] with CLS prepended."""
        from .data import patchify
        patches = patchify(images, self.patch_size)            # [B, N, in_dim]
        x = self.patch_embed(patches)                          # [B, N, d]
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)                 # [B, 1, d]
        x = torch.cat([cls, x], dim=1)                         # [B, N+1, d]
        x = x + self.pos_embed[:, : x.shape[1]]
        return self.drop(self.emb_norm(x))

    def forward(self, images: torch.Tensor, global_step: int = 0,
                mask: torch.Tensor | None = None):
        """Run the backbone and per-layer slot-JEPA prediction.

        Args:
            images: [B, C, H, W] input batch (a single augmented view).
            global_step: training step, used by the CMS update-frequency cadence.
            mask: [B, N] boolean patch mask (True = target). If None, a fresh
                  random mask at `mask_ratio` is sampled. CLS (position 0) is
                  never masked.
        Returns:
            dict with:
              layer_embeddings: list of [B, N+1, d] per layer.
              pooled: [B, d] slot-assembled embedding from the final layer
                      (sparse entmax mix of per-slot summaries; for probing).
              mask: [B, N] the mask actually used.
              jepa_losses: list of per-layer scalar losses.
              slot_weights: list of per-layer entmax weight tensors, each
                            [n, Nt, K] (ragged over the batch). The loss uses
                            these to report the genuine per-target sparsity.
        """
        from .data import random_mask
        from .slots import jepa_slot_layer_loss

        x = self.embed_patches(images)                         # [B, N+1, d]
        B = x.shape[0]
        device = x.device

        if mask is None:
            mask = random_mask(B, self.num_patches, self.mask_ratio, device)

        layer_embeddings = []
        jepa_losses = []
        slot_weights = []
        z = x
        for hope_layer, jepa_head in zip(self.hope_layers, self.jepa_heads):
            z = hope_layer(z, global_step)                     # [B, N+1, d]
            layer_embeddings.append(z)
            # Slot JEPA operates on patch tokens (drop CLS), using this layer's
            # own embeddings as targets. No stop-gradient (LeJEPA); SIGReg +
            # slot_divergence prevent collapse.
            z_patches = z[:, 1:, :]                             # [B, N, d]
            loss_l, w_l = jepa_slot_layer_loss(z_patches, mask, jepa_head,
                                               self.slots)
            jepa_losses.append(loss_l)
            slot_weights.append(w_l.detach())

        # Slot-assembled readout from the final layer (the "full picture").
        pooled = self.slot_readout(layer_embeddings[-1], self.slots)
        return {
            "layer_embeddings": layer_embeddings,
            "pooled": pooled,
            "mask": mask,
            "jepa_losses": jepa_losses,
            "slot_weights": slot_weights,
        }

    # ------------------------------------------------------------------
    def encode(self, images: torch.Tensor, global_step: int = 0) -> torch.Tensor:
        """Convenience: return just the pooled [B, d] embedding for the probe.

        Uses the slot-assembled readout so probing sees the same representation
        that training optimizes.
        """
        x = self.embed_patches(images)
        for hope_layer in self.hope_layers:
            x = hope_layer(x, global_step)
        return self.slot_readout(x, self.slots)
