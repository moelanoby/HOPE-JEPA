"""The full HOPE-JEPA-SIGReg model.

Pipeline:
  image --patchify--> patch tokens --[L x HOPE layer]--> layer embeddings
                                                                  |
                                  per-layer JEPA head predicts masked positions
                                                                  |
                          SIGReg regularizes the embedding covariance

`forward` returns everything the loss / training loop needs: the list of
per-layer embeddings (for the JEPA losses and SIGReg), and a pooled [CLS]
embedding used only by the downstream linear probe.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .hope import HopeLayer
from .jepa import JEPAPredictor
from .sigreg import SIGReg


class HopeJepaModel(nn.Module):
    """Patch embedding + stack of HOPE layers + per-layer JEPA predictors + SIGReg."""

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]

        self.img_size = m["img_size"]
        self.patch_size = m["patch_size"]
        self.d_model = m["d_model"]
        self.num_layers = m["num_layers"]
        assert self.img_size % self.patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.in_dim = 3 * self.patch_size * self.patch_size

        # Patch embedding: flattened patch pixels -> d_model.
        self.patch_embed = nn.Linear(self.in_dim, self.d_model)
        # Positional embedding + a CLS token (CLS used only by the linear probe).
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

        # Per-layer JEPA predictors.
        j = m["jepa"]
        self.jepa_heads = nn.ModuleList([
            JEPAPredictor(self.d_model, m["num_heads"], j["predictor_depth"],
                          m.get("dropout", 0.0))
            for _ in range(self.num_layers)
        ])

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
        """Run the backbone and per-layer JEPA prediction.

        Args:
            images: [B, C, H, W] input batch (a single augmented view).
            global_step: training step, used by the CMS update-frequency cadence.
            mask: [B, N] boolean patch mask (True = target). If None, a fresh
                  random mask at `mask_ratio` is sampled. CLS (position 0) is
                  never masked.
        Returns:
            dict with:
              layer_embeddings: list of [B, N+1, d] per layer.
              pooled: [B, d] CLS embedding from the final layer (for probing).
              mask: [B, N] the mask actually used.
              jepa_losses: list of per-layer scalar losses.
        """
        from .data import random_mask

        x = self.embed_patches(images)                         # [B, N+1, d]
        B = x.shape[0]
        device = x.device

        if mask is None:
            mask = random_mask(B, self.num_patches, self.mask_ratio, device)

        layer_embeddings = []
        jepa_losses = []
        z = x
        for hope_layer, jepa_head in zip(self.hope_layers, self.jepa_heads):
            z = hope_layer(z, global_step)                     # [B, N+1, d]
            layer_embeddings.append(z)
            # JEPA operates on patch tokens (drop CLS), using this layer's own
            # embeddings as targets. Detach the *input* to the target path? No:
            # LeJEPA does not use stop-gradient. SIGReg prevents collapse.
            from .jepa import jepa_layer_loss
            z_patches = z[:, 1:, :]                             # [B, N, d]
            jepa_losses.append(jepa_layer_loss(z_patches, mask, jepa_head))

        pooled = layer_embeddings[-1][:, 0, :]                 # CLS of last layer
        return {
            "layer_embeddings": layer_embeddings,
            "pooled": pooled,
            "mask": mask,
            "jepa_losses": jepa_losses,
        }

    # ------------------------------------------------------------------
    def encode(self, images: torch.Tensor, global_step: int = 0) -> torch.Tensor:
        """Convenience: return just the pooled [B, d] embedding for the probe."""
        x = self.embed_patches(images)
        for hope_layer in self.hope_layers:
            x = hope_layer(x, global_step)
        return x[:, 0, :]
