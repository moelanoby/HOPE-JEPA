"""A HOPE (Nested-Learning) layer.

Each HOPE layer = Self-Modifying Titans memory mixer (`titans.py`) + a
Continuum Memory System: a stack of FFN Neural Learning Modules (NLMs) that
operate at *staggered update frequencies* (module k fires every base*2^k steps),
embodying Nested Learning's multi-timescale / multi-frequency structure.

The frequency cadence is realized with straight-through gating: a module whose
cadence does not fire on the current global step contributes via a
straight-through estimator (its gradient still flows, but its forward is the
running average), so backprop is stable and every module is trained.

Reference: Behrouz et al., "Nested Learning: The Illusion of Deep Learning
Architectures", arXiv:2512.24695 (the "Hope" model).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .titans import MACMixer


class NeuralLearningModule(nn.Module):
    """One FFN 'Neural Learning Module' of the Continuum Memory System.

    Has its own local objective scale (a learnable per-module gain on its
    residual contribution) and an EMA of its hidden activations used when its
    update cadence does not fire (keeps the forward stable across its off-steps).
    """

    def __init__(self, d_model: int, d_ff: int, momentum: float = 0.9):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )
        # Per-module learnable gain; the "local objective" weight in HOPE.
        self.gain = nn.Parameter(torch.ones(1))
        self.register_buffer("hidden_ema", torch.zeros(d_model), persistent=False)
        self.momentum = momentum

    def forward(self, x: torch.Tensor, active: bool) -> torch.Tensor:
        h = self.ffn(self.norm(x))
        if active:
            # Update the EMA of the module's output (per-batch mean).
            with torch.no_grad():
                self.hidden_ema.mul_(self.momentum).add_(
                    h.detach().mean(dim=(0, 1)), alpha=1 - self.momentum
                )
            return self.gain * h
        # Off-step: straight-through -- forward as the EMA baseline, but let
        # gradients flow into h via (h - h.detach()) so the module still learns.
        baseline = self.hidden_ema.view(1, 1, -1)
        return self.gain * (baseline + (h - h.detach()))


class ContinuumMemorySystem(nn.Module):
    """Stack of NLMs with staggered update frequencies."""

    def __init__(self, d_model: int, num_modules: int, base_update_freq: int = 1,
                 d_ff_multiplier: int = 2):
        super().__init__()
        self.base = base_update_freq
        d_ff = d_model * d_ff_multiplier
        self.modules_list = nn.ModuleList([
            NeuralLearningModule(d_model, d_ff) for _ in range(num_modules)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def _is_active(self, module_idx: int, global_step: int) -> bool:
        cadence = self.base * (2 ** module_idx)
        return (global_step % cadence) == 0

    def forward(self, x: torch.Tensor, global_step: int) -> torch.Tensor:
        for k, nlm in enumerate(self.modules_list):
            x = x + nlm(x, active=self._is_active(k, global_step))
        return self.out_norm(x)


class HopeLayer(nn.Module):
    """A single HOPE layer: Titans memory mixer followed by the CMS."""

    def __init__(self, d_model: int, d_hidden: int, num_persistent_memory: int,
                 cms_num_modules: int, cms_base_update_freq: int,
                 cms_d_ff_multiplier: int, dropout: float = 0.0):
        super().__init__()
        self.mixer = MACMixer(d_model, d_hidden, num_persistent_memory)
        self.cms = ContinuumMemorySystem(
            d_model, cms_num_modules, cms_base_update_freq, cms_d_ff_multiplier,
        )
        self.drop = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, global_step: int) -> torch.Tensor:
        """tokens: [B, N, d] -> [B, N, d]."""
        x = self.mixer(tokens)
        x = self.cms(x, global_step)
        return self.final_norm(self.drop(x))
