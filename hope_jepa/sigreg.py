"""SIGReg: Sketched Isotropic Gaussian Regularization (LeJEPA).

Introduced in Balestriero & LeCun, "LeJEPA: Provable and Scalable
Self-Supervised Learning Without the Heuristics" (arXiv:2511.08544), SIGReg is
the regularizer that lets a JEPA target its own network's embeddings without
collapsing. It pushes the covariance of the embedding batch toward an isotropic
Gaussian (cov = sigma^2 * I).

To stay linear in time and memory -- the whole point of the "sketched" variant
-- we estimate the covariance of a **random projection** (sketch) of the
embeddings rather than the full d x d covariance. The sketch dimension m is
much smaller than d (e.g. 32-256). For m < d this is a Johnson-Lindenstrauss-
style reduction; for m >= d it reduces to the full covariance penalty.

We also expose a diagnostic `effective_rank` of the embedding covariance, which
should stay high when SIGReg is doing its job and collapse to ~1 when it is off
-- the canonical demonstration of SIGReg's role.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketched isotropic-Gaussian covariance regularizer.

    Args:
        d_model:     embedding dimensionality d.
        sketch_dim:  sketch dimension m (m <= d means a real sketch; m = d is the
                     full-covariance variant). Use null for m = d.
        target_scale: sigma^2, the target isotropic variance per dimension.
    """

    def __init__(self, d_model: int, sketch_dim: int | None = None,
                 target_scale: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.sketch_dim = sketch_dim or d_model
        self.target_scale = float(target_scale)
        # Fixed random projection matrix R in [d, m], drawn once at init.
        # We do not learn R; keeping it fixed makes the penalty a stable target.
        R = torch.randn(d_model, self.sketch_dim)
        Q, _ = torch.linalg.qr(R)                  # orthonormalize columns
        self.register_buffer("R", Q[:, :self.sketch_dim].contiguous())

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., d] -> [..., m] via the fixed random projection."""
        return z @ self.R

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the SIGReg penalty for an embedding batch.

        Args:
            z: [*, d] embeddings (any leading dims are flattened to a batch).
        Returns:
            scalar penalty = || Cov(sketch(z)) - sigma^2 I ||_F^2  /  m
        """
        z = z.reshape(-1, self.d_model)
        z_s = self.project(z)                       # [n, m]
        m = self.sketch_dim
        # Center, then estimate covariance. We normalize by n-1 for an unbiased
        # estimate; for large batches the difference is negligible.
        n = z_s.shape[0]
        zc = z_s - z_s.mean(dim=0, keepdim=True)
        cov = (zc.t() @ zc) / max(n - 1, 1)         # [m, m]
        eye = torch.eye(m, device=z.device, dtype=z.dtype)
        target = self.target_scale * eye
        return ((cov - target) ** 2).sum() / m

    @torch.no_grad()
    def effective_rank(self, z: torch.Tensor) -> float:
        """Diagnostic: effective rank of the (full) embedding covariance.

        eff_rank = exp(H(p)),  p_i = sv_i / sum(sv_i).  High (~d) = healthy,
        ~1 = collapsed. Computed on the full (un-sketched) covariance for an
        honest collapse signal; cheap because d is modest.
        """
        z = z.reshape(-1, self.d_model)
        if not torch.isfinite(z).all():
            return -1.0
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.t() @ zc) / max(zc.shape[0] - 1, 1)
        if not torch.isfinite(cov).all():
            return -1.0
        # singular values of the PSD covariance == eigenvalues.
        try:
            sv = torch.linalg.svdvals(cov)
        except torch._C._LinAlgError:
            return -1.0
        sv = sv.clamp_min(1e-12)
        p = sv / sv.sum()
        entropy = -(p * torch.log(p)).sum()
        return float(torch.exp(entropy).item())
