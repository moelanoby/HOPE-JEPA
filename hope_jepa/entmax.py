"""Entmax-1.5: a sparse, differentiable mapping onto the probability simplex.

Standard softmax can never output an exact zero -- every slot keeps at least an
infinitesimal weight. For the slot-mixing attention in this model we want a
weight that can be *exactly* zero, so a slot can be fully dropped per target.
Entmax-1.5 (the Tsallis entropy regularizer with alpha=1.5) is the sparse
projection onto the simplex that does this, while remaining smooth enough that
gradients keep flowing to the slots that survive.

Mapping (per row, last dimension is the simplex dimension):

    p_i = (relu(0.5 * z_i - tau))^2 ,   tau chosen so  sum_i p_i = 1.

`tau` is found by bisection (a fixed, generous number of iterations); the
support (where p_i > 0) is the set of "kept" slots, and the rest are exactly 0.

The backward pass uses the closed-form Jacobian of the Tsallis-alpha=1.5 map.
Writing w_i = sqrt(p_i) and s_i = w_i / sum_j w_j, the input gradient is

    grad_z_i = w_i * g_i  -  s_i * sum_j (w_j * g_j) ,    on the support,
    grad_z_i = 0                                          off the support.

This is the sparsemax Jacobian re-weighted by w (which is exactly the sqrt of
the entmax-1.5 output). It is exact and avoids forming any Jacobian matrix.

References:
  Peters et al., "Sparse Sequence-to-Sequence Models" (entmax), arXiv:1905.05702.
  Blondel, "Learning with Fenchel-Young Losses" (sparsemax Jacobian).
"""

from __future__ import annotations

import torch


class Entmax15Function(torch.autograd.Function):
    """Entmax-1.5 along the last dimension of a 2D+ tensor ([*, K])."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, n_iter: int = 50) -> torch.Tensor:
        # Work on a flattened last dim. We only ever call this with the slot
        # dimension on the last axis (see entmax15 below).
        z = z.contiguous()
        orig_shape = z.shape
        z2 = z.reshape(-1, orig_shape[-1])           # [M, K]
        M, K = z2.shape

        # The optimal threshold tau lies in [lo, hi]; bisect on it. The bracket
        # is chosen so it ALWAYS contains the root:
        #   - at tau = 0.5*z_max - 1, the max element alone contributes
        #     (0.5*z_max - tau)^2 = 1, so sum(p) >= 1  -> f(lo) >= 1.
        #   - at tau = 0.5*z_max, every element gives 0, so sum(p) = 0  -> f(hi)=0.
        # By monotonicity of sum(p) in tau, the root (sum = 1) lies in between.
        # (The naive bracket [min(z), max(z)] fails when the root lies outside
        # it -- e.g. uniform logits need a negative threshold.)
        z_max = z2.max(dim=-1).values                # [M]
        lo = 0.5 * z_max - 1.0                       # [M]
        hi = 0.5 * z_max                             # [M]
        p = torch.empty_like(z2)
        for _ in range(n_iter):
            tau = 0.5 * (lo + hi)                    # [M]
            # p_i = (relu(0.5*z_i - tau))^2  -> broadcast tau over columns.
            cand = torch.clamp(0.5 * z2 - tau.unsqueeze(-1), min=0.0)
            cand = cand * cand                       # [M, K]
            s = cand.sum(dim=-1)                     # [M]
            # If sum(p) > 1, tau is too small -> raise lo; else lower hi.
            too_small = (s > 1.0)
            lo = torch.where(too_small, tau, lo)
            hi = torch.where(too_small, hi, tau)
            p = cand

        # Clamp tiny numerical dust so p sums to exactly 1 on the support.
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        ctx.save_for_backward(p)
        return p.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (p,) = ctx.saved_tensors
        orig_shape = grad_output.shape
        g = grad_output.reshape(-1, p.shape[-1])     # [M, K]
        support = p > 0
        w = torch.sqrt(p.clamp_min(0.0))             # w_i = sqrt(p_i)
        wg = w * g                                   # [M, K]
        s = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        grad_z = wg - s * wg.sum(dim=-1, keepdim=True)
        # Zero out off-support entries (p=0 -> w=0 already does this, but we
        # make it explicit so floating dust on the threshold is killed).
        grad_z = grad_z * support.to(grad_z.dtype)
        grad_z = grad_z.reshape(orig_shape)
        return grad_z, None


def entmax15(z: torch.Tensor, dim: int = -1, n_iter: int = 50) -> torch.Tensor:
    """Entmax-1.5 sparse projection onto the simplex.

    Args:
        z:      logits; the simplex dimension must be `dim`.
        dim:    dimension to project over. Only `dim == -1` (the trailing axis)
                is supported -- both call sites in this model mix over the slot
                axis, which is always last.
        n_iter: bisection iterations for the threshold. 50 gives full float
                precision in practice.
    Returns:
        weights on the simplex; entries off the support are exactly 0.
    """
    assert dim == -1 or dim == z.ndim - 1, "entmax15 only supports the last dim"
    return Entmax15Function.apply(z, n_iter)
