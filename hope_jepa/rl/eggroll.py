"""EGGROLL: rank-r evolution strategies optimizer (arXiv:2511.16652).

This is the *parameter update* half of the self-play loop. It is a black-box
optimizer -- no gradients, no critic, no value model -- which is exactly why it
fits an adversarial self-play objective: the reward (did a fixer pass the real
test suite? did the injector's bug survive?) is discrete and non-differentiable,
and GRPO/PPO would need a value model and differentiable proxies that we don't
have. EGGROLL consumes a single scalar FITNESS per population member and updates
the shared center theta in the fitness-weighted perturbation direction.

Algorithm (one generation):

    # 1. sample P rank-r perturbations of the center theta
    for p in range(P):
        eps_p  = low_rank_noise(theta_shapes, r)   # eps = A @ B^T, A,B ~ N(0,1)
        theta_p = theta + sigma * eps_p            (+ antithetic: also theta - sigma*eps_p)

    # 2. evaluate each member -> scalar fitness f_p
    #    (GTPO for injectors, GRPO-S for fixers -- see reward.py)

    # 3. rank-shape the fitnesses into utilities u_p
    u = rank_utilities(f, shape="log")             # top member -> largest |u|

    # 4. update the center toward high-utility perturbations
    theta <- theta + (eta * sigma / r) * sum_p u_p * eps_p

KEY EFFICIENCY LEVER (the paper's headline): each perturbation is rank-r
(eps = A @ B^T, r << d) instead of full-rank unstructured noise. This raises
GPU arithmetic intensity ~100x at hyperscale, so a large population is cheap.
Here we materialize eps as a dense tensor per param (fine for the LoRA-scale
trainable set); the low-rank factors are how it's sampled, and the (1/r)
factor in the update normalizes for the rank.

Why only the TRAINABLE params: `train_llm_jepa.apply_qlora` freezes the
quantized base and leaves only LoRA adapters + the new HOPE/JEPA modules
trainable. EGGROLL perturbs exactly that small set -- a 7B model's perturbation
lives in the LoRA space, not the 7B weight space.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Sampling rank-r perturbations
# ---------------------------------------------------------------------------
def _low_rank_noise(
    shape: torch.Size,
    rank: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """A single rank-r perturbation tensor of the given shape.

    eps = A @ B^T where for a shape [..., d_in, d_out] we sample A in
    R^{d_in x r} and B in R^{d_out x r}, both ~ N(0, 1), then Frobenius-normalize
    the result to unit norm so `sigma` is the sole scale knob. For non-matrix
    shapes (1-D / higher-D) we flatten to a 2-D [prod(leading), last] view, build
    the low-rank factor there, and reshape back. rank is clamped to >=1 and to
    the last dim.
    """
    if not shape:
        return torch.zeros((), device=device)
    d_last = int(shape[-1])
    d_lead = 1
    for s in shape[:-1]:
        d_lead *= int(s)
    r = max(1, min(rank, d_last, d_lead))
    A = torch.randn(d_lead, r, device=device, generator=generator)
    B = torch.randn(d_last, r, device=device, generator=generator)
    eps = A @ B.t()                       # [d_lead, d_last]
    eps = eps.reshape(shape)
    norm = eps.norm() + 1e-12
    return eps / norm


def sample_perturbation(
    named_params: List[Tuple[str, torch.Tensor]],
    rank: int,
    sigma: float,
    antithetic: bool,
    device: torch.device,
    generator: torch.Generator,
) -> List[torch.Tensor]:
    """Sample one (or one antithetic pair of) rank-r perturbation(s).

    Args:
        named_params: the center's trainable params (name, tensor). Shapes and
            device are read from these.
        rank: EGGROLL low-rank r.
        sigma: perturbation scale (eps is unit-norm; sigma multiplies it).
        antithetic: if True, return TWO perturbations per member: +eps and -eps
            (the standard antithetic-variance trick; the caller evaluates both
            and they share one noise draw).
    Returns:
        list of perturbation tensors aligned with `named_params`. If antithetic,
        returns a tuple (plus_eps, minus_eps), each a list of tensors.
    """
    plus = [sigma * _low_rank_noise(p.shape, rank, device, generator)
            for _, p in named_params]
    if not antithetic:
        return plus
    minus = [-e for e in plus]
    return plus, minus


def apply_perturbation(
    center_state: Dict[str, torch.Tensor],
    eps: List[torch.Tensor],
    names: List[str],
) -> Dict[str, torch.Tensor]:
    """Return theta + eps as a NEW state dict (does not mutate center_state).

    `eps` and `names` must be aligned (same order). Used to build a population
    member's weights for an inference-only rollout -- the caller loads this into
    a model copy via `load_state_dict`.
    """
    return {n: center_state[n].clone().add_(e.to(center_state[n].device))
            for n, e in zip(names, eps)}


# ---------------------------------------------------------------------------
# 2. Rank-shaped utilities (the "best contributes most" mechanism)
# ---------------------------------------------------------------------------
def rank_utilities(
    fitnesses: torch.Tensor,
    shape: str = "log",
) -> torch.Tensor:
    """Map raw fitnesses to rank-shaped utilities u_p (zero-mean, normalized).

    This is the core of "the best fixer per batch contributes more": we rank
    members by fitness, assign each rank a utility weight (largest for the best),
    then zero-mean + normalize so the EGGROLL update is a contrastive move
    *toward* high-fitness perturbations and *away* from low-fitness ones.

    Two shaping modes (the paper/ANTES lineage):
      * "log"    -- u(rank) = max(0, log(P/2 + 1) - log(rank))  for rank=1..P,
                    i.e. the classic CMA-ES / NES log utility. The top members
                    get strong positive weight; the bottom get ~0 (not negative),
                    so the update is pulled toward the elite.
      * "linear" -- u(rank) = (P - rank) / P, a softer linear falloff.

    After shaping we zero-mean (so it's a relative/advantage-like signal) and
    normalize so ||u||_2 = sqrt(P). That keeps the update magnitude independent
    of the population size and the fitness scale.

    Args:
        fitnesses: [P] raw scalar fitness per member (higher = better).
        shape: "log" or "linear".
    Returns:
        [P] utilities aligned with the input order.
    """
    P = fitnesses.numel()
    if P == 0:
        return fitnesses.new_zeros(0)
    if P == 1:
        return fitnesses.new_zeros(1)          # nothing to contrast against

    # ranks: best (HIGHEST-fitness) member -> rank 1, so it gets the largest
    # utility weight and the center update is pulled toward it most strongly.
    # argsort descending puts the best first; we stamp rank 1..P in that order.
    order = torch.argsort(fitnesses, stable=True, descending=True)
    ranks = torch.empty_like(fitnesses)                # same dtype as fitnesses
    ranks[order] = torch.arange(1, P + 1, device=fitnesses.device,
                                dtype=fitnesses.dtype)

    if shape == "log":
        # CMA-ES/NES log utility: elite (small rank) get positive weight.
        import math
        w = torch.clamp(math.log(P / 2.0 + 1.0) - torch.log(ranks), min=0.0)
    elif shape == "linear":
        w = (P - ranks + 1).to(fitnesses.dtype) / P
    else:
        raise ValueError(f"Unknown utility_shape {shape!r}; use 'log' or 'linear'.")

    w = w - w.mean()                       # zero-mean -> contrastive
    norm = w.norm() + 1e-12
    return w * (P ** 0.5) / norm            # normalize so ||u||_2 = sqrt(P)


# ---------------------------------------------------------------------------
# 3. The center update
# ---------------------------------------------------------------------------
def eggroll_update(
    center_model: nn.Module,
    perturbations: List[List[torch.Tensor]],
    utilities: torch.Tensor,
    names: List[str],
    lr: float,
    sigma: float,
    rank: int,
) -> Dict[str, float]:
    """Apply the EGGROLL center update IN PLACE: theta <- theta + dtheta.

        dtheta = (eta * sigma / r) * sum_p u_p * eps_p

    The (1/r) normalizes for the rank of the perturbations (higher rank -> more
    noise directions -> scale up the step accordingly). `u_p` are the rank-shaped
    utilities from `rank_utilities` -- the top member's eps gets the largest
    weight, so the center moves most strongly toward the perturbation that won.

    No gradients are computed; this is pure tensor arithmetic. Members were
    evaluated with inference-only rollouts, so nothing here touches autograd.

    Args:
        center_model: the shared center; its trainable params are updated in place.
        perturbations: list (len P) of perturbation lists (each aligned with `names`).
        utilities: [P] utilities from `rank_utilities`.
        names: param names aligned with each perturbation list's order.
        lr: eta.
        sigma: perturbation scale (also the step normalization base).
        rank: the EGGROLL low-rank r used to sample `perturbations`.
    Returns:
        diagnostics dict {mean_abs_step, max_abs_utility, ...}.
    """
    if not perturbations or not names:
        return {"mean_abs_step": 0.0, "max_abs_utility": 0.0}

    step_coef = (lr * sigma) / max(1, rank)
    # Gather center param references by name.
    center_params = dict(center_model.named_parameters())
    max_abs_u = float(utilities.abs().max().item()) if utilities.numel() else 0.0
    total_abs_step = 0.0
    n_elems = 0

    # Each member's perturbation list is aligned with `names` (same order), so
    # we can zip names with each eps_p directly -- no per-name lookup needed.
    utilities_list = utilities.tolist()
    with torch.no_grad():
        for idx, name in enumerate(names):
            p = center_params[name]
            if not p.requires_grad:
                continue                      # never perturb frozen params
            acc = torch.zeros_like(p)
            for eps_p, u_p in zip(perturbations, utilities_list):
                acc.add_(eps_p[idx].to(p.device), alpha=float(u_p))
            delta = step_coef * acc
            p.add_(delta)
            total_abs_step += float(delta.abs().sum().item())
            n_elems += p.numel()

    return {
        "mean_abs_step": (total_abs_step / n_elems) if n_elems else 0.0,
        "max_abs_utility": max_abs_u,
    }
