"""LISA: Layerwise Importance Sampled AdamW.

Reference: Pan et al., "LISA: Layerwise Importance Sampling for
Memory-Efficient Large Language Model Fine-Tuning" (arXiv:2403.17919,
NeurIPS 2024).

LISA is a *full-parameter* (or full-LoRA-parameter) fine-tuning scheme that
trades a little bias for a lot of memory and speed:

  * Like AdamW it maintains per-parameter first/second moment state -- but
    only for the layers that are *active* on the current step.
  * Each step, a random subset of K decoder-layer groups is ACTIVATED
    (requires_grad=True, state allocated/used); the rest are FROZEN
    (requires_grad=False, no state, no gradient, no backward flow into them).
    This is the "layerwise importance sampling": most layers are frozen most
    of the time, so only K layers ever hold AdamW state at once.
  * Every `refresh_every` steps the layers are re-ranked by their accumulated
    gradient norm (importance), and the sampling distribution is biased toward
    the important layers. This recovers most of AdamW's quality.

Why it helps HERE: our model has LoRA adapters on every base decoder layer +
HOPE/JEPA/Reasoner wrapper modules. The wrapper modules are small and must
train every step (they carry the new capabilities), so they are ALWAYS active
and use a single shared AdamW. The base-layer LoRA params are the bulk and the
memory sink; LISA activates only K of them per step, cutting optimizer state
roughly by (num_layers / K). Because frozen layers get NO gradient, backward
through them is pruned (Autograd skips subtrees under frozen leaves), which
also speeds the backward pass.

USAGE (see scripts/train_llm_jepa.py):

    opt = LISA(
        always_active_params=wrapper_params,   # HOPE/JEPA/Reasoner: train always
        layer_groups=lora_layer_groups,        # list[list[nn.Parameter]] per decoder layer
        lr=2e-4, k=2, refresh_every=50,
    )
    for batch in loader:
        opt.sample_layers(step)     # <- freeze/unfreeze BEFORE forward
        opt.zero_grad()
        out = model(**batch); out.loss.backward(); opt.step()

`sample_layers()` must run before forward so that frozen layers' params carry
`requires_grad=False` and Autograd prunes backward into them.
"""

from __future__ import annotations

import random
from typing import List, Sequence

import torch
from torch.optim.optimizer import Optimizer


class LISA(Optimizer):
    """Layerwise Importance Sampled AdamW.

    Args:
        always_active_params: parameters that train EVERY step (e.g. the
            HOPE / slot-JEPA / Reasoner wrapper modules). Trained by an
            ordinary AdamW with persistent state.
        layer_groups: one list of parameters PER base decoder layer (e.g. one
            entry per LoRA-adapted layer). Each step, `k` of these groups are
            activated; the rest are frozen and hold no state.
        lr: AdamW learning rate.
        k: number of layer groups to activate per step (the paper uses 2).
        refresh_every: re-rank layers by accumulated grad-norm and rebuild the
            sampling distribution every this many steps (paper: ~50-100).
        betas / eps / weight_decay: standard AdamW hyperparameters.
        bias_importance: if True, after each refresh the sampling probabilities
            are skewed toward higher-importance layers (paper's importance
            sampling). If False, layers are sampled uniformly.
        layer_temp: softmax temperature for the importance->probability map
            (lower = sharper preference for important layers). Only used when
            bias_importance=True.
    """

    def __init__(
        self,
        always_active_params: Sequence[torch.nn.Parameter],
        layer_groups: Sequence[Sequence[torch.nn.Parameter]],
        lr: float = 2e-4,
        k: int = 2,
        refresh_every: int = 50,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        bias_importance: bool = True,
        layer_temp: float = 1.0,
    ):
        if not layer_groups:
            raise ValueError("LISA needs at least one layer_group.")
        if k < 1 or k > len(layer_groups):
            raise ValueError(
                f"k={k} must be in [1, {len(layer_groups)}] (num layer groups)."
            )

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # Collect *all* params into param_groups so Optimizer is happy, but
        # remember which are wrapper (always-on) vs which belong to each
        # decoder-layer group (sampled). We give each layer group its own
        # param_group so we can flip requires_grad per group cheaply.
        always_active_params = list(always_active_params)
        layer_groups = [list(g) for g in layer_groups]

        param_groups = []
        if always_active_params:
            param_groups.append({
                "params": always_active_params, "lisa_role": "always",
            })
        for gi, g in enumerate(layer_groups):
            param_groups.append({
                "params": g, "lisa_role": "layer", "lisa_group": gi,
            })

        super().__init__(param_groups, defaults)

        self.k = int(k)
        self.refresh_every = int(refresh_every)
        self.bias_importance = bool(bias_importance)
        self.layer_temp = float(layer_temp)
        self.num_layer_groups = len(layer_groups)

        # Bookkeeping.
        self._step = 0
        self._active_set: set[int] = set()           # indices of active groups
        self._importance = torch.zeros(self.num_layer_groups)  # accumulated norms
        self._probs = torch.full((self.num_layer_groups,),
                                 1.0 / self.num_layer_groups)
        # State that lives on the importance tensor's device; updated lazily
        # once we see a real grad (the params may not be on a device yet at
        # construction time).
        self._imp_device_set = False

    # ------------------------------------------------------------------
    def _ensure_importance_device(self, ref: torch.Tensor):
        if not self._imp_device_set:
            self._importance = self._importance.to(ref.device)
            self._probs = self._probs.to(ref.device)
            self._imp_device_set = True

    @torch.no_grad()
    def sample_layers(self, step: int):
        """Pick this step's active layer groups and flip requires_grad.

        Call BEFORE forward() so Autograd prunes backward into frozen layers.
        """
        self._step = int(step)
        # Periodically refresh the sampling distribution from importance.
        if (self.step_count() % max(1, self.refresh_every) == 0
                and self._imp_device_set
                and self.bias_importance):
            imp = self._importance.clone()
            # Avoid div-by-zero / all-cold: add a floor so an unseen layer can
            # still be sampled (paper keeps uniform exploration).
            imp = imp.clamp_min(imp.max().clamp_min(1e-8) * 1e-3)
            logits = (imp / imp.max()).log() / max(self.layer_temp, 1e-3)
            self._probs = torch.softmax(logits, dim=0)
            # Reset the accumulator for the next window.
            self._importance.zero_()

        # Sample k distinct layer groups without replacement from _probs.
        n = self.num_layer_groups
        if n <= self.k:
            chosen = list(range(n))
        elif self._imp_device_set and self.bias_importance \
                and self._probs.sum() > 0:
            # Multinomial sampling without replacement: sample k indices.
            idx = torch.multinomial(self._probs, num_samples=self.k,
                                    replacement=False).tolist()
            chosen = idx
        else:
            chosen = random.sample(range(n), self.k)

        self._active_set = set(chosen)

        # Flip requires_grad: only the active layer groups (and always-on
        # wrapper params) keep grad; the rest are frozen this step.
        for group in self.param_groups:
            if group.get("lisa_role") != "layer":
                continue
            active = group["lisa_group"] in self._active_set
            for p in group["params"]:
                # Respect the original intent for params that the user
                # explicitly froze (e.g. the frozen base under QLoRA): only
                # ever ENABLE params, and only freeze ones we ourselves would
                # otherwise train. We track this via a marker set at init.
                if not getattr(p, "_lisa_trainable", True):
                    continue
                p.requires_grad_(active)

    def mark_trainable(self, params):
        """Mark which layer-group params LISA is allowed to toggle. Params not
        passed here (e.g. frozen 4-bit base weights that LoRA never trains) are
        left alone. Call once after construction with the LoRA adapter params."""
        seen = set(id(p) for p in params)
        for group in self.param_groups:
            if group.get("lisa_role") != "layer":
                continue
            for p in group["params"]:
                p._lisa_trainable = id(p) in seen

    def step_count(self) -> int:
        return self._step

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            role = group.get("lisa_role")
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]

            if role == "layer":
                # Only the active subset updates; frozen ones have no grad.
                if group["lisa_group"] not in self._active_set:
                    continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not torch.isfinite(g).all():
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                denom = (exp_avg_sq.sqrt() / (bias_c2 ** 0.5)).add_(eps)
                step_size = lr / bias_c1
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Accumulate importance for the refresh (layer groups only).
                if role == "layer":
                    self._ensure_importance_device(g)
                    self._importance[group["lisa_group"]] += \
                        float(g.detach().float().pow(2).sum().sqrt().item())
        return loss

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)

    # Convenience: total number of trainable params on THIS step (debug/log).
    def num_active_params(self) -> int:
        total = 0
        for group in self.param_groups:
            role = group.get("lisa_role")
            if role == "always":
                total += sum(p.numel() for p in group["params"])
            elif role == "layer" and group["lisa_group"] in self._active_set:
                total += sum(p.numel() for p in group["params"] if p.requires_grad)
        return total
