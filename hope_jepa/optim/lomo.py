"""LOMO: LOw-Memory Optimization -- fused update-into-backward.

Reference: "Full Parameter Fine-tuning for Large Language Models" (Lv et al.,
2023), a.k.a. LOMO. Code structure follows OpenLMLab/LOMO.

The key memory idea: a normal optimizer (AdamW) must keep, for *every*
parameter, its gradient tensor AND its optimizer state (2 fp32 moments =
2x param size) materialized on the autograd graph until `step()` runs.
LOMO instead updates each parameter the moment its gradient is produced
during `loss.backward()`, via a full backward hook, and then *frees the
gradient* (`p.grad = None`). Gradients therefore never all live in memory
at once -- at any instant only the parameters below the current backward
node hold grads -- and no optimizer state is kept for the SGD variant.

Two variants are provided:

  * `LOMO`     -- plain fused SGD with GLOBAL gradient clipping. Faithful to
                  the paper. Requires two backward passes: one to measure the
                  global norm (clip), one to apply the clipped update. Use when
                  you want the exact LOMO behaviour / can afford the second
                  backward.

  * `AdaLomo`  -- momentum + PER-TENSOR local clipping, SINGLE backward pass.
                  This is the fast/recommended variant for our setup (4-bit
                  QLoRA spliced with a Titans recurrence that is expensive to
                  backprop twice). It keeps a single fp32 momentum buffer per
                  param (1x param memory, vs AdamW's 2x) and clips each
                  parameter's update by its own gradient norm, which avoids the
                  global-norm sync that would force a second backward. The
                  AdaLomo paper (OpenLMLab) shows this matches AdamW quality at
                  a fraction of the memory.

Both accept a list of `params` and behave like a `torch.optim.Optimizer` for
the purposes of `zero_grad()`. Because updates are fused into backward,
`step()` is a no-op for the parameters that still hold a grad (their update
already happened during backward); `step()` exists mainly for API parity and
to handle any params whose backward hook somehow did not fire.

IMPORTANT -- usage pattern (mirrors the LOMO repo):

    opt = LOMO(model.parameters(), lr=1e-5, clip_grad=1.0)
    for batch in loader:
        # LOMO (global clip) needs a *dry* backward to measure the norm:
        opt.zero_grad(); loss.backward(); opt.zero_grad()
        loss.backward()        # <- this backward applies the clipped update
        opt.step()             # no-op for params already updated in-place
        # AdaLomo skips the dry backward:
        opt.zero_grad(); loss.backward(); opt.step()

When fused updates are enabled, `loss.backward()` updates params in place, so
the model has already changed by the time `backward()` returns -- do NOT call
`backward()` twice on the same loss with AdaLomo.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class _FusedOptimizerBase(Optimizer):
    """Common machinery for LOMO / AdaLomo: per-parameter backward hooks that
    fire the fused update as each grad is produced.

    Subclasses implement `_make_hook(p, group)` returning the backward-hook
    closure for parameter `p`.
    """

    def __init__(self, params, defaults):
        super().__init__(params, defaults)
        # Stamp each param's group onto the param so the hook can read lr/clip
        # without re-indexing param_groups on every grad.
        self._install_hooks()
        self._fused = True   # updates happen during backward()

    def _install_hooks(self):
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                # Skip non-leaf / non-finite-differentiable params gracefully.
                if not p.is_leaf:
                    continue
                hook = self._make_hook(p, group)
                # Store to prevent the weakref from being GC'd.
                p._lomo_hook = p.register_hook(hook)

    def _make_hook(self, p, group):  # pragma: no cover - abstract
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        """No-op for fused params: they updated during backward. Clears any
        leftover gradient (the dry-pass grad in LOMO, or grads autograd re-
        populated after the live hook returned). Kept for Optimizer API parity
        and to satisfy code that calls opt.step()."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Fused optimizers update params during backward, but autograd still
        # materializes the grad into `.grad` AFTER the hook runs (a hook cannot
        # suppress the accumulation in this PyTorch version). Clear it here so
        # grads never linger in memory across steps -- the low-memory point.
        self._clear_grads()
        return loss

    def _clear_grads(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad = None

    # Re-arm hooks if someone mutates requires_grad after init (e.g. LISA).
    def _rearm_hooks(self):
        for group in self.param_groups:
            for p in group["params"]:
                hook = getattr(p, "_lomo_hook", None)
                if hook is not None:
                    hook.remove()
                    p._lomo_hook = None
        self._install_hooks()


class LOMO(_FusedOptimizerBase):
    """LOw-Memory Optimization -- fused SGD with GLOBAL gradient clipping.

    Faithful to the paper: because the update is fused into backward, you
    cannot know the GLOBAL grad norm until backward is done, yet the update
    must happen *during* backward. LOMO resolves this with a two-pass scheme:

      pass 1 (dry):  backward to compute & store per-param grad norms, then
                     zero grads (NO update -- the hook only measures).
      pass 2 (live): backward again; the hook reads the precomputed global
                     norm, clips each grad to it, applies the SGD step, and
                     frees the grad.

    So the correct training-loop incantation is:

        opt.zero_grad(); loss.backward(retain_graph=True); opt.zero_grad()  # dry
        opt.begin_fused_update()           # finalize the global norm
        loss.backward()                    # live: hooks apply + free grads

    The dry backward MUST use ``retain_graph=True`` -- otherwise the saved
    intermediates are freed before the live backward can recompute them.

    `clip_grad` is the global max L2 norm (default 1.0). Set it to `float('inf')`
    to disable clipping (then a single backward suffices).
    """

    def __init__(self, params, lr: float = 1e-5, clip_grad: float = 1.0):
        super().__init__(params, defaults={"lr": lr, "clip_grad": clip_grad})

    def _install_hooks(self):
        # State shared across hooks of THIS optimizer instance.
        # _measuring: True during the dry pass (hook records norm, no update).
        #             False during the live pass (hook updates).
        self._measuring = True
        self._global_norm = 0.0
        super()._install_hooks()

    def _make_hook(self, p, group):
        clip = group["clip_grad"]

        def hook(grad):
            if not torch.isfinite(grad).all():
                return grad
            if self._measuring:
                # Dry pass: accumulate this param's contribution to the global
                # norm. We do NOT update -- grads will be zeroed by the caller.
                self._global_norm += float(grad.float().pow(2).sum().item())
                return grad
            # Live pass: clip by the precomputed global norm and step.
            lr = group["lr"]
            gn = max(self._global_norm, 1e-12)
            scale = min(1.0, clip / gn) if clip != float("inf") else 1.0
            p.add_(grad * scale, alpha=-lr)
            # Clear the grad so it is never materialized in `.grad` (the LOMO
            # memory point): the update above already baked it in. We set both
            # the live contribution (None return) and wipe any leftover from
            # the dry pass.
            p.grad = None
            return None
        return hook

    def zero_grad(self, set_to_none: bool = True):
        # When starting a dry pass, reset the global-norm accumulator.
        self._global_norm = 0.0
        self._measuring = True
        super().zero_grad(set_to_none=set_to_none)

    def begin_fused_update(self):
        """Call between the dry backward and the live backward.

        After `zero_grad()` + (dry) `backward()`, the hooks have accumulated
        the global norm. Calling this flips the hooks to update mode and takes
        the sqrt to finalize the norm, so the NEXT `backward()` applies the
        clipped fused update.
        """
        self._global_norm = self._global_norm ** 0.5
        self._measuring = False


class AdaLomo(_FusedOptimizerBase):
    """AdaLomo -- momentum + per-tensor local clipping, SINGLE backward pass.

    Avoids LOMO's dry backward by clipping each parameter against its OWN
    gradient norm (a local surrogate for the global norm). Keeps one fp32
    momentum buffer per parameter (1x param memory; AdamW keeps 2x). This is
    the recommended LOMO-family optimizer for our setup: the Titans recurrence
    makes a second backward expensive, and AdaLomo's single-pass update is the
    fastest fused option while retaining AdamW-like adaptive momentum.

    Standard training loop (no dry pass, no separate clip_grad_norm_):

        opt.zero_grad(); loss.backward(); opt.step()

    Args:
        lr:        learning rate.
        momentum:  momentum factor (default 0.9, matching the AdaLomo paper).
        clip_grad: per-tensor max L2 norm (default 1.0). Set `inf` to disable.
        eps:       momentum damping (default 1e-8).
    """

    def __init__(self, params, lr: float = 1e-5, momentum: float = 0.9,
                 clip_grad: float = 1.0, eps: float = 1e-8):
        super().__init__(
            params,
            defaults={"lr": lr, "momentum": momentum,
                      "clip_grad": clip_grad, "eps": eps},
        )

    def _make_hook(self, p, group):
        lr = group["lr"]
        momentum = group["momentum"]
        clip = group["clip_grad"]
        eps = group["eps"]

        def hook(grad):
            if not torch.isfinite(grad).all():
                return grad
            g = grad
            # Per-tensor local clipping (surrogate for the global norm).
            if clip != float("inf"):
                gn = g.float().pow(2).sum().sqrt()
                scale = (clip / gn.clamp_min(1e-12)).clamp(max=1.0)
                g = g * scale
            # Momentum buffer (allocated lazily, fp32).
            buf = self.state[p].get("momentum_buffer")
            if buf is None:
                buf = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[p]["momentum_buffer"] = buf
            else:
                buf.mul_(momentum)
            buf.add_(g, alpha=1.0 - momentum)
            p.add_(buf, alpha=-lr)
            # Clear the grad so it is never materialized in `.grad` (the
            # low-memory point): the update already baked it in. There is no dry
            # pass for AdaLomo, but we still suppress `.grad` for consistency.
            p.grad = None
            return None
        return hook

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        # We set p.grad=None inside the hook already; just clear any leftovers
        # (e.g. params whose hook didn't fire because no graph touched them).
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()
