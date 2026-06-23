"""Thread the CMS `global_step` cadence into an HF model without changing
forward signatures.

The Continuum Memory System (`hope_jepa.hope.ContinuumMemorySystem`) staggers
its Neural Learning Module updates by `global_step % (base * 2**k) == 0`.
In the image-SSL trainer this is a plain int incremented once per optimizer
step and passed as a forward kwarg (`train_ssl.py:111`). HuggingFace's
`*ForCausalLM.forward` cannot accept an extra argument, so we instead keep a
process-wide counter (`hope_block.set_global_step_value`) and, as a belt-and-
braces measure, stamp it onto every HOPE block's `.global_step` attribute.

Usage in a training loop (mirrors `train_ssl.py`):

    for step, batch in enumerate(loader):
        out = model(**batch)
        loss = out.loss
        loss.backward(); opt.step()
        set_global_step(model, step + 1)   # advance the cadence
"""

from __future__ import annotations

import torch.nn as nn

from .hope_block import (
    MACAttnAdapter,
    HopeDecoderLayer,
    set_global_step_value,
)


def set_global_step(model: nn.Module, step: int) -> None:
    """Set the process-wide step counter and mirror it on every HOPE block.

    Call once per optimizer step, *after* `opt.step()`. The authoritative value
    the CMS reads comes from `get_global_step()` inside each HOPE block; we also
    write it to each block's `.global_step` attribute for discoverability /
    debugging (it is not read by the forward pass).
    """
    step = int(step)
    set_global_step_value(step)
    # Stamp the attribute on every HOPE-shaped module in the tree (covers both
    # HopeDecoderLayer instances and MACAttnAdapter self_attn sub-blocks).
    for mod in model.modules():
        if isinstance(mod, (HopeDecoderLayer, MACAttnAdapter)):
            mod.global_step = step


def get_global_step() -> int:
    """Return the current process-wide step (for the training-loop's records)."""
    from .hope_block import get_global_step as _g
    return _g()
