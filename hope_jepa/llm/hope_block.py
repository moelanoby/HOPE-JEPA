"""HOPE layers shaped to drop into a HuggingFace Llama-family decoder stack.

The image-SSL `HopeLayer` (see `hope_jepa/hope.py`) is a sequence-to-sequence
transform `[B, N, d] -> [B, N, d]` (Titans memory mixer + Continuum Memory
System). Here we re-wrap it so it can live *inside* an HF `*ForCausalLM` model
in place of (or beside) a `LlamaDecoderLayer`.

Two shapes are provided:

  * `HopeDecoderLayer` -- a full HOPE layer dressed up as a decoder layer.
    Its `forward` matches the HF decoder-layer contract
    `(hidden_states, attention_mask, position_ids, past_key_values,
       use_cache, position_embeddings, **kwargs) -> tuple`.
    This is used by the "replace" and "insert" placement modes in `surgery.py`.

  * `MACAttnAdapter` -- the Titans `MACMixer` dressed up as a `self_attn`
    sub-module (the `forward(hidden_states, attention_mask, ...)` a decoder
    layer calls on its attention block). This is used by the "swap_attention"
    mode, which keeps the decoder layer's MLP / RMSNorms and only swaps the
    attention sub-block for a memory-as-context mixer.

The `global_step` knob (which drives ONLY the CMS staggered-update cadence --
see `hope.py::ContinuumMemorySystem._is_active`) cannot be passed through
HF's `*ForCausalLM.forward` signature, so both shapes read it from a module
attribute `self.global_step` that `step.set_global_step` stamps onto every
HOPE block before each batch.

Known limitation: the MAC mixer is not a token-position-keyed attention, so it
has no KV cache. We therefore return a no-op cache and `use_cache=False`,
which means autoregressive generation *recomputes* through HOPE blocks. This is
correct, not fast -- fine for the QLoRA finetune and for correctness checks;
flagged for a future KV-aware variant.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..hope import HopeLayer
from ..titans import MACMixer


# ---------------------------------------------------------------------------
# global_step carrier: the value every HOPE block reads on its forward.
# We keep ONE module-level int rather than threading an arg through HF's
# fixed forward signature; `step.set_global_step` writes it per batch.
# ---------------------------------------------------------------------------
_GLOBAL_STEP = 0


def get_global_step() -> int:
    """Return the process-wide training step counter seen by HOPE blocks."""
    return _GLOBAL_STEP


def set_global_step_value(n: int) -> None:
    """Set the process-wide step counter (called by `step.set_global_step`)."""
    global _GLOBAL_STEP
    _GLOBAL_STEP = int(n)


def _hope_kwargs(layer: nn.Module) -> dict:
    """Pull the HOPE hyperparameters a `HopeLayer` needs off any module that
    carries them (a `HopeDecoderLayer`, or a `HopeLlmConfig`). Centralized so
    `surgery.py` and `hope_block.py` build identically-shaped layers."""
    return dict(
        d_model=layer.d_model,
        d_hidden=layer.hope_d_hidden,
        num_persistent_memory=layer.num_persistent_memory,
        cms_num_modules=layer.cms_num_modules,
        cms_base_update_freq=layer.cms_base_update_freq,
        cms_d_ff_multiplier=layer.cms_d_ff_multiplier,
        dropout=layer.dropout_p,
    )


def build_hope_layer(d_model: int, hope_d_hidden: int,
                     num_persistent_memory: int, cms_num_modules: int,
                     cms_base_update_freq: int, cms_d_ff_multiplier: int,
                     dropout: float = 0.0) -> HopeLayer:
    """Construct a bare `HopeLayer` with explicit args (used by `surgery.py`)."""
    return HopeLayer(
        d_model=d_model,
        d_hidden=hope_d_hidden,
        num_persistent_memory=num_persistent_memory,
        cms_num_modules=cms_num_modules,
        cms_base_update_freq=cms_base_update_freq,
        cms_d_ff_multiplier=cms_d_ff_multiplier,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# HopeDecoderLayer: a full HOPE layer that quacks like a LlamaDecoderLayer.
# ---------------------------------------------------------------------------
class HopeDecoderLayer(nn.Module):
    """A HOPE layer (Titans MAC mixer + Continuum Memory System) wrapped to
    match the HF decoder-layer interface, so it can be spliced straight into
    `model.model.layers` (the `nn.ModuleList` every Llama-family model exposes).

    Carries its hyperparameters as attributes (see `_hope_kwargs`) so
    `step.set_global_step` / introspection can find them uniformly. The CMS
    cadence reads the process-wide step via `get_global_step`.

    Args mirror the HOPE section of `HopeLlmConfig`.
    """

    def __init__(self, d_model: int, hope_d_hidden: int,
                 num_persistent_memory: int, cms_num_modules: int,
                 cms_base_update_freq: int, cms_d_ff_multiplier: int,
                 dropout: float = 0.0):
        super().__init__()
        # Carried as attributes for uniform introspection (see _hope_kwargs).
        self.d_model = d_model
        self.hope_d_hidden = hope_d_hidden
        self.num_persistent_memory = num_persistent_memory
        self.cms_num_modules = cms_num_modules
        self.cms_base_update_freq = cms_base_update_freq
        self.cms_d_ff_multiplier = cms_d_ff_multiplier
        self.dropout_p = dropout

        self.hope = build_hope_layer(
            d_model, hope_d_hidden, num_persistent_memory,
            cms_num_modules, cms_base_update_freq, cms_d_ff_multiplier, dropout,
        )
        # self.global_step is a per-instance mirror kept for discoverability;
        # the *authoritative* value the CMS reads is get_global_step().
        self.global_step = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        position_embeddings=None,
        **kwargs,
    ) -> torch.Tensor:
        """`hidden_states: [B, T, d] -> [B, T, d]` (returns a single tensor).

        Matches the HF decoder-layer contract: in transformers 4.x the layer
        returned a tuple `(hidden_states, attn_weights, present_kv)` and the
        model loop unpacked it; in transformers 5.x it returns a *single*
        tensor. We follow the 5.x contract (the installed version) -- returning
        a single tensor. We have no attention weights and no KV cache (see the
        module docstring).
        """
        # attention_mask: HF passes a causal/extended mask. The MAC mixer is a
        # token-by-token recurrence over the sequence and is itself order-aware,
        # so it does not consume an additive attention mask; we ignore it (the
        # causal structure the LLM relies on is preserved by the surrounding
        # decoder layers and by the MAC recurrence over prefix tokens).
        out = self.hope(hidden_states, get_global_step())
        self.global_step = get_global_step()
        return out


# ---------------------------------------------------------------------------
# MACAttnAdapter: the Titans MAC mixer as a drop-in self_attn sub-block.
# ---------------------------------------------------------------------------
class MACAttnAdapter(nn.Module):
    """Wraps `MACMixer` to look like a decoder layer's `self_attn`.

    A Llama-family decoder layer calls
        `self.self_attn(hidden_states=..., attention_mask=..., ...)`
    and expects `(attn_output, attn_weights, past_key_value)`. We delegate the
    actual compute to `MACMixer` (the memory-as-context mixer from `titans.py`)
    and return its single tensor padded with Nones.

    This lets the "swap_attention" mode keep the pretrained MLP, RMSNorms and
    residual structure of a decoder layer while replacing only the attention
    sub-block with a HOPE/Titans memory mixer.
    """

    def __init__(self, d_model: int, d_hidden: int,
                 num_persistent_memory: int = 16):
        super().__init__()
        self.d_model = d_model
        self.mixer = MACMixer(d_model, d_hidden, num_persistent_memory)
        self.global_step = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """`hidden_states: [B, T, d] -> ([B, T, d], None)`.

        Returns a 2-tuple `(attn_output, attn_weights)` to match the HF
        attention-block contract in transformers 5.x (4.x returned a 3-tuple
        with an extra `present_key_value` slot; we follow the installed 5.x
        arity). See `MACMixer.forward`; the additive attention_mask is ignored
        (the MAC recurrence is order-aware and not position-keyed; causal
        masking is provided by the surrounding decoder structure on the
        non-swapped layers).
        """
        out = self.mixer(hidden_states)
        self.global_step = get_global_step()
        return out, None
