"""Model surgery: splice HOPE layers into an HF Llama-family decoder stack.

`build_hope_llm(cfg)` loads `AutoModelForCausalLM.from_pretrained(cfg.model_id,
...)` (or accepts a pre-built model -- used by the smoke test, which constructs
from a tiny `LlamaConfig` with no weights download), then walks
`model.model.layers` (the `nn.ModuleList` every Llama-family model exposes) and
applies one of three *placement modes*:

  * "replace"        -- target layer i := HopeDecoderLayer(...) wholesale.
  * "insert"         -- splice a bare HopeLayer in *after* target layer i
                        (lengthens the stack; pretrained weights untouched).
  * "swap_attention" -- keep the layer's MLP/RMSNorms, replace only its
                        `.self_attn` with a MACAttnAdapter (Titans MAC mixer).

`install_hope_layers` is the pure surgery (no loading); `build_hope_llm` is the
loading + surgery entrypoint. The returned model still *is* an HF
`*ForCausalLM` -- it gains HOPE blocks but keeps its `lm_head`, tokenizer
interface, `from_pretrained` story and `generate()` (the latter recomputes
through HOPE blocks; see `hope_block` docstring).

Note on hidden_size: the HOPE hyperparameters in `cfg.hope` default to
auto-resolution (`d_hidden = 0` -> base model's hidden_size), so one config
works for both Qwen2.5-7B (hidden 3584) and LLaMA3-8B (hidden 4096) unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .config import HopeLlmConfig, parse_layer_spec
from .hope_block import HopeDecoderLayer, MACAttnAdapter, build_hope_layer


def _find_layer_list(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder `ModuleList` (`model.model.layers` in every
    Llama-family model). Raises with a helpful message if not found."""
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    raise AttributeError(
        "Could not find a decoder `nn.ModuleList` on the model (looked for "
        "`model.model.layers`, `model.transformer.h`, `model.gpt_neox.layers`). "
        "This integration targets Llama-family architectures (Llama/Qwen2/"
        "Mistral/Gemma)."
    )


def _resolve_hope_dims(cfg: HopeLlmConfig, hidden_size: int) -> dict:
    """Resolve HOPE hyperparameters, auto-filling d_hidden from hidden_size."""
    h = cfg.hope
    d_hidden = h.d_hidden if h.d_hidden > 0 else hidden_size
    return dict(
        d_model=hidden_size,
        hope_d_hidden=d_hidden,
        num_persistent_memory=h.num_persistent_memory,
        cms_num_modules=h.cms_num_modules,
        cms_base_update_freq=h.cms_base_update_freq,
        cms_d_ff_multiplier=h.cms_d_ff_multiplier,
        dropout=h.dropout,
    )


def install_hope_layers(model: nn.Module, cfg: HopeLlmConfig) -> nn.Module:
    """Apply the configured HOPE placement to `model` in place.

    Returns the same model (mutated). Stamps `model._hope_config` and
    `model._hope_target_layers` for downstream introspection.
    """
    layers = _find_layer_list(model)
    n = len(layers)

    # hidden_size: read from the base model config (every Llama-family model
    # carries `config.hidden_size`).
    hidden_size = model.config.hidden_size
    hp = _resolve_hope_dims(cfg, hidden_size)
    target = parse_layer_spec(cfg.target_layers, n)
    mode = cfg.placement

    if mode == "replace":
        for i in target:
            layers[i] = HopeDecoderLayer(**hp)
    elif mode == "insert":
        # Splice a HopeDecoderLayer (HF-contract shape) after each chosen index.
        # We must use the wrapped shape, not a bare HopeLayer, because HF calls
        # every entry in `model.model.layers` with the decoder-layer kwargs
        # (attention_mask=..., position_embeddings=...). Walk target in reverse
        # so earlier insertions don't shift the indices still to process; each
        # insertion lengthens the stack by 1.
        for i in sorted(target, reverse=True):
            new_block = HopeDecoderLayer(**hp)
            layers.insert(i + 1, new_block)
    elif mode == "swap_attention":
        for i in target:
            layer = layers[i]
            if not hasattr(layer, "self_attn"):
                raise AttributeError(
                    f"Layer {i} has no `.self_attn` to swap (placement="
                    f"'swap_attention'). Its type: {type(layer).__name__}."
                )
            layer.self_attn = MACAttnAdapter(
                d_model=hidden_size,
                d_hidden=hp["hope_d_hidden"],
                num_persistent_memory=hp["num_persistent_memory"],
            )
    else:
        raise ValueError(
            f"Unknown placement mode: {mode!r}. "
            "Expected 'replace', 'insert', or 'swap_attention'."
        )

    model._hope_config = cfg
    model._hope_target_layers = target
    model._hope_placement = mode
    return model


def build_hope_llm(
    cfg: HopeLlmConfig,
    model: Optional[nn.Module] = None,
):
    """Load an HF CausalLM and splice HOPE layers in.

    If `model` is given (e.g. a tiny model built from a LlamaConfig for the
    smoke test), use it directly -- no `from_pretrained` call. Otherwise load
    via `AutoModelForCausalLM.from_pretrained(cfg.model_id, ...)` honoring the
    quantization / device_map in `cfg.load`.

    NOTE: the actual `from_pretrained` path (4-bit QLoRA, device_map="auto")
    requires a GPU + bitsandbytes and is NOT exercised on this machine; it is
    only exercised by `scripts/train_llm_jepa.py` on a GPU box. The
    no-`model` branch here is thin and untrusted until run on that box.
    """
    if model is None:
        model = _from_pretrained(cfg)
    return install_hope_layers(model, cfg)


def _from_pretrained(cfg: HopeLlmConfig):
    """Load the base HF model with optional 4-bit/8-bit quantization.

    Kept separate so the smoke test never imports this path (it builds from a
    config object, avoiding any network/weights/GPU dependency).
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    load = cfg.load
    kwargs = {"device_map": load.device_map} if load.device_map else {}

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    if load.torch_dtype in dtype_map:
        kwargs["torch_dtype"] = dtype_map[load.torch_dtype]

    if load.quantize in ("4bit", "8bit"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(load.quantize == "4bit"),
            load_in_8bit=(load.quantize == "8bit"),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    return AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
