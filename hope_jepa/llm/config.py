"""Config objects for putting HOPE + slot-JEPA (+ JEPA-Reasoner) into an HF LLM.

`HopeLlmConfig` is a plain dataclass that mirrors `config/llm_default.yaml`. It
is consumed by `surgery.build_hope_llm` (placement), `jepa_llm.SlotJEPAForLLM`
(the slot-JEPA aux objective), and `reasoner.JepaReasoner` (latent rollout).

`parse_layer_spec` resolves the shorthand layer selectors ("last:3", "all", or
an explicit list) used by both `placement.target_layers` and `jepa.layers`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class LoadCfg:
    model_id: str = "Qwen/Qwen2.5-7B"
    # "4bit" | "8bit" | "none". 4bit/8bit need bitsandbytes + CUDA; the CPU
    # smoke test always uses "none" (it builds from a tiny config, not weights).
    quantize: str = "4bit"
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"


@dataclass
class HopeCfg:
    d_hidden: int = 0            # 0 => auto = hidden_size of the base model
    num_persistent_memory: int = 16
    cms_num_modules: int = 4
    cms_base_update_freq: int = 1
    cms_d_ff_multiplier: int = 4
    dropout: float = 0.0


@dataclass
class JepaCfg:
    enabled: bool = True
    layers: Union[str, List[int]] = "last:3"   # resolved against num layers
    num_slots: int = 8                          # K diverging slots
    predictor_depth: int = 2
    num_heads: int = 8
    mask_ratio: float = 0.4                     # fraction of BPE positions masked
    weight: float = 1.0                         # lambda_jepa


@dataclass
class SigregCfg:
    enabled: bool = True
    sketch_dim: int = 128
    target_scale: float = 1.0
    weight: float = 1.0                         # lambda_sig


@dataclass
class ReasonerCfg:
    enabled: bool = False
    steps: int = 4                              # R latent rollout steps
    talker_dim_mult: float = 1.0                # talker hidden multiplier


@dataclass
class TrainCfg:
    qlora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lr: float = 2.0e-4


@dataclass
class HopeLlmConfig:
    model_id: str = "Qwen/Qwen2.5-7B"
    load: LoadCfg = field(default_factory=LoadCfg)
    placement: str = "swap_attention"           # "replace"|"insert"|"swap_attention"
    target_layers: Union[str, List[int]] = "last:3"  # which layers get HOPE
    hope: HopeCfg = field(default_factory=HopeCfg)
    jepa: JepaCfg = field(default_factory=JepaCfg)
    sigreg: SigregCfg = field(default_factory=SigregCfg)
    slot_div_weight: float = 0.1                # lambda_div
    reasoner: ReasonerCfg = field(default_factory=ReasonerCfg)
    training: TrainCfg = field(default_factory=TrainCfg)

    @classmethod
    def from_dict(cls, d: dict) -> "HopeLlmConfig":
        """Build from a parsed yaml dict (nested keys -> nested dataclasses)."""
        def sub(dc, prefix):
            kw = {}
            for f in dc.__dataclass_fields__:
                if f in d.get(prefix, {}):
                    kw[f] = d[prefix][f]
            return dc(**kw)

        top = {k: v for k, v in d.items()
               if k in cls.__dataclass_fields__ and not isinstance(v, dict)}
        return cls(
            **top,
            load=sub(LoadCfg, "load") if "load" in d else LoadCfg(model_id=d.get("model_id", "Qwen/Qwen2.5-7B")),
            hope=sub(HopeCfg, "hope"),
            jepa=sub(JepaCfg, "jepa"),
            sigreg=sub(SigregCfg, "sigreg"),
            reasoner=sub(ReasonerCfg, "reasoner"),
            training=sub(TrainCfg, "training"),
        )


def parse_layer_spec(spec: Union[str, List[int]], num_layers: int) -> List[int]:
    """Resolve a layer selector to an explicit sorted list of indices.

    Accepted forms:
      * "all"                -> every layer index
      * "last:N"             -> the last N layers
      * "first:N"            -> the first N layers
      * "even" / "odd"       -> every other layer
      * [3, 7, 11]           -> used verbatim (clamped to range)
    """
    if isinstance(spec, list):
        return sorted({i for i in spec if 0 <= i < num_layers})
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s == "all":
            return list(range(num_layers))
        if s.startswith("last:"):
            n = int(s.split(":")[1])
            return list(range(max(0, num_layers - n), num_layers))
        if s.startswith("first:"):
            n = int(s.split(":")[1])
            return list(range(min(n, num_layers)))
        if s == "even":
            return list(range(0, num_layers, 2))
        if s == "odd":
            return list(range(1, num_layers, 2))
    raise ValueError(f"Unrecognized layer spec: {spec!r}")
