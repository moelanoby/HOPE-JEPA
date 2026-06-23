"""Put HOPE layers + slot-JEPA (+ optional JEPA-Reasoner) into any HF Llama-family LLM.

Public API:
    HopeLlmConfig             -- dataclass mirroring config/llm_default.yaml.
    build_hope_llm(cfg, ...)  -- load + splice HOPE layers into an HF CausalLM.
    HopeLLM                   -- wrapper owning (HF model + SlotJEPAForLLM + JepaReasoner),
                                 with a single `forward` returning CE + aux loss.
    set_global_step(model, n) -- advance the CMS cadence (call per optimizer step).

Submodules:
    hope_block.HopeDecoderLayer / MACAttnAdapter  -- HOPE shaped as HF blocks.
    surgery.install_hope_layers / build_hope_llm  -- the 3 placement modes.
    jepa_llm.SlotJEPAForLLM                       -- slot-JEPA + SIGReg + div aux loss.
    reasoner.JepaReasoner                         -- latent rollout + talker.
    step.set_global_step                          -- cadence threading.
"""

from .config import HopeLlmConfig, LoadCfg, HopeCfg, JepaCfg, SigregCfg, ReasonerCfg, TrainCfg, parse_layer_spec
from .hope_block import HopeDecoderLayer, MACAttnAdapter, get_global_step
from .surgery import build_hope_llm, install_hope_layers
from .jepa_llm import SlotJEPAForLLM
from .reasoner import JepaReasoner
from .step import set_global_step
from .wrapper import HopeLLM

__all__ = [
    "HopeLlmConfig", "LoadCfg", "HopeCfg", "JepaCfg", "SigregCfg",
    "ReasonerCfg", "TrainCfg", "parse_layer_spec",
    "HopeDecoderLayer", "MACAttnAdapter", "get_global_step",
    "build_hope_llm", "install_hope_layers",
    "SlotJEPAForLLM", "JepaReasoner", "set_global_step",
    "HopeLLM",
]
