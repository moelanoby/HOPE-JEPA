"""EGGROLL: Evolution-Guided self-play RL for the HOPE-JEPA LLM.

A second training phase (after the slot-JEPA + CE pretraining in
`scripts/train_llm_jepa.py`). 3 "injector" agents introduce plausible bugs
into snapshots of one or more real git repos; 12 "fixer" agents (6 bare +
6 tool-augmented) race to repair them, validated against the repo's real
test suite (SWE-RL style). All 15 agents are population members of a single
EGGROLL evolution-strategies optimizer sharing one center model theta:

    theta_p = theta + sigma * eps_p          (eps_p a rank-r perturbation)
    f_p     = fitness(member_p)              (GTPO for injectors, GRPO-S for fixers)
    u_p     = rank_shape(f_p)               (top member dominates the update)
    theta  <- theta + (eta * sigma / r) * sum_p u_p * eps_p

References:
  * EGGROLL  -- arXiv:2511.16652 (rank-r black-box evolution strategies).
  * GTPO     -- arXiv:2508.04349 (token-level reward shaping + entropy),
                used to shape the INJECTOR fitness.
  * GRPO-S   -- arXiv:2508.04349 (sequence-level reward shaping + entropy),
                used to shape the FIXER fitness.
  * SWE-RL   -- facebookresearch/swe-rl (edit-similarity + real test-suite
                pass as the environment reward signal).

Public API:
    EggrollConfig                 -- dataclass mirroring config/eggroll_default.yaml.
    sample_perturbation /         -- the EGGROLL rank-r ES primitives.
      rank_utilities /
      eggroll_update
    Injector / Fixer              -- the two agent roles.
    RepoEnv                       -- the hybrid repo environment.
    ToolRegistry                  -- agent-authored tool cache + counter.
    injector_fitness /            -- the GTPO / GRPO-S fitness shapers.
      fixer_fitness
    EggrollTrainer                -- runs the generational self-play loop.
"""

from .config import (
    EggrollConfig, ReposCfg, RolesCfg, EggrollCfg, RewardCfg,
    ModelCfg, TrainCfg,
)
from .eggroll import (
    sample_perturbation, apply_perturbation, rank_utilities, eggroll_update,
)
from .tools import ToolRegistry, ToolDef, parse_tools, compile_tool
from .env import RepoEnv, Snapshot, InjectionResult, FixResult
from .reward import injector_fitness, fixer_fitness
from .roles import Injector, Fixer
from .population import EggrollTrainer

__all__ = [
    # config
    "EggrollConfig", "ReposCfg", "RolesCfg", "EggrollCfg", "RewardCfg",
    "ModelCfg", "TrainCfg",
    # EGGROLL ES core
    "sample_perturbation", "apply_perturbation", "rank_utilities",
    "eggroll_update",
    # tools
    "ToolRegistry", "ToolDef", "parse_tools", "compile_tool",
    # env
    "RepoEnv", "Snapshot", "InjectionResult", "FixResult",
    # reward
    "injector_fitness", "fixer_fitness",
    # roles
    "Injector", "Fixer",
    # orchestrator
    "EggrollTrainer",
]
