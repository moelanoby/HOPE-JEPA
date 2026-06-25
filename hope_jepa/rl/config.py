"""Config objects for the EGGROLL self-play RL phase.

`EggrollConfig` is a plain dataclass that mirrors `config/eggroll_default.yaml`.
It follows the SAME nested-dataclass + `from_dict` pattern as
`hope_jepa.llm.config.HopeLlmConfig` so the two configs read consistently.

Consumed by:
  * `eggroll.py`       -- EggrollCfg (the ES knobs: rank, sigma, lr, utility shape).
  * `env.py`           -- ReposCfg (which repos, how to test them).
  * `reward.py`        -- RewardCfg (GTPO/GRPO-S weights, tool/retry penalties).
  * `roles.py`         -- RolesCfg (how many injectors / fixers / tool-fixers).
  * `population.py`    -- TrainCfg (generations, turns, seed) + everything above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ReposCfg:
    """The repo environment: one or many repos the agents work against.

    `repos` accepts either local paths (used in place via a copytree snapshot)
    or git URLs (cloned once on init). A generation rotates across all listed
    repos so a single config can train against multiple codebases.
    """
    repos: List[str] = field(default_factory=list)   # 1..N local paths or git URLs
    max_files_per_snapshot: int = 20
    test_command: str = "python -m pytest -q"         # run inside the sandbox cwd
    workdir: str = "runs/eggroll/sandboxes"           # where snapshots live
    test_timeout: int = 60                            # seconds per test run


@dataclass
class RolesCfg:
    """Population role split. Total members = num_injectors + num_fixers."""
    num_injectors: int = 3
    num_fixers: int = 12
    tool_fixers: int = 6    # of the `num_fixers` fixers, how many can author tools
    bare_fixers: int = 6    # the rest get edits only (no tool authoring/calling)


@dataclass
class EggrollCfg:
    """EGGROLL evolution-strategies knobs (arXiv:2511.16652).

    The optimizer perturbs only the TRAINABLE params of the center model
    (LoRA adapters + HOPE/JEPA modules), so a 7B model's perturbation lives in
    the small LoRA space -- exactly the param set `train_llm_jepa.apply_qlora`
    leaves trainable. `rank` is the EGGROLL low-rank r: eps = A @ B^T with
    A, B in R^{d x r}, the lever that makes the update GPU-efficient.
    """
    rank: int = 8
    sigma: float = 0.02            # perturbation scale
    lr: float = 0.01               # eta
    antithetic: bool = True        # use +/- eps to halve variance
    utility_shape: str = "log"     # "log" | "linear" rank shaping; top member dominates


@dataclass
class RewardCfg:
    """Fitness shaping weights.

    `fixer_fitness` (GRPO-S, sequence-level):
        test_pass_weight*pass + edit_sim_weight*sim
        - tool_penalty*tools - retry_penalty*retries + speed_bonus*fastest
    `injector_fitness` (GTPO, token-level):
        fraction of fixers that FAILED, weighted by per-token diff advantage,
        + entropy_weight * token-entropy bonus (exploration).
    """
    tool_penalty: float = 0.1
    retry_penalty: float = 0.05
    speed_bonus: float = 0.2
    edit_sim_weight: float = 0.4
    test_pass_weight: float = 0.6
    entropy_weight: float = 0.01


@dataclass
class ModelCfg:
    """How to build the center model theta.

    `hope_config` points at an llm_default-style YAML whose HopeLLM is the
    center theta. `quantize` mirrors the LoadCfg options ("4bit"/"8bit"/"none").
    On CPU (smoke test) we bypass this and build a tiny model directly.
    """
    hope_config: str = "config/llm_default.yaml"
    quantize: str = "none"        # smoke/CPU uses none; GPU run sets "4bit"


@dataclass
class TrainCfg:
    generations: int = 50         # EGGROLL generations to run
    max_turns: int = 8            # action budget per agent per injection
    seed: int = 0
    log_every: int = 1            # print every N generations
    ckpt_every: int = 10          # checkpoint center theta every N generations


@dataclass
class EggrollConfig:
    """Top-level config mirroring config/eggroll_default.yaml."""
    repos: ReposCfg = field(default_factory=ReposCfg)
    roles: RolesCfg = field(default_factory=RolesCfg)
    eggroll: EggrollCfg = field(default_factory=EggrollCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    @classmethod
    def from_dict(cls, d: dict) -> "EggrollConfig":
        """Build from a parsed yaml dict (nested keys -> nested dataclasses).

        Same shape as HopeLlmConfig.from_dict: each top-level dict key maps to
        a sub-dataclass by matching field names.
        """
        def sub(dc, key):
            kw = {}
            for f in dc.__dataclass_fields__:
                if f in d.get(key, {}):
                    kw[f] = d[key][f]
            return dc(**kw)

        return cls(
            repos=sub(ReposCfg, "repos"),
            roles=sub(RolesCfg, "roles"),
            eggroll=sub(EggrollCfg, "eggroll"),
            reward=sub(RewardCfg, "reward"),
            model=sub(ModelCfg, "model"),
            train=sub(TrainCfg, "train"),
        )
