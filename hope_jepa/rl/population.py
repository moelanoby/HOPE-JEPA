"""The EGGROLL self-play orchestrator.

One generation of training, end to end:

    1. SAMPLE P perturbations of the shared center theta
       (P = num_injectors + num_fixers = 15 by default). Each is a rank-r
       EGGROLL perturbation; with antithetic sampling we get +/- pairs from one
       noise draw.

    2. EVALUATE every member as a role over the repo environment:
       - injectors produce bugs; each bug is injected into a clean snapshot.
       - all fixers race on every injection, validated by the REAL test suite.
       We do this SEQUENTIALLY (one member at a time) so we never hold P model
       copies in memory at once -- only the perturbation tensors are materialized.

    3. SHAPE fitnesses:
       - injector members -> GTPO fitness (token-level credit + entropy).
       - fixer members    -> GRPO-S fitness (sequence-level, tool/retry penalties).

    4. UPDATE the center theta via EGGROLL:
       u = rank_utilities(all_fitnesses)        # best member dominates
       theta <- theta + (eta*sigma/r) * sum_p u_p * eps_p

The "best of 12 contributes more" requirement is realized by `rank_utilities`:
the top-ranked fixer's perturbation gets the largest |utility| and therefore
moves theta the most. The "fewest tools / retries" requirement is baked into
`fixer_fitness` (every tool/retry subtracts), so a lean fixer ranks higher.

Memory note: members are inference-only. We snapshot theta, materialize one
perturbation, load it into the SHARED model (no copy), run the rollout, then
discard. Only theta + the perturbation list are kept for the update.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .config import EggrollConfig
from .eggroll import (
    apply_perturbation, eggroll_update, rank_utilities, sample_perturbation,
)
from .env import RepoEnv, Snapshot
from .reward import fixer_fitness, injector_fitness
from .roles import Fixer, Injector
from .tools import ToolRegistry


@dataclass
class GenerationStats:
    """Per-generation diagnostics."""
    generation: int
    mean_fitness: float
    best_fitness: float
    best_role: str             # "injector" or "fixer"
    mean_fixer_tools: float
    mean_fixer_turns: float
    mean_pass_rate: float
    mean_abs_step: float
    injectors_won: int         # injections no fixer repaired


@dataclass
class _Member:
    """A population member: a perturbation + its role assignment."""
    idx: int
    role: str                  # "injector" or "fixer"
    has_tools: bool            # only meaningful for fixers
    eps: List[torch.Tensor]    # the perturbation aligned with trainable names
    eps_neg: Optional[List[torch.Tensor]] = None   # antithetic partner (if any)
    fitness: float = 0.0


class EggrollTrainer:
    """Runs EGGROLL self-play over a shared center HopeLLM.

    Args:
        cfg:        the EggrollConfig.
        center_model: the HopeLLM that is theta. Its trainable params are
                    perturbed/updated; the frozen quantized base is untouched.
        tokenizer:  the center model's tokenizer (for agent generation).
        device:     "cuda" | "cpu".
        role_factory: optional injection seam for the smoke test. Called as
                    `role_factory(role, has_tools, device) -> (Injector|Fixer)`.
                    If None, real Injector/Fixer are built around center_model.
    """

    def __init__(
        self,
        cfg: EggrollConfig,
        center_model,
        tokenizer,
        device: str,
        role_factory: Optional[Callable] = None,
    ):
        self.cfg = cfg
        self.center_model = center_model
        self.tokenizer = tokenizer
        self.device = device
        self.role_factory = role_factory

        self.env = RepoEnv(cfg.repos)
        self.generator = torch.Generator(device=device).manual_seed(cfg.train.seed)

        # Cache the trainable param names + a snapshot of theta for fast restore.
        self.trainable = [(n, p) for n, p in center_model.named_parameters()
                          if p.requires_grad]
        self.names = [n for n, _ in self.trainable]

    # ------------------------------------------------------------------
    def _sample_population(self) -> List[_Member]:
        """Build the P-member population: role assignments + perturbations."""
        rc, ec = self.cfg.roles, self.cfg.eggroll
        n_inj, n_fix = rc.num_injectors, rc.num_fixers
        n_tool = min(rc.tool_fixers, n_fix)
        members: List[_Member] = []

        # Injectors first, then fixers. Tool-fixers are the first n_tool fixers.
        for i in range(n_inj):
            eps = sample_perturbation(self.trainable, ec.rank, ec.sigma,
                                      antithetic=False, device=self.device,
                                      generator=self.generator)
            members.append(_Member(idx=i, role="injector", has_tools=False, eps=eps))

        for j in range(n_fix):
            has_tools = j < n_tool
            eps = sample_perturbation(self.trainable, ec.rank, ec.sigma,
                                      antithetic=False, device=self.device,
                                      generator=self.generator)
            members.append(_Member(idx=n_inj + j, role="fixer",
                                   has_tools=has_tools, eps=eps))
        return members

    # ------------------------------------------------------------------
    def _load_member(self, eps: List[torch.Tensor]) -> None:
        """theta + eps -> shared model (in place), inference-only.

        Under no_grad: we're mutating leaf params that require grad (the LoRA
        adapters / HOPE modules), which is only legal outside the autograd graph.
        """
        center_state = {n: p.detach() for n, p in self.trainable}
        perturbed = apply_perturbation(center_state, eps, self.names)
        with torch.no_grad():
            for n, p in self.trainable:
                p.copy_(perturbed[n])

    def _restore_center(self) -> None:
        """Undo a member load: restore the pristine center theta."""
        with torch.no_grad():
            for n, p in self.trainable:
                p.copy_(self._theta_snapshot[n])

    # ------------------------------------------------------------------
    def _make_role(self, member: _Member):
        """Build an Injector or Fixer for a member (real LLM or smoke stub)."""
        if self.role_factory is not None:
            return self.role_factory(member.role, member.has_tools, self.device)
        if member.role == "injector":
            return Injector(self.center_model, self.tokenizer, self.device)
        return Fixer(self.center_model, self.tokenizer, self.device,
                     has_tools=member.has_tools)

    # ------------------------------------------------------------------
    def _evaluate(self, members: List[_Member]) -> Tuple[List[_Member], GenerationStats]:
        """Run the self-play round and assign each member a fitness.

        Shape of a round:
          for each repo in rotation:
            clean = snapshot(repo)
            for each injector member:
                inject its bug -> injected snapshot
                for each fixer member:
                    repair + validate against the real tests -> FixResult
                injector fitness = GTPO(fix results)   [over all fixers]
            fixer fitness = GRPO-S averaged across the injections it faced

        We keep per-injection fixer outcomes to compute injector fitness (GTPO:
        how many fixers failed), and per-fixer aggregates for the fixer fitness.
        """
        rc = self.cfg.roles
        n_repos = max(1, len(self.env._sources)) if self.env._sources else 1
        injectors = [m for m in members if m.role == "injector"]
        fixers = [m for m in members if m.role == "fixer"]

        # Per-fixer accumulators (averaged across all injections it sees).
        fix_acc = {m.idx: {"pass": 0, "sim": 0.0, "tools": 0, "retries": 0,
                           "elapsed": 0.0, "n": 0, "ent": 0.0, "fastest": 0}
                   for m in fixers}

        injections_total = 0
        injectors_won = 0

        for repo_idx in range(n_repos):
            clean = self.env.snapshot(repo_idx)
            for inj_m in injectors:
                # Load the injector member, run it, restore center.
                self._load_member(inj_m.eps)
                inj_role = self._make_role(inj_m)
                # Each injector gets its own clean snapshot to avoid cross-contam.
                inj_snap = self.env.snapshot(repo_idx)
                action = inj_role.act(inj_snap)
                injection = self.env.inject(inj_snap, action.edits)
                self._restore_center()

                injections_total += 1

                # All fixers race on THIS injection.
                fix_times = []
                per_fixer: List[Tuple[_Member, "object"]] = []   # (member, FixResult)
                for fix_m in fixers:
                    self._load_member(fix_m.eps)
                    registry = ToolRegistry() if fix_m.has_tools else None
                    fix_role = self._make_role(fix_m)
                    fr_action, _ = fix_role.act(injection.injected,
                                                registry or ToolRegistry(),
                                                self.cfg.train.max_turns, self.env)
                    # Validate the fixer's edits against the real test suite.
                    from .env import FixResult
                    fr = self.env.validate_fix(injection.injected, fr_action.edits)
                    self._restore_center()
                    per_fixer.append((fix_m, fr, fr_action, registry))
                    fix_times.append(fr_action.elapsed)

                fastest_t = min(fix_times) if fix_times else 0.0

                # Injector fitness = GTPO over the fixer outcomes.
                passes = [fr.passed for _, fr, _, _ in per_fixer]
                inj_m.fitness = injector_fitness(
                    injection, passes, self.cfg.reward,
                    diff_token_entropies=action.token_entropies or None,
                )
                if not any(passes):
                    injectors_won += 1   # nobody fixed it -> injector win

                # Accumulate fixer outcomes for their GRPO-S fitness.
                for fix_m, fr, fa, reg in per_fixer:
                    a = fix_acc[fix_m.idx]
                    a["pass"] += int(fr.passed)
                    a["sim"] += fr.edit_similarity
                    a["tools"] += (reg.tools_used if reg else 0)
                    a["retries"] += fa.turns
                    a["elapsed"] += fa.elapsed
                    a["n"] += 1
                    a["ent"] += fa.action_entropy
                    if abs(fa.elapsed - fastest_t) < 1e-9:
                        a["fastest"] += 1

        # Compute each fixer's GRPO-S fitness (averaged across injections).
        for fix_m in fixers:
            a = fix_acc[fix_m.idx]
            n = max(1, a["n"])
            pass_rate = a["pass"] / n
            mean_sim = a["sim"] / n
            mean_tools = a["tools"] / n
            mean_turns = a["retries"] / n
            mean_ent = a["ent"] / n
            fastest_frac = a["fastest"] / n
            from .env import FixResult
            avg_fix = FixResult(passed=pass_rate >= 1.0, pass_rate=pass_rate,
                                n_tests=0, edit_similarity=mean_sim,
                                elapsed=a["elapsed"] / n)
            fix_m.fitness = fixer_fitness(
                avg_fix, tool_count=mean_tools, retries=mean_turns,
                fastest=fastest_frac >= 1.0, cfg=self.cfg.reward,
                action_entropy=mean_ent,
            )

        # Stats
        all_f = torch.tensor([m.fitness for m in members], dtype=torch.float32)
        best_idx = int(torch.argmax(all_f).item())
        best = members[best_idx]
        mean_tools = (sum(fix_acc[m.idx]["tools"]
                          for m in fixers) / max(1, len(fixers)))
        mean_turns = (sum(fix_acc[m.idx]["retries"]
                          for m in fixers) / max(1, len(fixers)))
        mean_pass = (sum((fix_acc[m.idx]["pass"] / max(1, fix_acc[m.idx]["n"]))
                         for m in fixers) / max(1, len(fixers)))
        stats = GenerationStats(
            generation=-1, mean_fitness=float(all_f.mean().item()),
            best_fitness=float(all_f.max().item()), best_role=best.role,
            mean_fixer_tools=float(mean_tools),
            mean_fixer_turns=float(mean_turns),
            mean_pass_rate=float(mean_pass),
            mean_abs_step=0.0, injectors_won=injectors_won,
        )
        return members, stats

    # ------------------------------------------------------------------
    def train(self, output_dir: str = "runs/eggroll") -> List[GenerationStats]:
        """Run `cfg.train.generations` EGGROLL generations."""
        os.makedirs(output_dir, exist_ok=True)
        ec = self.cfg.eggroll
        history: List[GenerationStats] = []

        for gen in range(self.cfg.train.generations):
            # Snapshot pristine theta to restore after each member eval.
            self._theta_snapshot = {n: p.detach().clone()
                                    for n, p in self.trainable}
            members = self._sample_population()
            members, stats = self._evaluate(members)
            stats.generation = gen

            # EGGROLL center update from the population's shaped utilities.
            fitnesses = torch.tensor([m.fitness for m in members],
                                     dtype=torch.float32, device=self.device)
            utilities = rank_utilities(fitnesses, shape=ec.utility_shape)
            diag = eggroll_update(self.center_model, [m.eps for m in members],
                                  utilities, self.names, ec.lr, ec.sigma, ec.rank)
            stats.mean_abs_step = diag["mean_abs_step"]
            history.append(stats)

            if gen % max(1, self.cfg.train.log_every) == 0:
                print(
                    f"[gen {gen:3d}] mean_fit={stats.mean_fitness:.4f} "
                    f"best={stats.best_fitness:.4f}({stats.best_role}) "
                    f"inj_won={stats.injectors_won} "
                    f"pass={stats.mean_pass_rate:.2f} "
                    f"tools={stats.mean_fixer_tools:.1f} "
                    f"turns={stats.mean_fixer_turns:.1f} "
                    f"step={stats.mean_abs_step:.2e}", flush=True)

            if (gen + 1) % max(1, self.cfg.train.ckpt_every) == 0:
                ckpt = os.path.join(output_dir, f"eggroll_gen{gen}.pt")
                torch.save({"center": {n: p.detach().cpu() for n, p in self.trainable},
                            "gen": gen, "cfg": self.cfg}, ckpt)

        final = os.path.join(output_dir, "eggroll_final.pt")
        torch.save({"center": {n: p.detach().cpu() for n, p in self.trainable},
                    "cfg": self.cfg}, final)
        print(f"Saved final center theta to {final}")
        return history
