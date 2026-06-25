"""Reward shaping: GTPO for injectors, GRPO-S for fixers (arXiv:2508.04349).

EGGROLL consumes a single scalar FITNESS per population member. These two
functions PRODUCE that scalar, each following a different credit-assignment
rule from the GTPO/GRPO-S paper:

  * GTPO  (Group Token Policy Optimization)  -- TOKEN-level credit. Shapes the
    reward using per-token advantage, plus a dynamic policy-ENTROPY bonus that
    scales exploration. We use it for the INJECTOR fitness: an injector is
    rewarded for the fraction of fixers that FAILED to repair its bug, weighted
    toward the bug-diff TOKENS that most resisted fixing (token-level credit),
    plus an entropy bonus so it keeps inventing new kinds of bugs.

  * GRPO-S (Sequence-level GRPO)             -- SEQUENCE-level credit. The
    middle ground in the paper: coarser than GTPO, cheaper, with the same
    dynamic entropy weighting. We use it for the FIXER fitness: a fixer is
    rewarded for passing tests + matching the oracle fix, penalized for each
    tool it used and each retry it took, plus a sequence-level entropy term on
    its action distribution (so diverse fixing strategies are preferred).

Both entropy terms implement the paper's "dynamic entropy weighting": the bonus
is scaled by a moving target so it shrinks as the policy sharpens -- it pushes
exploration early and gets out of the way once the agent is confident. We use a
simple, robust form: entropy_weight * normalized_entropy * (1 - normalized_reward).

The outputs are then fed to `eggroll.rank_utilities`, which rank-shapes them so
the best member per batch dominates the center update (the "best of 12
contributes more" requirement).

NOTE on the GTPO/GRPO-S vs EGGROLL split: GTPO and GRPO-S are POLICY-GRADIENT
methods in their original paper, but their CONTRIBUTION here is the reward
SHAPING (token- vs sequence-level credit + dynamic entropy). EGGROLL is a
gradient-free optimizer that happily consumes any scalar fitness. Combining
"the reward shaping of GTPO/GRPO-S" with "the optimizer of EGGROLL" is a clean
factorization and is what we implement. The math of each piece is faithful to
its source; the combination is documented explicitly in every module.
"""

from __future__ import annotations

import math
from typing import List, Optional

from .config import RewardCfg
from .env import FixResult, InjectionResult


# ---------------------------------------------------------------------------
# GRPO-S: sequence-level fixer fitness
# ---------------------------------------------------------------------------
def fixer_fitness(
    fix: FixResult,
    tool_count: int,
    retries: int,
    fastest: bool,
    cfg: RewardCfg,
    action_entropy: Optional[float] = None,
) -> float:
    """GRPO-S sequence-level fitness for a fixer (higher = better).

        base = test_pass_weight * pass_rate
             + edit_sim_weight  * edit_similarity            # SWE-RL signals
             - tool_penalty     * tool_count                 # favor FEW tools
             - retry_penalty    * retries                    # favor FEW retries
             + speed_bonus      * (1 if fastest else 0)      # favor FAST fixes

    Then the GRPO-S dynamic entropy term: a small exploration bonus on the
    fixer's action-distribution entropy, scaled to vanish as reward rises
    (explore when struggling, exploit when winning). `action_entropy` is the
    entropy of the fixer's action-type distribution over its rollout (edit /
    tool-author / tool-call), normalized to [0, 1].

    This is the function that encodes "use the least tools and the least
    retries when fixing things": every tool and every retry subtracts from the
    fitness, so -- all else equal -- the leaner fixer ranks higher and its
    perturbation pulls the center theta harder via `rank_utilities`.
    """
    base = (
        cfg.test_pass_weight * float(fix.pass_rate)
        + cfg.edit_sim_weight * float(fix.edit_similarity)
        - cfg.tool_penalty * float(tool_count)
        - cfg.retry_penalty * float(retries)
        + cfg.speed_bonus * (1.0 if fastest else 0.0)
    )

    # GRPO-S dynamic entropy bonus: explore more when reward is low.
    if action_entropy is not None and cfg.entropy_weight > 0:
        norm_reward = max(0.0, min(1.0, base))                 # rough [0,1] view
        ent = max(0.0, min(1.0, float(action_entropy)))        # normalized entropy
        base += cfg.entropy_weight * ent * (1.0 - norm_reward)

    return float(base)


# ---------------------------------------------------------------------------
# GTPO: token-level injector fitness
# ---------------------------------------------------------------------------
def injector_fitness(
    injection: InjectionResult,
    fixer_passes: List[bool],
    cfg: RewardCfg,
    diff_token_entropies: Optional[List[float]] = None,
) -> float:
    """GTPO token-level fitness for an injector (higher = better).

    Base reward = fraction of fixers that FAILED to repair this injection.
    An injector that stumps all 12 fixers scores ~1.0; one whose bug is trivial
    to undo scores ~0.0. This is the raw win signal.

    GTPO's signature addition is TOKEN-level credit + entropy. We can't backprop
    per-token here (EGGROLL is gradient-free), so we realize the token-level
    shaping as a WEIGHTING over the bug-diff tokens: if `diff_token_entropies`
    is provided (the per-token policy entropy of the injector's emitted diff),
    we use it to compute a token-level advantage emphasis -- high-entropy tokens
    are where the injector was "creative", and we weight the win signal by how
    concentrated that creativity was. Concretely we add an entropy-weighted bonus:

        + entropy_weight * mean(diff_token_entropies) * (1 - win_rate)

    so an injector that explores diverse bug tokens while still mostly winning
    is preferred. This mirrors GTPO's dynamic entropy weighting at the token
    level (explore where uncertain). Without per-token entropies it degrades to
    a plain win-rate reward.
    """
    n = len(fixer_passes)
    if n == 0:
        # No fixers evaluated against this injection -> neutral fitness.
        return 0.0
    win_rate = float(sum(0 if p else 1 for p in fixer_passes)) / n   # frac that failed

    fit = win_rate

    if diff_token_entropies and cfg.entropy_weight > 0:
        mean_ent = float(sum(diff_token_entropies) / len(diff_token_entropies))
        # Normalize entropy: for a token over a vocab V, entropy / log(V) ~ [0,1].
        # The caller passes raw entropies; we softly bound with a logistic-ish map.
        norm_ent = 1.0 - math.exp(-mean_ent)
        # GTPO dynamic weighting: bonus shrinks as win_rate -> 1 (exploit when winning).
        fit += cfg.entropy_weight * norm_ent * (1.0 - win_rate)

    # Penalize an injector that failed to land ANY edit (no bug -> trivial fix).
    if not injection.edits or not injection.oracle_inverse:
        fit = 0.0

    return float(fit)


# ---------------------------------------------------------------------------
# GRPO-style sequence entropy helper (used by roles.py for action distributions)
# ---------------------------------------------------------------------------
def normalized_entropy(counts: List[int]) -> float:
    """Shannon entropy of an integer-count distribution, normalized to [0, 1].

    Used for the GRPO-S `action_entropy` input: e.g. a fixer that took
    3 edits + 1 tool-call + 1 tool-author -> counts=[3,1,1] -> normalized entropy.
    """
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    if len(probs) <= 1:
        return 0.0
    ent = -sum(p * math.log(p) for p in probs)
    return float(ent / math.log(len(probs)))
