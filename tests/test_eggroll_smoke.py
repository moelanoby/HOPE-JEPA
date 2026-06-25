"""CPU smoke test for the EGGROLL self-play RL phase.

Builds everything with NO model download and NO network -- a tiny model, a
synthetic in-memory repo with a real pytest test, and deterministic stub
injector/fixer agents. Verifies the whole pipeline end to end:

  1. EGGROLL math: rank_utilities assigns the largest |u| to the top fitness
     (best member dominates the update), and eggroll_update moves theta toward
     higher-fitness perturbations.
  2. RepoEnv: inject a real bug -> tests fail; apply the oracle inverse fix
     -> tests pass; edit_similarity scores it.
  3. ToolRegistry: compiles a valid tool, rejects a broken one, counts usage.
  4. Reward shaping: fixer_fitness rewards pass+sim and penalizes tools/retries;
     injector_fitness rewards fixer failures.
  5. A full 1-generation self-play round (3 injectors + 12 fixers) runs
     end-to-end on the synthetic repo and produces finite fitnesses + a center
     theta update with nonzero step.

Run with:
    python -m tests.test_eggroll_smoke
"""

from __future__ import annotations

import os
import shutil
import tempfile

import torch
import torch.nn as nn

from hope_jepa.rl import (
    EggrollConfig, eggroll_update, rank_utilities,
    RepoEnv, ToolRegistry, ToolDef, compile_tool, parse_tools,
    fixer_fitness, injector_fitness,
)
from hope_jepa.rl.config import ReposCfg, RewardCfg
from hope_jepa.rl.env import FileEdit, InjectionResult
from hope_jepa.rl.population import EggrollTrainer
from hope_jepa.rl.roles import make_stub_injector, make_stub_fixer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_synthetic_repo() -> str:
    """A tiny real repo: a module + a test that passes when correct.

    The injector will break `add` (e.g. change `+` to `-`); the fixer restores it.
    The test file is BOTH pytest-compatible AND runnable standalone
    (`python test_calc.py`), so the smoke test exercises the real subprocess +
    exit-code path WITHOUT requiring pytest to be installed. Real repos in the
    actual trainer use `python -m pytest -q` (the config default).
    """
    d = tempfile.mkdtemp(prefix="eggroll_repo_")
    with open(os.path.join(d, "calc.py"), "w") as f:
        f.write("def add(a, b):\n    return a + b\n")
    with open(os.path.join(d, "test_calc.py"), "w") as f:
        f.write(
            "from calc import add\n\n"
            "def test_add():\n    assert add(2, 3) == 5\n\n"
            "if __name__ == '__main__':\n"
            "    test_add()\n"
            "    print('1 passed')\n"
        )
    return d


def _tiny_linear_model() -> nn.Module:
    """A tiny trainable module with 2 params -- enough for EGGROLL math tests."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 8))
            self.b = nn.Parameter(torch.randn(8))

        def forward(self, x):           # noqa: ARG002
            return None
    m = M()
    for p in m.parameters():
        p.requires_grad_(True)
    return m


# ---------------------------------------------------------------------------
# 1. EGGROLL math
# ---------------------------------------------------------------------------
def test_rank_utilities():
    """The best fitness gets the largest |utility| (best contributes most)."""
    f = torch.tensor([0.1, 0.9, 0.5, 0.2, 0.8])
    u = rank_utilities(f, shape="log")
    assert u.shape == f.shape
    # zero-mean
    assert abs(u.mean().item()) < 1e-5, "utilities not zero-mean"
    # normalized: ||u||_2 == sqrt(P)
    assert abs(u.norm().item() - (f.numel() ** 0.5)) < 1e-4, \
        f"||u||={u.norm().item()} != sqrt(P)={f.numel()**0.5}"
    # the BEST member (index 1, fitness 0.9) gets the largest utility
    assert u.argmax().item() == 1, f"best member not highest utility: {u}"
    # monotonic in fitness among the positive members
    order = torch.argsort(f, descending=True)
    pos_utils = u[order[u[order] > 0]]
    assert torch.all(pos_utils[:-1] >= pos_utils[1:] - 1e-6), \
        "utilities not monotonic in fitness"
    print(f"[ok] rank_utilities: best member dominates. utilities={u.tolist()}")


def test_eggroll_update_moves_toward_winner():
    """eggroll_update shifts theta in the direction of the winning perturbation."""
    model = _tiny_linear_model()
    names = ["w", "b"]
    w0 = model.w.detach().clone()
    b0 = model.b.detach().clone()

    # Two perturbations: member 0 (low fitness) and member 1 (high fitness).
    eps0 = [torch.randn_like(model.w), torch.randn_like(model.b)]
    eps1 = [torch.randn_like(model.w), torch.randn_like(model.b)]
    fitnesses = torch.tensor([0.1, 0.9])
    utilities = rank_utilities(fitnesses, shape="log")

    eggroll_update(model, [eps0, eps1], utilities, names,
                   lr=0.1, sigma=1.0, rank=1)

    dw = (model.w - w0)
    db = (model.b - b0)
    assert dw.abs().sum().item() > 0, "theta didn't move"
    # The move is dominated by the winning (high-utility) perturbation eps1.
    # Since u1 > 0 and u0 < 0 (zero-mean shaping), theta should align with eps1
    # and ANTI-align with eps0 on the dims where they differ.
    move_w = dw.flatten()
    e1_w = eps1[0].flatten()
    cos = torch.dot(move_w, e1_w) / (move_w.norm() * e1_w.norm() + 1e-9)
    assert cos.item() > 0.5, f"theta didn't move toward winner (cos={cos:.3f})"
    print(f"[ok] eggroll_update moves theta toward winner (cos={cos:.3f}, "
          f"dw_abs_sum={dw.abs().sum().item():.4f})")


def test_update_preserves_param_count():
    """eggroll_update never changes param shapes or count."""
    model = _tiny_linear_model()
    before = sum(p.numel() for p in model.parameters())
    eps = [[torch.randn_like(model.w), torch.randn_like(model.b)]]
    u = rank_utilities(torch.tensor([1.0]), shape="log")  # single -> zero util
    eggroll_update(model, eps, u, ["w", "b"], lr=0.1, sigma=1.0, rank=1)
    after = sum(p.numel() for p in model.parameters())
    assert before == after
    print(f"[ok] eggroll_update preserves param count ({before} elements)")


# ---------------------------------------------------------------------------
# 2. RepoEnv: inject / validate / oracle
# ---------------------------------------------------------------------------
def test_repo_env_inject_and_fix():
    """A real bug -> tests fail; oracle inverse -> tests pass; edit-sim scores."""
    repo = _make_synthetic_repo()
    try:
        cfg = ReposCfg(repos=[repo], test_command="python test_calc.py",
                       workdir=tempfile.mkdtemp(prefix="eggroll_env_"),
                       test_timeout=30)
        env = RepoEnv(cfg)
        snap = env.snapshot(0)

        # Injector breaks `add`: `return a + b` -> `return a - b`
        bug = FileEdit(path="calc.py", old="return a + b", new="return a - b")
        injection = env.inject(snap, [bug])
        assert injection.oracle_inverse, "no oracle inverse recorded"
        assert injection.oracle_inverse[0].new == "return a + b"

        # Validate with NO fix -> should fail (the bug is present).
        fr_bad = env.validate_fix(injection.injected, [])
        assert fr_bad.pass_rate < 1.0, "bugged repo passed tests unexpectedly"

        # Validate with the oracle fix -> should pass.
        fix = injection.oracle_inverse[0]
        fr_good = env.validate_fix(injection.injected, [fix])
        assert fr_good.pass_rate >= 1.0, \
            f"oracle fix didn't pass tests: {fr_good.error}"
        assert fr_good.edit_similarity > 0.9, \
            f"edit similarity low: {fr_good.edit_similarity}"
        print(f"[ok] RepoEnv inject->fail, fix->pass "
              f"(pass_rate {fr_bad.pass_rate:.2f}->{fr_good.pass_rate:.2f}, "
              f"sim={fr_good.edit_similarity:.2f})")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. ToolRegistry
# ---------------------------------------------------------------------------
def test_tool_registry():
    reg = ToolRegistry()
    good = ToolDef(name="double", source="def double(x):\n    return x * 2\n")
    bad = ToolDef(name="broken", source="def broken(  :")     # SyntaxError
    assert reg.register(good) is True, "valid tool rejected"
    assert reg.register(bad) is False, "broken tool accepted"
    assert reg.tools_authored == 1
    assert reg.tools_used == 2                 # both authorings charged budget
    out = reg.invoke("double", x=21)
    assert out == 42
    assert reg.tools_used == 3                 # invocation charged too
    # parse_tools picks up blocks from agent prose
    parsed = parse_tools('noise <tool name="q">def q():\n    return 1</tool> more')
    assert len(parsed) == 1 and parsed[0].name == "q"
    assert compile_tool(parsed[0]).compiled
    print(f"[ok] ToolRegistry: compiles valid, rejects broken, counts usage "
          f"(tools_used={reg.tools_used})")


# ---------------------------------------------------------------------------
# 4. Reward shaping
# ---------------------------------------------------------------------------
def test_fixer_fitness_penalizes_tools_and_retries():
    """A fixer with fewer tools/retries scores higher, all else equal."""
    from hope_jepa.rl.env import FixResult
    cfg = RewardCfg(tool_penalty=0.1, retry_penalty=0.05, speed_bonus=0.2,
                    edit_sim_weight=0.4, test_pass_weight=0.6, entropy_weight=0.0)
    fr = FixResult(passed=True, pass_rate=1.0, n_tests=1, edit_similarity=1.0,
                   elapsed=0.1)
    lean = fixer_fitness(fr, tool_count=0, retries=1, fastest=True, cfg=cfg)
    heavy = fixer_fitness(fr, tool_count=5, retries=8, fastest=False, cfg=cfg)
    assert lean > heavy, f"lean ({lean}) should beat heavy ({heavy})"
    print(f"[ok] fixer_fitness: lean={lean:.3f} > heavy={heavy:.3f} "
          f"(penalizes tools/retries)")


def test_injector_fitness_rewards_fixer_failure():
    """An injector whose bug stumps all fixers scores higher than one fixed instantly."""
    cfg = RewardCfg(entropy_weight=0.0)
    edits = [FileEdit(path="x.py", old="a", new="b")]
    inj = InjectionResult(injected=None, oracle_inverse=edits, edits=edits)
    won = injector_fitness(inj, [False, False, False], cfg)   # all fixers failed
    lost = injector_fitness(inj, [True, True, True], cfg)     # all fixed it
    assert won > lost, f"won ({won}) should beat lost ({lost})"
    assert abs(won - 1.0) < 1e-6 and abs(lost) < 1e-6
    print(f"[ok] injector_fitness: won={won:.3f} > lost={lost:.3f}")


# ---------------------------------------------------------------------------
# 5. Full self-play generation (end-to-end on the synthetic repo)
# ---------------------------------------------------------------------------
def test_full_selfplay_generation():
    """One generation of 3 injectors + 12 fixers runs end-to-end on CPU.

    Uses deterministic stub agents (a fixed bug edit + a fixed fix edit) so the
    run is reproducible and needs no LLM. Verifies the orchestrator produces
    finite fitnesses for all 15 members and a nonzero center update.
    """
    repo = _make_synthetic_repo()
    try:
        cfg = EggrollConfig()
        cfg.repos = ReposCfg(repos=[repo], test_command="python test_calc.py",
                             workdir=tempfile.mkdtemp(prefix="eggroll_sp_"),
                             test_timeout=30)
        cfg.roles.num_injectors = 3
        cfg.roles.num_fixers = 12
        cfg.roles.tool_fixers = 6
        cfg.roles.bare_fixers = 6
        cfg.train.generations = 1
        cfg.train.max_turns = 2
        cfg.eggroll.rank = 2
        cfg.eggroll.sigma = 0.05
        cfg.eggroll.lr = 0.05

        model = _tiny_linear_model()
        device = "cpu"

        # Stub role factory: injectors emit a bug, fixers emit the oracle fix.
        bug_edit = FileEdit(path="calc.py", old="return a + b",
                            new="return a - b")
        fix_edit = FileEdit(path="calc.py", old="return a - b",
                            new="return a + b")

        def role_factory(role, has_tools, dev):
            if role == "injector":
                return make_stub_injector(dev, bug_edit)
            return make_stub_fixer(dev, fix_edit, has_tools)

        trainer = EggrollTrainer(cfg, model, tokenizer=None, device=device,
                                 role_factory=role_factory)
        history = trainer.train(output_dir=tempfile.mkdtemp(prefix="eggroll_out_"))

        assert len(history) == 1
        stats = history[0]
        assert torch.isfinite(torch.tensor(stats.mean_fitness)), \
            f"mean fitness not finite: {stats.mean_fitness}"
        assert torch.isfinite(torch.tensor(stats.best_fitness)), \
            f"best fitness not finite: {stats.best_fitness}"
        assert stats.mean_abs_step > 0, "center theta didn't update"
        # Fixers using the correct oracle fix should pass -> high pass rate.
        assert stats.mean_pass_rate > 0.5, \
            f"expected fixers to mostly pass, got {stats.mean_pass_rate}"
        print(f"[ok] full self-play gen: 3 inj + 12 fix on synthetic repo. "
              f"mean_fit={stats.mean_fitness:.3f} best={stats.best_fitness:.3f} "
              f"({stats.best_role}) pass={stats.mean_pass_rate:.2f} "
              f"step={stats.mean_abs_step:.2e}")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("EGGROLL self-play RL CPU smoke test")
    print("=" * 60)
    test_rank_utilities()
    test_eggroll_update_moves_toward_winner()
    test_update_preserves_param_count()
    test_repo_env_inject_and_fix()
    test_tool_registry()
    test_fixer_fitness_penalizes_tools_and_retries()
    test_injector_fitness_rewards_fixer_failure()
    test_full_selfplay_generation()
    print("=" * 60)
    print("ALL EGGROLL SMOKE TESTS PASSED")
    print("=" * 60)
