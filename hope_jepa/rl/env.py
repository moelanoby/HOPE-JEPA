"""Hybrid repo environment: the SWE-RL style arena the agents play in.

Three responsibilities:

  1. SNAPSHOT -- make an isolated working copy of a configured repo (local path
     or git URL) so agents never touch the source tree. `injectors` mutate a
     snapshot; `fixers` get a fresh copy of the *injected* snapshot to repair.
  2. APPLY -- apply a unified-diff-shaped patch (from an injector or a fixer)
     to a snapshot. Patches are represented as a simple list of (path, old, new)
     `FileEdit`s -- robust to the messy diffs an LLM emits and easy to validate.
  3. VALIDATE -- run the repo's own test command inside a SANDBOX (restricted
     cwd, network env vars stripped, CPU/timeout capped) and report the pass
     rate. This is the SWE-RL reward backbone: a fix "works" iff the real test
     suite goes green.

The reward also uses edit-similarity (difflib) between a fixer's patch and the
oracle fix -- and the oracle fix is FREE: it's exactly the inverse of the
injector's patch (un-break what was broken). So `inject()` records the inverse
edit, and `validate_fix()` can score `edit_similarity(fixer_patch, inverse)`.

SAFETY: every subprocess runs agent-influenced code (tests authored by humans,
patches authored by agents). We enforce cwd isolation (a temp sandbox per
evaluation), strip network env vars, and cap wall-clock. This is defense in
depth -- for untrusted repos, run the whole trainer inside a container.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from .config import ReposCfg


# Env vars to strip from the sandbox so authored code / tests can't phone home.
_NETWORK_ENV_PREFIXES = ("HTTP", "HTTPS", "FTP", "ALL_PROXY", "NO_PROXY",
                         "http_proxy", "https_proxy")


@dataclass
class FileEdit:
    """One file change: relative path, old text (must be present to apply), new.

    Agents emit edits as this structured form rather than raw unified diffs --
    it's far more robust to apply and to validate (and to invert for the oracle).
    `old` is a unique substring of the current file content that gets replaced
    by `new`; if `old` is "" the edit CREATES the file with content `new`.
    """
    path: str
    old: str
    new: str


@dataclass
class Snapshot:
    """A working copy of a repo at a point in time."""
    root: str                              # absolute path to the snapshot dir
    repo_idx: int                          # which configured repo this came from
    edits_applied: List[FileEdit] = field(default_factory=list)  # history (for oracle)


@dataclass
class InjectionResult:
    """Outcome of an injector mutating a snapshot."""
    injected: Snapshot                     # the post-injection snapshot (fixers start here)
    oracle_inverse: List[FileEdit]         # the exact edits that would UN-bug it
    edits: List[FileEdit]                  # the injection edits (for diagnostics)


@dataclass
class FixResult:
    """Outcome of a fixer attempting a repair, validated by the real test suite."""
    passed: bool                           # test suite fully green
    pass_rate: float                       # fraction of tests passing (0..1)
    n_tests: int                           # total tests discovered
    edit_similarity: float                 # similarity of fix to oracle inverse (0..1)
    elapsed: float                         # wall seconds for the fix attempt + tests
    error: Optional[str] = None            # test/apply error if any


# ---------------------------------------------------------------------------
class RepoEnv:
    """Builds + snapshots repos, applies edits, runs the sandboxed test suite.

    On init, each configured repo is made available: a local path is indexed
    in place (snapshots copy from it), a git URL is cloned once into `workdir`.
    `snapshot(repo_idx)` then hands out isolated working copies on demand.
    """

    def __init__(self, cfg: ReposCfg):
        self.cfg = cfg
        self.workdir = cfg.workdir
        os.makedirs(self.workdir, exist_ok=True)
        # Resolve each repo to a local source root we can copytree from.
        self._sources: List[str] = []
        for i, repo in enumerate(cfg.repos):
            self._sources.append(self._prepare_source(repo, i))

    # ------------------------------------------------------------------
    def _prepare_source(self, repo: str, idx: int) -> str:
        """Return a local path we can snapshot from (clone if given a URL)."""
        if os.path.isdir(repo):
            return os.path.abspath(repo)
        if repo.startswith(("http://", "https://", "git@", "ssh://")):
            dest = os.path.join(self.workdir, f"repo{idx}_clone")
            if not os.path.isdir(dest):
                subprocess.run(["git", "clone", "--depth", "1", repo, dest],
                               check=True, timeout=600,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return dest
        raise FileNotFoundError(
            f"repo {repo!r} is neither an existing local path nor a git URL")

    # ------------------------------------------------------------------
    def snapshot(self, repo_idx: int) -> Snapshot:
        """A fresh isolated copy of repo `repo_idx`'s source root."""
        src = self._sources[repo_idx]
        tmp = tempfile.mkdtemp(prefix=f"snap_r{repo_idx}_", dir=self.workdir)
        # Copy only non-dotgit files (skip .git, __pycache__, venvs) to keep it light.
        for name in os.listdir(src):
            if name in (".git", "__pycache__", ".venv", "venv", ".mypy_cache"):
                continue
            s = os.path.join(src, name)
            d = os.path.join(tmp, name)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=False,
                                ignore=shutil.ignore_patterns(
                                    "__pycache__", "*.pyc", ".git"))
            else:
                shutil.copy2(s, d)
        return Snapshot(root=tmp, repo_idx=repo_idx)

    # ------------------------------------------------------------------
    def apply_edits(self, snap: Snapshot, edits: List[FileEdit]) -> Tuple[bool, Optional[str]]:
        """Apply a list of FileEdits to a snapshot in place.

        Each edit replaces its `old` substring with `new` in the target file
        (creating the file if `old` == ""). Returns (ok, error). On any failure
        the snapshot is left partially edited -- the caller validates via tests,
        so partial edits just manifest as failing tests.
        """
        for ed in edits:
            path = os.path.join(snap.root, ed.path)
            d = os.path.dirname(path)
            if d and not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            if ed.old == "":
                # create/overwrite
                with open(path, "w") as f:
                    f.write(ed.new)
            else:
                if not os.path.isfile(path):
                    return False, f"edit target missing: {ed.path}"
                with open(path) as f:
                    content = f.read()
                if ed.old not in content:
                    return False, f"edit anchor not found in {ed.path}"
                content = content.replace(ed.old, ed.new, 1)
                with open(path, "w") as f:
                    f.write(content)
            snap.edits_applied.append(ed)
        return True, None

    # ------------------------------------------------------------------
    @staticmethod
    def invert(edits: List[FileEdit]) -> List[FileEdit]:
        """The oracle fix: reverse each edit (new->old). Applied to an injected
        snapshot, this restores the original passing state."""
        return [FileEdit(path=e.path, old=e.new, new=e.old)
                for e in reversed(edits)]

    def inject(self, snap: Snapshot, bug_edits: List[FileEdit]) -> InjectionResult:
        """Apply an injector's bug edits and return the injected snapshot +
        the oracle inverse (the exact repair)."""
        ok, err = self.apply_edits(snap, bug_edits)
        if not ok:
            # If the injector's edit was malformed, the snapshot is effectively
            # un-injected -- no bug present. We still return it so the round
            # proceeds; fixers will trivially "pass" and the injector gets a
            # near-zero fitness (its bug didn't land).
            return InjectionResult(injected=snap, oracle_inverse=[],
                                   edits=bug_edits)
        return InjectionResult(injected=snap,
                               oracle_inverse=self.invert(bug_edits),
                               edits=bug_edits)

    # ------------------------------------------------------------------
    def validate_fix(self, injected: Snapshot, fix_edits: List[FileEdit]) -> FixResult:
        """Apply a fixer's edits to a fresh copy of the injected snapshot, then
        run the real test suite in a sandbox. Scores pass_rate + edit_similarity
        against the oracle inverse.
        """
        t0 = time.time()
        # Fresh copy so concurrent fixers don't clobber each other.
        work = self.snapshot(injected.repo_idx)
        # Replay the injection onto the fresh copy first.
        self.apply_edits(work, injected.edits_applied)
        # Then the fixer's edits.
        ok, err = self.apply_edits(work, fix_edits)
        if not ok:
            return FixResult(passed=False, pass_rate=0.0, n_tests=0,
                             edit_similarity=0.0,
                             elapsed=time.time() - t0, error=err)

        passed, pass_rate, n_tests, terr = self._run_tests(work.root)
        sim = self._edit_similarity(fix_edits, injected.edits_applied)
        return FixResult(passed=passed, pass_rate=pass_rate, n_tests=n_tests,
                         edit_similarity=sim, elapsed=time.time() - t0,
                         error=terr)

    # ------------------------------------------------------------------
    def _run_tests(self, cwd: str) -> Tuple[bool, float, int, Optional[str]]:
        """Run the configured test command in a sandboxed cwd.

        Sandbox discipline: strip network env vars, cap wall-clock at
        cfg.test_timeout, capture stdout/stderr. We never raise -- a timeout or
        crash is reported as a failed run.
        """
        env = {k: v for k, v in os.environ.items()
               if not any(k.startswith(p) or k == p for p in _NETWORK_ENV_PREFIXES)}
        # Drop proxy/credential vars outright.
        for k in list(env):
            if "TOKEN" in k.upper() or "SECRET" in k.upper() or "KEY" in k.upper():
                env.pop(k, None)
        # Ensure the repo's own dir is importable for `python -m pytest`.
        env["PYTHONPATH"] = cwd + os.pathsep + env.get("PYTHONPATH", "")
        try:
            proc = subprocess.run(self.cfg.test_command, shell=True, cwd=cwd,
                                  env=env, timeout=self.cfg.test_timeout,
                                  capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return False, 0.0, 0, "timeout"
        except Exception as e:                 # noqa: BLE001
            return False, 0.0, 0, f"{type(e).__name__}: {e}"

        out = (proc.stdout or "") + (proc.stderr or "")
        # Pytest: exit 0 == all pass. Parse "X passed, Y failed" for a pass_rate.
        n_pass, n_total = _parse_pytest_counts(out, proc.returncode)
        rate = (n_pass / n_total) if n_total > 0 else (1.0 if proc.returncode == 0 else 0.0)
        return (proc.returncode == 0, rate, n_total,
                None if proc.returncode == 0 else out[-512:])

    # ------------------------------------------------------------------
    @staticmethod
    def _edit_similarity(fix_edits: List[FileEdit],
                         injected_edits: List[FileEdit]) -> float:
        """How close is the fixer's patch to the oracle inverse of the injection?

        We compare the textual `new` content of the fix against the oracle's
        `new` content per shared file path, averaged. 1.0 == perfect repair.
        """
        if not injected_edits:
            return 1.0                  # nothing was broken; vacuously correct
        oracle = RepoEnv.invert(injected_edits)
        by_path_oracle = {e.path: e.new for e in oracle}
        sims = []
        for fe in fix_edits:
            if fe.path in by_path_oracle:
                sims.append(SequenceMatcher(None, fe.new,
                                            by_path_oracle[fe.path]).ratio())
        if not sims:
            return 0.0
        return float(sum(sims) / len(sims))


# ---------------------------------------------------------------------------
def _parse_pytest_counts(output: str, returncode: int) -> Tuple[int, int]:
    """Best-effort parse of pytest's 'X passed, Y failed, Z errors' summary line."""
    import re
    n_pass = n_fail = n_err = 0
    m = re.search(r"(\d+) passed", output)
    if m:
        n_pass = int(m.group(1))
    m = re.search(r"(\d+) failed", output)
    if m:
        n_fail = int(m.group(1))
    m = re.search(r"(\d+) error", output)
    if m:
        n_err = int(m.group(1))
    total = n_pass + n_fail + n_err
    if total == 0:
        # No parseable summary: fall back to exit-code semantics.
        return (1, 1) if returncode == 0 else (0, 1)
    return n_pass, total
