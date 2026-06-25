"""The two agent roles: Injector and Fixer.

Each role wraps a *perturbed* HopeLLM population member (theta_p) and turns it
into an action-producing agent over the repo environment. The key point: under
EGGROLL these members are evaluated with INFERENCE-ONLY rollouts (no gradients),
so `act()` runs the model in `torch.no_grad()` and produces:

  * the agent's emitted actions (edits / tools / tool-calls),
  * per-token entropies (GTPO needs these for injector credit; GRPO-S uses an
    action-distribution entropy instead),
  * bookkeeping the orchestrator needs (tool count, retries/turns, elapsed).

Parsing strategy: agents emit a small structured protocol so we don't have to
deal with arbitrary prose. The shared grammar is:

    <edit path="rel/path.py">
    old: <exact anchor text>
    new: <replacement text>
    </edit>

    <tool name="foo"> def foo(...): ... </tool>

    <call name="foo"> {"k": "v"} </call>

This is robust to LLM messiness (each block is parsed independently; malformed
blocks are skipped and charged to the retry budget). The Injector only emits
`<edit>` blocks (its "bug"). The Fixer emits `<edit>`, and if `has_tools`, also
`<tool>` and `<call>` blocks.

All generation goes through `model.model.generate(...)` (the HopeLLM wrapper
delegates to the HF model's generate, which recomputes through the HOPE blocks
-- correct, not fast, which is fine for rollouts). On CPU/smoke we bypass the
LLM entirely with a deterministic stub (see `_StubLLM`) so the smoke test runs
with no model download.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from .env import FileEdit, Snapshot
from .tools import ToolRegistry, parse_tools
from .reward import normalized_entropy


# ---------------------------------------------------------------------------
# Structured-action parser (shared by both roles)
# ---------------------------------------------------------------------------
_EDIT_RE = re.compile(
    r"<edit(?:\s+path=[\"'](?P<path>[^\"']+)[\"']\s*)?\s*>"
    r"(?P<body>.*?)</edit>",
    re.DOTALL | re.IGNORECASE,
)
_CALL_RE = re.compile(
    r"<call\s+name=[\"'](?P<name>[^\"']+)[\"']\s*>(?P<args>.*?)</call>",
    re.DOTALL | re.IGNORECASE,
)


def _parse_edit_block(block: str, path_hint: Optional[str] = None) -> Optional[FileEdit]:
    """Parse the body of one <edit> block into a FileEdit.

    Body convention:
        old: <anchor>
        new: <replacement>
    Lines after the first `new:` form the replacement (joined with \n).
    """
    lines = block.strip().splitlines()
    old_lines, new_lines = [], []
    mode = None
    for ln in lines:
        ls = ln.strip()
        if ls.lower().startswith("old:"):
            mode = "old"
            old_lines.append(ls[4:].strip())
            continue
        if ls.lower().startswith("new:"):
            mode = "new"
            new_lines.append(ls[4:].strip())
            continue
        if mode == "old":
            old_lines.append(ln)
        elif mode == "new":
            new_lines.append(ln)
    if not new_lines:
        return None
    return FileEdit(
        path=path_hint or "",
        old="\n".join(old_lines).strip(),
        new="\n".join(new_lines).strip(),
    )


def parse_edits(text: str) -> List[FileEdit]:
    """All <edit path="...">...</edit> blocks in agent output -> FileEdits."""
    edits: List[FileEdit] = []
    for m in _EDIT_RE.finditer(text):
        path = m.group("path") or ""
        ed = _parse_edit_block(m.group("body"), path_hint=path)
        if ed is not None:
            edits.append(ed)
    return edits


def parse_calls(text: str) -> List[Tuple[str, dict]]:
    """All <call name="...">{json}</call> blocks -> (name, kwargs)."""
    calls = []
    for m in _CALL_RE.finditer(text):
        name = m.group("name")
        raw = m.group("args").strip()
        try:
            kwargs = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            kwargs = {}
        calls.append((name, kwargs))
    return calls


def _token_entropy(token_ids: torch.Tensor, logits: torch.Tensor) -> List[float]:
    """Per-token entropy of a sampled sequence from its logits (nats).

    logits: [T, V] for the T generated tokens. Used as the GTPO per-token signal
    for the injector's diff tokens.
    """
    import torch.nn.functional as F
    logp = F.log_softmax(logits.float(), dim=-1)          # [T, V]
    p = logp.exp()
    ent = -(p * logp).sum(dim=-1)                          # [T]
    return ent.tolist()


# ---------------------------------------------------------------------------
# Action records (what act() returns)
# ---------------------------------------------------------------------------
@dataclass
class InjectorAction:
    edits: List[FileEdit]                   # the bug
    token_entropies: List[float] = field(default_factory=list)  # for GTPO


@dataclass
class FixerAction:
    edits: List[FileEdit]
    tools_authored: int
    tool_calls: int
    turns: int                              # retries used
    elapsed: float
    action_entropy: float                   # normalized -> GRPO-S term


# ---------------------------------------------------------------------------
# Role abstractions
# ---------------------------------------------------------------------------
class _Role:
    """Common base: holds a model + tokenizer and a `_generate` helper.

    The model is a HopeLLM (or the stub below for the smoke test). We call
    `.model.generate(...)` (the underlying HF model) so the HOPE-augmented stack
    is used. Inference-only throughout -- EGGROLL does not backprop through the
    agents.
    """

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _generate(self, prompt: str, max_new_tokens: int,
                  temperature: float = 0.7) -> Tuple[str, List[float]]:
        """Sample a completion + return per-token entropies (for GTPO)."""
        toks = self.tokenizer(prompt, return_tensors="pt")
        ids = toks["input_ids"].to(self.device)
        with torch.no_grad():
            out = self.model.model.generate(
                input_ids=ids,
                attention_mask=toks.get("attention_mask",
                                        torch.ones_like(ids)).to(self.device),
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-3),
                top_p=0.9,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )
        gen_ids = out.sequences[0, ids.shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        # Per-token entropy from the step scores (log-probs over vocab).
        ents: List[float] = []
        if getattr(out, "scores", None):
            for sc in out.scores:
                ents.append(_token_entropy(torch.zeros(1, dtype=torch.long),
                                           sc[-1:].float())[0])
        return text, ents


class Injector(_Role):
    """An injector agent: reads a clean snapshot and emits a bug (edits).

    GTPO uses the per-token entropy of its emitted diff, so `_generate` returns it.
    """

    def __init__(self, model, tokenizer, device: str, max_bug_lines: int = 40):
        super().__init__(model, tokenizer, device)
        self.max_bug_lines = max_bug_lines

    def act(self, snap: Snapshot) -> InjectorAction:
        listing = _list_snapshot(snap, max_files=8, max_lines=12)
        prompt = _injector_prompt(listing)
        text, ents = self._generate(prompt, max_new_tokens=self.max_bug_lines * 3,
                                    temperature=0.9)
        edits = parse_edits(text)
        return InjectorAction(edits=edits, token_entropies=ents[:max(len(edits) * 4, 1)])


class Fixer(_Role):
    """A fixer agent: races to repair an injected snapshot.

    `has_tools` controls whether it may author/call tools (the 6 tool-fixers
    get True; the 6 bare-fixers get False). Runs a multi-turn loop up to
    `max_turns`: each turn it emits edits + (optionally) tool blocks/calls,
    which are applied; it stops when it emits an explicit `<done/>` marker or
    runs out of turns.
    """

    def __init__(self, model, tokenizer, device: str, has_tools: bool):
        super().__init__(model, tokenizer, device)
        self.has_tools = has_tools

    def act(self, snap: Snapshot, registry: ToolRegistry,
            max_turns: int, env) -> Tuple[FixerAction, List[FileEdit]]:
        """Return (action record, all edits emitted across turns)."""
        t0 = time.time()
        all_edits: List[FileEdit] = []
        action_counts = [0, 0, 0]   # [edits, tool_author, tool_call]
        turns = 0

        for turn in range(max_turns):
            turns = turn + 1
            listing = _list_snapshot(snap, max_files=8, max_lines=12,
                                     highlight="breaks|fail|error|bug")
            prompt = _fixer_prompt(listing, self.has_tools,
                                   tools=registry.available, turn=turn)
            text, _ = self._generate(prompt, max_new_tokens=160, temperature=0.6)

            if "<done/>" in text.lower():
                break

            # Tool authors + calls (only honored if has_tools).
            tools_authored = 0
            tool_calls = 0
            if self.has_tools:
                for td in parse_tools(text):
                    if registry.register(td):
                        tools_authored += 1
                for name, kwargs in parse_calls(text):
                    try:
                        registry.invoke(name, **kwargs)
                        tool_calls += 1
                    except Exception:           # noqa: BLE001 - fail-closed
                        tool_calls += 1         # counts as a spent action

            edits = parse_edits(text)
            if edits:
                ok, _ = env.apply_edits(snap, edits)
                all_edits.extend(edits)
                action_counts[0] += len(edits)
            action_counts[1] += tools_authored
            action_counts[2] += tool_calls

            # If the fixer emitted nothing actionable this turn, stop early --
            # spinning wastes the retry budget anyway.
            if not edits and not tools_authored and not tool_calls:
                break

        elapsed = time.time() - t0
        action = FixerAction(
            edits=all_edits,
            tools_authored=action_counts[1],
            tool_calls=action_counts[2],
            turns=turns,
            elapsed=elapsed,
            action_entropy=normalized_entropy([max(c, 0) for c in action_counts]),
        )
        return action, all_edits


# ---------------------------------------------------------------------------
# Prompts + snapshot listing helpers
# ---------------------------------------------------------------------------
def _list_snapshot(snap: Snapshot, max_files: int = 8,
                   max_lines: int = 10, highlight: Optional[str] = None) -> str:
    """A compact textual view of a snapshot for the agent prompt."""
    import os
    files = []
    for root, _, names in os.walk(snap.root):
        for n in sorted(names):
            if n.endswith((".py", ".txt", ".md")) and "__pycache__" not in root:
                rel = os.path.relpath(os.path.join(root, n), snap.root)
                files.append(rel)
    files = files[:max_files]
    out = []
    for rel in files:
        full = os.path.join(snap.root, rel)
        try:
            with open(full) as f:
                lines = f.readlines()
        except (OSError, UnicodeDecodeError):
            continue
        if highlight:
            lines = [l for l in lines
                     if re.search(highlight, l, re.IGNORECASE)] or lines[:2]
        body = "".join(lines[:max_lines])
        out.append(f"--- {rel} ---\n{body}")
    return "\n\n".join(out) if out else "(empty snapshot)"


def _injector_prompt(listing: str) -> str:
    return (
        "You are an injector. Introduce ONE subtle bug that makes a test fail "
        "but is hard to spot. Emit exactly one edit block:\n\n"
        "<edit path=\"rel/path.py\">\nold: <exact existing line>\n"
        "new: <buggy line>\n</edit>\n\n"
        "Repo files:\n" + listing + "\n\nYour edit:")


def _fixer_prompt(listing: str, has_tools: bool, tools: List[str],
                  turn: int) -> str:
    tools_line = ""
    if has_tools:
        avail = ", ".join(tools) if tools else "(none yet)"
        tools_line = (
            f"\nYou may author tools (<tool name=\"x\">def x(...): ...</tool>) "
            f"and call them (<call name=\"x\">{{\"k\":\"v\"}}</call>). "
            f"Available tools: {avail}. Emit <done/> when the bug is fixed.")
    return (
        f"You are a fixer (turn {turn}). A bug was injected. Find and fix it with "
        f"the FEWEST actions possible. Emit an <edit path=\"...\"> block.\n"
        f"Repo files:\n{listing}{tools_line}\n\nYour action:")


# ---------------------------------------------------------------------------
# CPU/smoke stub: lets the smoke test run without a real LLM
# ---------------------------------------------------------------------------
class _StubLLM:
    """A deterministic stand-in for HopeLLM in the smoke test.

    Produces a fixed edit to a known anchor, so the orchestrator + env + reward
    pipeline can be exercised end-to-end on CPU with no model download. The real
    trainer never uses this -- it builds a HopeLLM from `model.hope_config`.
    """

    def __init__(self, *, inject_bug: bool):
        # inject_bug=True: emit a bug (injector stub); False: emit a no-op (fixer stub).
        self._inject = inject_bug
        self.model = self       # so _Role._generate's `self.model.model.generate` works

    def generate(self, **kwargs):                       # noqa: ARG002
        return None   # the stub bypasses _generate via the injected _generate below


def make_stub_injector(device: str, bug_edit: FileEdit) -> Injector:
    """An Injector that deterministically emits `bug_edit` (smoke test only)."""

    class _StubInjector(Injector):
        def act(self, snap):  # noqa: ARG002
            return InjectorAction(edits=[bug_edit], token_entropies=[1.0])

    tok = _StubTokenizer()
    return _StubInjector(_StubLLM(inject_bug=True), tok, device)


def make_stub_fixer(device: str, fix_edit: FileEdit, has_tools: bool) -> Fixer:
    """A Fixer that deterministically emits `fix_edit` (smoke test only)."""

    class _StubFixer(Fixer):
        def act(self, snap, registry, max_turns, env):  # noqa: ARG002
            return (
                FixerAction(edits=[fix_edit], tools_authored=0, tool_calls=0,
                            turns=1, elapsed=0.001,
                            action_entropy=0.0),
                [fix_edit],
            )

    tok = _StubTokenizer()
    return _StubFixer(_StubLLM(inject_bug=False), tok, device, has_tools)


class _StubTokenizer:
    """Minimal tokenizer interface for the stub path."""
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, *a, **k):
        return {"input_ids": torch.zeros((1, 1), dtype=torch.long),
                "attention_mask": torch.ones((1, 1), dtype=torch.long)}

    def decode(self, *a, **k):
        return ""
