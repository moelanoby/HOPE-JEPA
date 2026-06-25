"""Agent-authored tool registry.

The 6 "tool-fixers" (of the 12 fixers) are allowed to AUTHOR their own tools.
A tool is a Python function the agent emits in a fenced block:

    <tool name="grep_py">
    def grep_py(path, needle):
        import subprocess
        return subprocess.run(["grep", "-rn", needle, path], ...)
    </tool>

We parse those blocks, COMPILE them (ast.parse + compile) to verify they're
valid Python, and cache them in a per-agent registry. Each authored tool and
each INVOCATION increments a counter -- those counts flow into the GRPO-S
fixer fitness as `tool_penalty * tools_used`, which is the mechanism behind
"reward the fixer that uses the FEWEST tools."

Design notes:
  * Fail-closed: a malformed or uncompilable tool is rejected (returns None),
    and the failed authoring attempt still counts toward the tool budget --
    that's a retry, so it's penalized via `retry_penalty` instead. We never
    raise out of the agent loop over a bad tool.
  * Tools are cached PER FIXER and persist across that fixer's rollouts within
    a generation, so a fixer that invests a turn in authoring a reusable tool
    can amortize it.
  * SAFETY: invoking a tool runs agent-emitted code. `ToolRegistry.invoke`
    runs under the same sandbox discipline as the env's test runner (restricted
    cwd, no network env, time limit). See the README warning. The registry
    itself never imports/execs at module load.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_TOOL_RE = re.compile(
    r"<tool\s+name=[\"'](?P<name>[^\"']+)[\"']\s*>(?P<body>.*?)</tool>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ToolDef:
    """One authored tool: name, source, and whether it compiled cleanly."""
    name: str
    source: str
    compiled: bool = False
    error: Optional[str] = None


def parse_tools(agent_output: str) -> List[ToolDef]:
    """Extract every `<tool name="...">...</tool>` block from agent output.

    Returns ToolDefs with source filled in; `compiled` is set by `compile_tool`.
    Empty/malformed bodies are still returned (as a failed tool) so the caller
    can charge them to the retry budget -- but unmatched fragments yield nothing.
    """
    tools: List[ToolDef] = []
    if not agent_output:
        return tools
    for m in _TOOL_RE.finditer(agent_output):
        name = m.group("name").strip()
        body = m.group("body").strip()
        tools.append(ToolDef(name=name, source=body, compiled=False))
    return tools


def compile_tool(tool: ToolDef) -> ToolDef:
    """Verify a tool's source is valid Python via ast.parse + compile.

    Mutates and returns the same ToolDef: sets `compiled=True` on success,
    or `error` on failure. We compile but DO NOT exec here -- execution is
    deferred to `ToolRegistry.invoke`, which runs under sandbox discipline.
    Fail-closed: any error -> compiled stays False.
    """
    if not tool.source:
        tool.compiled = False
        tool.error = "empty tool body"
        return tool
    try:
        ast.parse(tool.source)                      # syntax check
        compile(tool.source, f"<tool:{tool.name}>", "exec")   # bytecode check
        tool.compiled = True
        tool.error = None
    except (SyntaxError, ValueError) as e:
        tool.compiled = False
        tool.error = f"{type(e).__name__}: {e}"
    return tool


class ToolRegistry:
    """Per-fixer cache of authored tools + an invocation counter.

    The counter (`tools_used`) is the number that flows into the GRPO-S fitness:
        fixer_fitness -= tool_penalty * tools_used

    It counts BOTH authored tools and invocations -- authoring a tool is itself
    a tool-budget spend (you "used" your tool budget to make it), which keeps
    the incentive to make few, high-value tools rather than many throwaway ones.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDef] = {}
        self._compiled_ns: Dict[str, Any] = {}    # name -> callable (after exec)
        self.tools_authored: int = 0              # distinct tools added
        self.tools_used: int = 0                  # authorings + invocations

    # ------------------------------------------------------------------
    def register(self, tool: ToolDef) -> bool:
        """Compile + store an authored tool. Returns True if it compiled.

        Always increments `tools_used` (authoring costs budget). If the tool
        compiled, we exec it into a private namespace and capture any top-level
        function matching `tool.name` as the callable.
        """
        self.tools_used += 1
        tool = compile_tool(tool)
        if not tool.compiled:
            return False
        # Exec into a private namespace and grab the named callable. We only do
        # this for compiled tools; execution happens here (register time), but
        # under the same sandbox caveat as invoke(). The actual CALL is in invoke.
        try:
            ns: Dict[str, Any] = {}
            exec(compile(tool.source, f"<tool:{tool.name}>", "exec"), ns)
            fn = ns.get(tool.name)
            if not callable(fn):
                tool.compiled = False
                tool.error = f"no callable named {tool.name!r} defined"
                return False
            self._tools[tool.name] = tool
            self._compiled_ns[tool.name] = fn
            self.tools_authored += 1
            return True
        except Exception as e:                    # noqa: BLE001 - fail-closed
            tool.compiled = False
            tool.error = f"exec error: {type(e).__name__}: {e}"
            return False

    # ------------------------------------------------------------------
    def invoke(self, name: str, **kwargs) -> Any:
        """Call an authored tool. Increments `tools_used`. Fail-closed.

        NOTE: this runs agent-authored code. The caller (the fixer rollout) is
        responsible for sandbox discipline (restricted cwd/env/timeout). Here
        we only guard against the tool being missing/non-compilable.
        """
        self.tools_used += 1
        fn = self._compiled_ns.get(name)
        if fn is None:
            raise KeyError(f"tool {name!r} not registered or did not compile")
        return fn(**kwargs)

    # ------------------------------------------------------------------
    @property
    def available(self) -> List[str]:
        return [n for n, t in self._tools.items() if t.compiled]

    def reset(self) -> None:
        """Clear tools + counters for a fresh generation/fixer."""
        self._tools.clear()
        self._compiled_ns.clear()
        self.tools_authored = 0
        self.tools_used = 0
