"""tau2-bench evaluators.

``ActionMatchEvaluator``
    Compares the agent's executed tool-call sequence against
    ``task.expected_actions`` using the same canonicalization as the
    consensus mechanism.  Works fully offline — no tau2-bench install needed.

``Tau2BenchEvaluator``
    Delegates to the official tau2-bench evaluator once the repo is installed.
    Used for the real benchmark run.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from tracerazor._canonicalize import canonical_key as _ckey
from benchmark.tau2_loader import Tau2Task


# ── action-match evaluator (offline) ─────────────────────────────────────


class ActionMatchEvaluator:
    """Pass/fail by exact-match comparison of action sequences.

    Two sequences match if they have the same length and each corresponding
    (tool_name, arguments) pair is canonically equal.

    Parameters
    ----------
    null_equiv:
        Passed to the canonicalizer (treat None/empty/missing as equal).
    require_final_answer:
        If True, the trajectory must also end with a ``final_answer`` step
        (i.e. the agent signalled completion).
    """

    def __init__(
        self,
        *,
        null_equiv: bool = True,
        require_final_answer: bool = False,
    ) -> None:
        self._null_equiv = null_equiv
        self._require_final_answer = require_final_answer

    def evaluate(self, trajectory: List[dict], task: Tau2Task) -> bool:
        """Return True if the trajectory matches ``task.expected_actions``."""
        tool_steps = [s for s in trajectory if s.get("type") == "tool_call"]
        expected = task.expected_actions

        if len(tool_steps) != len(expected):
            return False

        for actual_step, exp in zip(tool_steps, expected):
            ak = _ckey(actual_step["tool"], actual_step.get("args", {}),
                       null_equiv=self._null_equiv)
            ek = _ckey(exp["name"], exp.get("arguments", {}),
                       null_equiv=self._null_equiv)
            if ak != ek:
                return False

        if self._require_final_answer:
            has_answer = any(s.get("type") == "final_answer" for s in trajectory)
            if not has_answer:
                return False

        return True

    def __call__(self, trajectory: List[dict], task: Tau2Task) -> bool:
        return self.evaluate(trajectory, task)


# ── tau2-bench official evaluator (stub) ─────────────────────────────────


class Tau2BenchEvaluator:
    """Delegates to the tau2-bench library evaluator.

    Requires ``tau2-bench`` to be installed:
        pip install git+https://github.com/sierra-research/tau2-bench.git

    Falls back gracefully with a clear error if the import fails.
    """

    def __init__(self, domain: str) -> None:
        self.domain = domain
        self._delegate: Optional[Any] = None
        self._load_error: Optional[str] = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            from tau_bench.envs import get_env  # type: ignore
            self._env = get_env(self.domain)
        except ImportError as e:
            self._load_error = str(e)

    @property
    def available(self) -> bool:
        return self._load_error is None

    def evaluate(self, trajectory: List[dict], task: Tau2Task) -> bool:
        if self._load_error:
            raise RuntimeError(
                f"tau2-bench not installed ({self._load_error}).\n"
                "Use ActionMatchEvaluator for offline evaluation, or install:\n"
                "  pip install git+https://github.com/sierra-research/tau2-bench.git"
            )
        # Delegate to tau2-bench's own state-based evaluator
        # (concrete implementation depends on tau2-bench API)
        raise NotImplementedError("Wire to tau2-bench evaluator once installed")

    def __call__(self, trajectory: List[dict], task: Tau2Task) -> bool:
        return self.evaluate(trajectory, task)


# ── factory ───────────────────────────────────────────────────────────────


def get_evaluator(domain: str, *, offline: bool = True) -> Callable:
    """Return the best available evaluator for the domain."""
    if not offline:
        ev = Tau2BenchEvaluator(domain)
        if ev.available:
            return ev
    return ActionMatchEvaluator()
