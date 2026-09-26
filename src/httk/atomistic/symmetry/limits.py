"""Cooperative resource limits for explicit canonicalization and symmetry searches."""

import math
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

__all__ = ["CanonicalizationLimitError"]


class CanonicalizationLimitError(RuntimeError):
    """Canonicalization could not finish within its requested resource limit.

    No partially selected canonical representative is returned. Deadlines are
    cooperative: a native call or a single arithmetic operation is not preempted.
    """


@dataclass
class _Budget:
    deadline: float | None
    reasons: list[str] = field(default_factory=list)
    incomplete_events: int = 0


_active: ContextVar[_Budget | None] = ContextVar("httk_symmetry_budget", default=None)


@contextmanager
def _budget_scope(timeout: float | None) -> Iterator[_Budget]:
    if timeout is not None and (isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be a positive finite number or None")
    parent = _active.get()
    deadline = None if timeout is None else time.monotonic() + timeout
    if parent is not None and parent.deadline is not None:
        deadline = parent.deadline if deadline is None else min(deadline, parent.deadline)
    budget = _Budget(deadline)
    token = _active.set(budget)
    try:
        yield budget
    finally:
        _active.reset(token)


def _checkpoint() -> None:
    budget = _active.get()
    if budget is not None and budget.deadline is not None and time.monotonic() >= budget.deadline:
        _incomplete("deadline_exceeded")
        raise CanonicalizationLimitError("canonicalization deadline exceeded; no canonical result was selected")


def _incomplete(reason: str) -> None:
    budget = _active.get()
    if budget is not None:
        budget.incomplete_events += 1
        if reason not in budget.reasons:
            budget.reasons.append(reason)


def _incomplete_count() -> int:
    budget = _active.get()
    return 0 if budget is None else budget.incomplete_events
