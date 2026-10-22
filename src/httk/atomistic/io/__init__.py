"""Atomistic file-format readers and writers."""

import math


def _check_load_precision(precision: float | None, reader: str) -> None:
    """Validate the uniform ``load(..., precision=...)`` keyword shared by the structure readers."""
    if precision is not None and (
        isinstance(precision, bool)
        or not isinstance(precision, (int, float))
        or not math.isfinite(precision)
        or precision <= 0
    ):
        raise ValueError(f"{reader} precision must be a finite number greater than zero, got {precision!r}.")
