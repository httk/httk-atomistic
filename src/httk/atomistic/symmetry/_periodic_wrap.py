"""Wrapping reduced coordinates into the cell along the periodic directions only.

:meth:`~httk.core.FracVector.normalize` and
:meth:`~httk.core.FracVector.normalize_half` are elementwise over all three components,
which is exactly right for a crystal and wrong for anything else. Along a direction that
carries no lattice translation there is nothing to wrap *into*: a coordinate of ``1.05``
there is a real position 1.05 frame-vectors out, not the same place as ``0.05``. Folding it
would teleport the atom, and — worse, because it is silent — would make two genuinely
different structures compare equal.

These helpers apply the underlying wrap and then put the non-periodic columns back as they
were. For the overwhelmingly common fully-periodic case they short-circuit to the plain
httk-core call, so a crystal pays nothing for the generality.
"""

from typing import Any

from httk.core import FracVector

__all__ = ["wrap_periodic", "wrap_periodic_half"]


def wrap_periodic(coords: Any, periodicity: tuple[bool, bool, bool]) -> FracVector:
    """``coords`` wrapped into ``[0, 1)``, along the periodic directions only."""
    return _wrap(coords, periodicity, half=False)


def wrap_periodic_half(coords: Any, periodicity: tuple[bool, bool, bool]) -> FracVector:
    """``coords`` wrapped into ``[-1/2, 1/2)``, along the periodic directions only.

    The nearest-image reduction of a difference vector. Along a non-periodic direction there
    is no other image to be nearer, so the difference is left as it stands.
    """
    return _wrap(coords, periodicity, half=True)


def _wrap(coords: Any, periodicity: tuple[bool, bool, bool], *, half: bool) -> FracVector:
    exact = coords if isinstance(coords, FracVector) else FracVector(coords)
    wrapped = exact.normalize_half() if half else exact.normalize()
    if all(periodicity):
        return wrapped
    if not any(periodicity):
        return exact
    # Rebuild column by column rather than in place: these vectors are immutable, and the
    # rebuild keeps the exact rationals exact.
    rows = exact.to_fractions()
    wrapped_rows = wrapped.to_fractions()
    if exact.dim == (3,):
        return FracVector([wrapped_rows[i] if periodicity[i] else rows[i] for i in range(3)])
    return FracVector(
        [
            [wrapped_row[i] if periodicity[i] else row[i] for i in range(3)]
            for row, wrapped_row in zip(rows, wrapped_rows)
        ]
    )
