"""
The Cell class for httk-atomistic.
"""

import math
from collections.abc import Sequence


class Cell:
    """
    A crystallographic cell: the 3x3 matrix of cell (basis) vectors.

    A Cell holds three cell vectors as the rows of a 3x3 ``matrix`` and exposes the
    basic derived quantities ``lengths`` (the row norms), ``angles`` (the crystallographic
    ``alpha``/``beta``/``gamma`` in degrees), and ``volume`` (the absolute determinant).

    Note: the numeric values are stored as interim nested tuples of floats and the
    derived quantities use plain float arithmetic. They are intended to be replaced by
    the httk exact vector representation fairly soon; keep numeric access behind the
    ``matrix`` accessor so that change stays contained.
    """

    _matrix: tuple[tuple[float, ...], ...]

    def __init__(self, matrix: Sequence[Sequence[float]]) -> None:
        norm = tuple(tuple(float(x) for x in row) for row in matrix)
        if len(norm) != 3 or any(len(row) != 3 for row in norm):
            raise ValueError("Cell matrix must be a 3x3 sequence")
        self._matrix = norm

    @property
    def matrix(self) -> tuple[tuple[float, ...], ...]:
        """The 3x3 cell vectors as nested float tuples (one vector per row)."""
        return self._matrix

    @property
    def lengths(self) -> tuple[float, float, float]:
        """The lengths of the three cell vectors (the row norms)."""
        rows = self._matrix
        return (
            math.sqrt(sum(x * x for x in rows[0])),
            math.sqrt(sum(x * x for x in rows[1])),
            math.sqrt(sum(x * x for x in rows[2])),
        )

    @property
    def angles(self) -> tuple[float, float, float]:
        """
        The cell angles ``(alpha, beta, gamma)`` in degrees.

        Following the crystallographic convention, ``alpha`` is the angle between rows
        ``b`` and ``c``, ``beta`` between ``a`` and ``c``, and ``gamma`` between ``a``
        and ``b``.
        """
        a, b, c = self._matrix
        return (self._angle(b, c), self._angle(a, c), self._angle(a, b))

    @property
    def volume(self) -> float:
        """The cell volume, the absolute value of the determinant of ``matrix``."""
        (a0, a1, a2), (b0, b1, b2), (c0, c1, c2) = self._matrix
        det = a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0)
        return abs(det)

    @staticmethod
    def _angle(u: Sequence[float], v: Sequence[float]) -> float:
        dot = sum(ui * vi for ui, vi in zip(u, v))
        nu = math.sqrt(sum(ui * ui for ui in u))
        nv = math.sqrt(sum(vi * vi for vi in v))
        cosine = max(-1.0, min(1.0, dot / (nu * nv)))
        return math.degrees(math.acos(cosine))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cell):
            return NotImplemented
        return self._matrix == other._matrix

    def __repr__(self) -> str:
        return f"Cell(matrix={self._matrix!r})"
