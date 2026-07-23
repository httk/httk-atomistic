"""
The Cell class for httk-atomistic.
"""

import fractions
from typing import Any

from httk.core import SurdScalar, SurdVector, VectorLike
from httk.core.vectors import exactmath
from httk.core.vectors.exactmath import integer_sqrt

from ._vector_guards import to_surdscalar, to_surdvector

# Deterministic precision for the fallbacks where a length or angle is not exact in the surd field
# (a non-Niven angle, or an irrational squared length outside the crystallographic metric-rational
# case). Matches the exactmath default accuracy.
_FALLBACK_PREC = exactmath.default_accuracy

# Above this size the exact-surd square root (which factors the radicand via trial division up to
# its cube root) is skipped in favour of a deterministic rational approximation. Exact
# crystallographic data has tiny radicands (well under this); the only values that exceed it are the
# huge binary rationals produced by embedding an irrational float, for which no exact surd root is
# wanted anyway. A perfect square is always taken exactly (a cheap integer-sqrt test, any size).
_MAX_EXACT_RADICAND = 10**18


def _scalar_length(lsq: SurdScalar) -> SurdScalar:
    """
    The exact length ``sqrt(lsq)`` as a ``SurdScalar``, with a deterministic fallback.

    Exact when ``lsq`` is a rational perfect square (any size) or a rational with a small radicand
    (the crystallographic case). Otherwise — a huge float-derived rational, or an irrational squared
    length — a deterministic rational approximation at ``_FALLBACK_PREC``.
    """
    if lsq.is_rational:
        q = lsq._rational_fraction()
        num, den = q.numerator, q.denominator
        root_num, root_den = integer_sqrt(num), integer_sqrt(den)
        if root_num * root_num == num and root_den * root_den == den:
            return SurdVector.create(fractions.Fraction(root_num, root_den))._as_scalar()
        if num * den <= _MAX_EXACT_RADICAND:
            return SurdVector.sqrt_of(q)
        return SurdVector.create(exactmath.sqrt(q, prec=_FALLBACK_PREC, limit=True))._as_scalar()
    approx = lsq.to_fractions_approx(_FALLBACK_PREC)
    return SurdVector.create(exactmath.sqrt(approx, prec=_FALLBACK_PREC, limit=True))._as_scalar()


class Cell:
    """
    A crystallographic cell: the 3x3 matrix of cell (basis) vectors, held **exactly**.

    The lattice vectors are the rows of ``matrix``. Internally a Cell factors that matrix into a
    positive :class:`~httk.core.SurdScalar` ``scale`` times an ``unscaled_matrix``
    (a :class:`~httk.core.SurdVector` of shape ``(3, 3)``), with ``matrix == scale * unscaled_matrix``.
    The split lets an overall length factor be carried symbolically: a hexagonal cell of lattice
    parameter ``a`` and ratio ``c/a`` is the exact ``unscaled`` rows ``(1, 0, 0)``,
    ``(-1/2, sqrt(3)/2, 0)``, ``(0, 0, c/a)`` scaled by ``a`` — so the ``sqrt(3)`` stays exact
    regardless of ``a``. A cell built from an absolute matrix simply has ``scale == 1``.

    Numbers embed exactly: rationals (and rational-valued floats) stay rational, and a
    :class:`~httk.core.SurdVector` matrix keeps its radicals. Derived quantities are exact whenever
    the geometry is metric-rational (the crystallographic case): ``lengths`` come from
    :meth:`~httk.core.SurdVector.sqrt_of` of the rational squared row lengths, ``angles`` (degrees)
    from the exact reverse-Niven :meth:`~httk.core.SurdScalar.acos_degrees` where possible,
    ``volume`` from the exact determinant, and ``metric`` is the exact rational Gram matrix. When a
    squared length happens to be irrational, ``lengths``/``angles`` fall back to a deterministic
    rational approximation (documented per accessor). Floats appear only at the presentation
    boundary via :meth:`matrix_floats`.
    """

    _scale: SurdScalar
    _unscaled_matrix: SurdVector
    _matrix_cache: SurdVector | None
    _metric_cache: SurdVector | None
    _lengths_cache: tuple[SurdScalar, ...] | None
    _angles_cache: tuple[fractions.Fraction, ...] | None
    _volume_cache: SurdScalar | None

    def __init__(self, matrix: VectorLike, scale: Any = 1) -> None:
        unscaled = to_surdvector(matrix)
        if unscaled.dim != (3, 3):
            raise ValueError("Cell matrix must be a 3x3 vector-like")
        scale_scalar = to_surdscalar(scale)
        if scale_scalar.sign() <= 0:
            raise ValueError("Cell scale must be strictly positive")
        self._unscaled_matrix = unscaled
        self._scale = scale_scalar
        self._matrix_cache = None
        self._metric_cache = None
        self._lengths_cache = None
        self._angles_cache = None
        self._volume_cache = None

    @property
    def scale(self) -> SurdScalar:
        """The overall (strictly positive) length factor, as an exact ``SurdScalar``."""
        return self._scale

    @property
    def unscaled_matrix(self) -> SurdVector:
        """The 3x3 cell vectors before applying ``scale``, as an exact ``SurdVector``."""
        return self._unscaled_matrix

    @property
    def matrix(self) -> SurdVector:
        """The 3x3 lattice vectors ``scale * unscaled_matrix`` (one vector per row), exact."""
        if self._matrix_cache is None:
            self._matrix_cache = self._scale * self._unscaled_matrix
        return self._matrix_cache

    def matrix_floats(self) -> tuple[tuple[float, ...], ...]:
        """The lattice vectors as nested float tuples (presentation boundary)."""
        return tuple(tuple(row) for row in self.matrix.to_floats())

    def metric(self) -> SurdVector:
        """The exact Gram matrix ``matrix * matrix^T`` (rational for a metric-rational cell)."""
        if self._metric_cache is None:
            m = self.matrix
            self._metric_cache = m * m.T()
        return self._metric_cache

    @property
    def lengths(self) -> tuple[SurdScalar, ...]:
        """
        The lengths of the three cell vectors (the scaled row norms).

        Exact via :meth:`~httk.core.SurdVector.sqrt_of` whenever the row's squared length is
        rational (the crystallographic case); otherwise a deterministic rational-approximation
        ``SurdScalar`` at ``_FALLBACK_PREC`` (the length would be a nested radical, outside the
        surd field).
        """
        if self._lengths_cache is None:
            metric = self.metric()
            self._lengths_cache = tuple(_scalar_length(metric._element((i, i))) for i in range(3))
        return self._lengths_cache

    @property
    def angles(self) -> tuple[fractions.Fraction, ...]:
        """
        The cell angles ``(alpha, beta, gamma)`` in degrees, as exact ``Fraction`` values.

        Following the crystallographic convention, ``alpha`` is the angle between rows ``b`` and
        ``c``, ``beta`` between ``a`` and ``c``, and ``gamma`` between ``a`` and ``b``. Angles are
        scale-independent, so they are computed from the unscaled matrix. The cosine is formed
        exactly in the surd field and reversed through the Niven table
        (:meth:`~httk.core.SurdScalar.acos_degrees`) for an exact answer; a non-Niven angle falls
        back to a deterministic :func:`~httk.core.vectors.exactmath.acos` at ``_FALLBACK_PREC``.
        """
        if self._angles_cache is None:
            u = self._unscaled_matrix
            gram = u * u.T()
            self._angles_cache = (
                self._angle_from_gram(gram, 1, 2),
                self._angle_from_gram(gram, 0, 2),
                self._angle_from_gram(gram, 0, 1),
            )
        return self._angles_cache

    @staticmethod
    def _angle_from_gram(gram: SurdVector, i: int, j: int) -> fractions.Fraction:
        dot = gram._element((i, j))
        li = _scalar_length(gram._element((i, i)))
        lj = _scalar_length(gram._element((j, j)))
        cosine = (dot * (li * lj)._as_scalar()._inverse())._as_scalar()
        try:
            exact = cosine.acos_degrees()
        except ValueError:
            exact = None
        if exact is not None:
            return exact
        cos_value = max(fractions.Fraction(-1), min(fractions.Fraction(1), cosine.to_fractions_approx(_FALLBACK_PREC)))
        return fractions.Fraction(exactmath.acos(cos_value, degrees=True, prec=_FALLBACK_PREC, limit=False))

    @property
    def volume(self) -> SurdScalar:
        """The cell volume, the exact absolute determinant of ``matrix``."""
        if self._volume_cache is None:
            det = self.matrix.det()
            self._volume_cache = (-det)._as_scalar() if det.sign() < 0 else det
        return self._volume_cache

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cell):
            return NotImplemented
        return self.matrix == other.matrix

    def __repr__(self) -> str:
        return f"Cell(matrix={self.matrix!r}, scale={self._scale!r})"
