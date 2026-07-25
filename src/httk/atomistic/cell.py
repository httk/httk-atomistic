"""
The Cell class for httk-atomistic.
"""

import fractions
from typing import TYPE_CHECKING, Any

from httk.core import SurdScalar, SurdVector, VectorLike
from httk.core.vectors import exactmath
from httk.core.vectors.exactmath import integer_sqrt

from ._vector_guards import to_periodicity, to_precision, to_surdscalar, to_surdvector

if TYPE_CHECKING:
    from .numeric_cell import NumericCell

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
    A crystallographic cell: its basis, the 3x3 matrix of cell vectors, held **exactly**.

    The lattice vectors are the rows of ``basis``. Internally a Cell factors that basis into a
    positive :class:`~httk.core.SurdScalar` ``scale`` times an ``unscaled_basis``
    (a :class:`~httk.core.SurdVector` of shape ``(3, 3)``), with ``basis == scale * unscaled_basis``.
    The split lets an overall length factor be carried symbolically: a hexagonal cell of lattice
    parameter ``a`` and ratio ``c/a`` is the exact ``unscaled`` rows ``(1, 0, 0)``,
    ``(-1/2, sqrt(3)/2, 0)``, ``(0, 0, c/a)`` scaled by ``a`` — so the ``sqrt(3)`` stays exact
    regardless of ``a``. A cell built from an absolute basis simply has ``scale == 1``.

    Numbers embed exactly: rationals (and rational-valued floats) stay rational, and a
    :class:`~httk.core.SurdVector` basis keeps its radicals. Derived quantities are exact whenever
    the geometry is metric-rational (the crystallographic case): ``lengths`` come from
    :meth:`~httk.core.SurdVector.sqrt_of` of the rational squared row lengths, ``angles`` (degrees)
    from the exact reverse-Niven :meth:`~httk.core.SurdScalar.acos_degrees` where possible,
    ``volume`` from the exact determinant, and ``metric`` is the exact rational Gram matrix. When a
    squared length happens to be irrational, ``lengths``/``angles`` fall back to a deterministic
    rational approximation (documented per accessor). Exact accessors return vector objects —
    render them with ``.to_floats()`` (nested plain-float lists, numpy-free), ``float(...)`` on
    scalars, :meth:`numeric` (true numpy arrays), or a view of your choice.

    A cell also records its :attr:`periodicity`, which defaults to periodic in all three
    directions. Where it is not, the basis stops being purely a lattice and becomes partly a
    *coordinate frame*: see :attr:`periodicity` for what that means and
    :attr:`periodic_measure` for the quantity that replaces :attr:`volume`.
    """

    _scale: SurdScalar
    _unscaled_basis: SurdVector
    _basis_cache: SurdVector | None
    _metric_cache: SurdVector | None
    _lengths_cache: tuple[SurdScalar, ...] | None
    _angles_cache: tuple[fractions.Fraction, ...] | None
    _volume_cache: SurdScalar | None
    _precision: fractions.Fraction | None
    _periodicity: tuple[bool, bool, bool]

    def __init__(
        self,
        basis: VectorLike,
        scale: Any = 1,
        precision: Any = None,
        periodicity: Any = None,
    ) -> None:
        unscaled = to_surdvector(basis)
        if unscaled.dim != (3, 3):
            raise ValueError("Cell basis must be a 3x3 vector-like")
        scale_scalar = to_surdscalar(scale)
        if scale_scalar.sign() <= 0:
            raise ValueError("Cell scale must be strictly positive")
        if unscaled.det().sign() == 0:
            # A degenerate basis is not a cell in any sense: it spans less than three
            # dimensions, so it cannot be inverted and Cartesian coordinates cannot be
            # turned back into fractional ones. Rejecting it here rather than letting
            # `volume` quietly return 0 and `angles` raise a bare ZeroDivisionError far
            # downstream also makes this route agree with CellParams, which has always
            # rejected the same geometry.
            raise ValueError(
                "Cell basis must be non-degenerate (its three vectors must span three "
                "dimensions); a zero or linearly dependent row cannot be a cell, and for a "
                "non-periodic direction it should be a unit vector rather than a zero one"
            )
        self._precision = to_precision(precision)
        self._periodicity = to_periodicity(periodicity)
        self._unscaled_basis = unscaled
        self._scale = scale_scalar
        self._basis_cache = None
        self._metric_cache = None
        self._lengths_cache = None
        self._angles_cache = None
        self._volume_cache = None

    @property
    def scale(self) -> SurdScalar:
        """The overall (strictly positive) length factor, as an exact ``SurdScalar``."""
        return self._scale

    @property
    def unscaled_basis(self) -> SurdVector:
        """The 3x3 cell vectors before applying ``scale``, as an exact ``SurdVector``."""
        return self._unscaled_basis

    @property
    def basis(self) -> SurdVector:
        """The 3x3 lattice vectors ``scale * unscaled_basis`` (one vector per row), exact."""
        if self._basis_cache is None:
            self._basis_cache = self._scale * self._unscaled_basis
        return self._basis_cache

    @property
    def precision(self) -> fractions.Fraction | None:
        """How precisely this basis was stated, as an absolute length, or ``None`` if unknown.

        In the same units as the basis itself, so for ordinary crystallographic data it is
        an ångström. Derived from the source's written digits and any stated uncertainty —
        a CIF cell edge of ``5.6402(3)`` is precise to ``3e-4``, not to the ``1e-4`` its
        four decimals alone would suggest.

        ``None`` means unknown, which is not the same as exact. It is what a cell built by
        hand or from a bare matrix reports.
        """
        return self._precision

    @property
    def periodicity(self) -> tuple[bool, bool, bool]:
        """Which of the three basis rows is a genuine lattice translation.

        ``(True, True, True)`` — the default, and what every ordinary crystal is. A slab is
        ``(True, True, False)``, a nanowire has one ``True``, and an isolated molecule is
        ``(False, False, False)``.

        A row flagged ``False`` is **not** a lattice vector. It is only a frame: it says
        what a fractional coordinate means along that direction, and nothing more.
        Coordinates there are unbounded — freely below 0 or above 1 — and are never wrapped
        into ``[0, 1)``. There is no vacuum and no padding involved, so making that row a
        unit vector simply means the coordinate along it *is* a length in the basis's units.

        This is the same notion, in the same order, as OPTIMADE's ``dimension_types``.
        """
        return self._periodicity

    @property
    def nperiodic_dimensions(self) -> int:
        """How many of the three directions are periodic, from 0 to 3."""
        return sum(self._periodicity)

    def numeric(self) -> "NumericCell":
        """A plain-numpy presentation of this cell (requires the ``httk-atomistic[numpy]`` extra)."""
        from .numeric_cell import NumericCell

        return NumericCell(self)

    def metric(self) -> SurdVector:
        """The exact Gram matrix ``matrix * matrix^T`` (rational for a metric-rational cell)."""
        if self._metric_cache is None:
            m = self.basis
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
        scale-independent, so they are computed from the unscaled basis. The cosine is formed
        exactly in the surd field and reversed through the Niven table
        (:meth:`~httk.core.SurdScalar.acos_degrees`) for an exact answer; a non-Niven angle falls
        back to a deterministic :func:`~httk.core.vectors.exactmath.acos` at ``_FALLBACK_PREC``.
        """
        if self._angles_cache is None:
            u = self._unscaled_basis
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
        """The cell volume, the exact absolute determinant of ``basis``.

        Defined only for a fully periodic cell, and raises :class:`ValueError` otherwise.
        For anything less, the determinant mixes real lattice vectors with frame vectors,
        so it is not a volume: it changes when a frame vector is rescaled, even though
        nothing about the material did. Any density or packing fraction derived from it
        would inherit that. See :attr:`periodic_measure` for the quantity that *is* defined.
        """
        if self._periodicity != (True, True, True):
            raise ValueError(
                "volume is defined only for a fully 3D-periodic cell; this one is periodic in "
                f"{self.nperiodic_dimensions} of 3 directions ({self._periodicity}). "
                "Use periodic_measure for the area, length or volume of the periodic sublattice."
            )
        return self._unchecked_volume

    @property
    def _unchecked_volume(self) -> SurdScalar:
        """``|det(basis)|``, without the periodicity check. Internal."""
        if self._volume_cache is None:
            det = self.basis.det()
            self._volume_cache = (-det)._as_scalar() if det.sign() < 0 else det
        return self._volume_cache

    @property
    def periodic_measure(self) -> SurdScalar:
        """The measure of the periodic sublattice: a volume, an area, a length, or one.

        The size of the repeating unit, whatever its dimension — volume for a crystal, area
        for a slab, length for a nanowire. For a fully non-periodic cell there is no
        repeating unit and this is the empty product, ``1``, which is dimensionless rather
        than a length of any kind.

        Exact in the crystallographic case. The 3D case is the determinant and needs no
        square root at all; the 2D and 1D cases go through the same
        :func:`_scalar_length` that :attr:`lengths` uses, so they are exact whenever the
        squared measure is a rational with a small radicand and fall back to a deterministic
        rational approximation otherwise.
        """
        rows = [index for index, periodic in enumerate(self._periodicity) if periodic]
        if len(rows) == 3:
            return self._unchecked_volume
        if not rows:
            return SurdVector.create(1)._as_scalar()
        if len(rows) == 1:
            return self.lengths[rows[0]]
        # The k-dimensional measure of the sublattice is sqrt(det(G)) over the Gram matrix of
        # its rows -- for k=2 the familiar |a x b|, formed here without leaving the surd field.
        metric = self.metric()
        first, second = rows
        gram_det = (
            metric._element((first, first)) * metric._element((second, second))
            - metric._element((first, second)) * metric._element((second, first))
        )._as_scalar()
        return _scalar_length(gram_det)

    def __eq__(self, other: object) -> bool:
        """Equality of the basis and the periodicity, and of nothing else.

        Neither the ``scale``/``unscaled_basis`` factoring nor the stated ``precision``
        takes part: how the scale was factored out and how precisely the source wrote the
        numbers are both metadata about the cell's provenance, not part of its geometry.

        ``periodicity`` does take part, because it is not provenance — it says which rows
        are lattice vectors at all. A slab and a bulk crystal that happen to share a basis
        are different cells, and one of them has a volume while the other does not.
        """
        if not isinstance(other, Cell):
            return NotImplemented
        return self.basis == other.basis and self._periodicity == other._periodicity

    def __repr__(self) -> str:
        if self._periodicity == (True, True, True):
            return f"Cell(basis={self.basis!r}, scale={self._scale!r})"
        return f"Cell(basis={self.basis!r}, scale={self._scale!r}, periodicity={self._periodicity!r})"
