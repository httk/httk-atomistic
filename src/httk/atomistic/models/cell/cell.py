"""
The Cell class for httk-atomistic.
"""

import fractions
from typing import TYPE_CHECKING, Any

from httk.core import SurdScalar, SurdVector, VectorLike

from httk.atomistic.models._vector_guards import to_periodicity, to_precision, to_surdscalar, to_surdvector
from httk.atomistic.models.cell.api import _scalar_length
from httk.atomistic.models.cell.backend import CellBackend

if TYPE_CHECKING:
    from httk.atomistic.models.cell.numeric import NumericCell


class Cell(CellBackend):
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
    :class:`~httk.core.SurdVector` basis keeps its radicals. Derived quantities retain exact forms
    where the underlying operation stays in the supported exact fields, including the usual
    metric-rational crystallographic case: ``lengths`` use
    :meth:`~httk.core.SurdVector.sqrt_of` for rational squared row lengths when the rational
    radicand is a perfect square or stays below the deterministic small-radicand threshold,
    ``angles`` (degrees) use the exact reverse-Niven :meth:`~httk.core.SurdScalar.acos_degrees`
    where possible, ``volume`` comes from the exact determinant, and ``metric`` is the exact Gram
    matrix, which may itself contain surds. For larger rational or irrational squared lengths,
    ``lengths``/``angles`` fall back to a deterministic rational approximation (documented per
    accessor). Exact accessors return vector objects —
    render them with ``.to_floats()`` (nested plain-float lists, numpy-free), ``float(...)`` on
    scalars, :meth:`numeric` (true numpy arrays), or a view of your choice.

    A cell also records its :attr:`periodicity`, which defaults to periodic in all three
    directions. Where it is not, the basis stops being purely a lattice and becomes partly a
    *coordinate frame*: see :attr:`periodicity` for what that means and
    :attr:`periodic_measure` for the quantity that replaces :attr:`volume`.

    :param basis: The three cell vectors, one per row.
    :param scale: The positive factor separated from ``basis``.
    :param precision: The absolute precision carried from the source, if known.
    :param periodicity: Flags identifying which basis rows are lattice translations.
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
        """The overall (strictly positive) length factor.

        :return: The factor applied to ``unscaled_basis``.
        """
        return self._scale

    @property
    def unscaled_basis(self) -> SurdVector:
        """The 3x3 cell vectors before applying ``scale``.

        :return: The unscaled lattice vectors.
        """
        return self._unscaled_basis

    @property
    def basis(self) -> SurdVector:
        """The 3x3 lattice vectors ``scale * unscaled_basis``.

        :return: The scaled lattice vectors.
        """
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

        :return: The absolute precision, or ``None`` when it is unknown.
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

        :return: Flags identifying the periodic basis rows.
        """
        return self._periodicity

    @property
    def nperiodic_dimensions(self) -> int:
        """How many of the three directions are periodic.

        :return: The number of periodic directions.
        """
        return sum(self._periodicity)

    def numeric(self) -> "NumericCell":
        """Return a plain-numpy presentation of this cell.

        :return: The numpy-backed presentation.
        :raises ImportError: If numpy is unavailable.
        """
        from httk.atomistic.models.cell.numeric import NumericCell

        return NumericCell(self)

    def metric(self) -> SurdVector:
        """Return the exact, potentially surd-valued Gram matrix ``matrix * matrix^T``.

        :return: The Gram matrix of the cell vectors.
        """
        if self._metric_cache is None:
            self._metric_cache = super().metric()
        return self._metric_cache

    @property
    def lengths(self) -> tuple[SurdScalar, ...]:
        """
        The lengths of the three cell vectors (the scaled row norms).

        Exact via :meth:`~httk.core.SurdVector.sqrt_of` when the row's squared length is a
        perfect-square rational or a rational with numerator times denominator at most
        ``10**18``. Larger rational radicands and irrational squared lengths use a deterministic
        rational approximation at ``_FALLBACK_PREC``.

        :return: The three cell-vector lengths.
        """
        if self._lengths_cache is None:
            self._lengths_cache = super().lengths
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
        back to a deterministic :func:`~httk.core.exactmath.acos` at ``_FALLBACK_PREC``.

        :return: ``(alpha, beta, gamma)`` in degrees.
        """
        if self._angles_cache is None:
            self._angles_cache = super().angles
        return self._angles_cache

    @property
    def volume(self) -> SurdScalar:
        """The cell volume, the exact absolute determinant of ``basis``.

        Defined only for a fully periodic cell, and raises :class:`ValueError` otherwise.
        For anything less, the determinant mixes real lattice vectors with frame vectors,
        so it is not a volume: it changes when a frame vector is rescaled, even though
        nothing about the material did. Any density or packing fraction derived from it
        would inherit that. See :attr:`periodic_measure` for the quantity that *is* defined.

        :return: The absolute determinant of the basis.
        :raises ValueError: If the cell is not periodic in all three directions.
        """
        if self._periodicity != (True, True, True):
            return super().volume
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
        """The size of the repeating unit, whatever its dimension.

        A volume for a crystal, an area for a slab, a length for a nanowire. For a fully
        non-periodic cell there is no repeating unit and this is the empty product, ``1``,
        which is dimensionless rather than a length of any kind.

        Exact in the crystallographic case. The 3D case is the determinant and needs no
        square root at all; the 2D and 1D cases take the same square root that
        :attr:`lengths` does, so they are exact whenever the squared measure is a rational
        with a small radicand and fall back to a deterministic rational approximation
        otherwise.

        :return: The measure of the periodic sublattice.
        """
        rows = [index for index, periodic in enumerate(self._periodicity) if periodic]
        if len(rows) == 3:
            return self._unchecked_volume
        if not rows:
            return SurdVector(1)._as_scalar()
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

        :param other: The object to compare with.
        :return: Whether the basis and periodicity match.
        """
        if not isinstance(other, Cell):
            return NotImplemented
        return self.basis == other.basis and self._periodicity == other._periodicity

    def __repr__(self) -> str:
        if self._periodicity == (True, True, True):
            return f"Cell(basis={self.basis!r}, scale={self._scale!r})"
        return f"Cell(basis={self.basis!r}, scale={self._scale!r}, periodicity={self._periodicity!r})"
