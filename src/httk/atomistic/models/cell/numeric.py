"""
The NumericCell value: a cell that owns its geometry as plain floats.
"""

import math
from collections.abc import Sequence
from typing import Any

from httk.core import NumericVector

from httk.atomistic.models._vector_guards import require_numpy, to_periodicity
from httk.atomistic.models.cell.api import CellAPI

type FloatRows = tuple[tuple[float, ...], ...]


def finite_float(value: Any, what: str) -> float:
    """Return ``value`` as a finite float with ``-0.0`` normalised to ``0.0``.

    :param value: The real number to convert.
    :param what: What the value is, for the error message.
    :return: The canonical float.
    :raises ValueError: If the value is NaN or infinite.
    """
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{what} must be finite, got {result!r}")
    return result + 0.0  # -0.0 + 0.0 is 0.0, so equal values keep equal content identities


def float_rows(values: Any, what: str) -> FloatRows:
    """Normalise an N x 3 real array-like into canonical rows of finite floats.

    :param values: The array-like to convert.
    :param what: What the rows are, for the error message.
    :return: The rows as nested tuples of floats.
    :raises ValueError: If the input is not N x 3 or holds a non-finite value.
    """
    import numpy

    array = numpy.asarray(values, dtype=float)
    if array.shape == (0,):  # no rows at all; an (N, 0) input is a wrong width and must raise
        array = array.reshape(0, 3)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{what} must be an Nx3 array-like, got shape {array.shape}")
    return tuple(tuple(finite_float(x, what) for x in row) for row in array.tolist())


def readonly_array(rows: Sequence[Sequence[float]] | Sequence[float]) -> NumericVector:
    """Return ``rows`` as a fresh read-only ``float64`` ndarray.

    :param rows: The values to present.
    :return: The read-only array.
    """
    import numpy

    array = numpy.array(rows, dtype=float)
    array.flags.writeable = False
    return array


def float_precision(value: Any) -> float | None:
    """Return a stated precision as a strictly positive finite float, or ``None``.

    :param value: The precision, or ``None`` when unknown.
    :return: The precision as a float.
    :raises ValueError: If the precision is non-finite or not strictly positive.
    """
    if value is None:
        return None
    result = finite_float(value, "precision")
    if result <= 0:
        raise ValueError(f"a precision must be strictly positive, got {result}; use None for unknown")
    return result


class NumericCell:
    """
    A cell that owns its basis as plain floats, an immutable value.

    Where a :class:`~httk.atomistic.Cell` holds its geometry **exactly**, a ``NumericCell`` holds
    the 3x3 lattice vectors (one per row) as Python floats and returns plain numpy numbers: a
    read-only ``float64`` :class:`numpy.ndarray` for every vector and a :class:`float` for every
    scalar. Constructing one from a ``Cell`` (or any cell backend) is the explicit lossy step:
    ``NumericCell(cell)`` keeps its float basis, precision and periodicity.

    Values are canonical: ``-0.0`` becomes ``0.0`` and non-finite numbers are rejected, so equal
    values have equal content. Construction **requires numpy** (the ``httk-atomistic[numpy]``
    extra) and raises :class:`ImportError` eagerly when it is unavailable.

    :param basis: The three cell vectors as a 3x3 real array-like, or a cell to convert.
    :param precision: The absolute basis precision, if known (default taken from a converted cell).
    :param periodicity: Flags identifying which basis rows are lattice translations (default all,
        or taken from a converted cell).
    :raises ValueError: If the basis is not 3x3, holds a non-finite value or is degenerate.
    """

    _basis: FloatRows
    _precision: float | None
    _periodicity: tuple[bool, bool, bool]

    def __init__(self, basis: Any, *, precision: Any = None, periodicity: Any = None) -> None:
        require_numpy()
        import numpy

        if isinstance(basis, NumericCell):
            precision = basis.precision if precision is None else precision
            periodicity = basis.periodicity if periodicity is None else periodicity
            basis = basis._basis
        elif isinstance(basis, CellAPI):
            precision = basis.precision if precision is None else precision
            periodicity = basis.periodicity if periodicity is None else periodicity
            basis = basis.basis_floats()
        rows = float_rows(basis, "NumericCell basis")
        if len(rows) != 3:
            raise ValueError("NumericCell basis must be a 3x3 array-like")
        if numpy.linalg.matrix_rank(numpy.asarray(rows)) < 3:
            raise ValueError("NumericCell basis must be non-degenerate (its three vectors must span three dimensions)")
        self._basis = rows
        self._precision = float_precision(precision)
        self._periodicity = to_periodicity(periodicity)

    @property
    def basis(self) -> NumericVector:
        """The 3x3 lattice vectors.

        :return: The lattice vectors as a read-only float array.
        """
        return readonly_array(self._basis)

    @property
    def precision(self) -> float | None:
        """The absolute basis precision, or ``None`` if unknown.

        :return: The precision as a floating-point value.
        """
        return self._precision

    @property
    def periodicity(self) -> tuple[bool, bool, bool]:
        """Which of the three basis rows is a genuine lattice translation.

        :return: Flags identifying the periodic basis rows.
        """
        return self._periodicity

    @property
    def nperiodic_dimensions(self) -> int:
        """How many of the three directions are periodic.

        :return: The number of periodic directions.
        """
        return sum(self._periodicity)

    @property
    def lengths(self) -> NumericVector:
        """The three cell-vector lengths.

        :return: The lengths as a read-only float array.
        """
        import numpy

        return readonly_array(numpy.linalg.norm(numpy.asarray(self._basis), axis=1).tolist())

    @property
    def angles(self) -> NumericVector:
        """The cell angles ``(alpha, beta, gamma)`` in degrees.

        :return: The angles as a read-only float array.
        """
        import numpy

        basis = numpy.asarray(self._basis)
        lengths = numpy.linalg.norm(basis, axis=1)

        def angle(i: int, j: int) -> float:
            cosine = float(basis[i] @ basis[j]) / float(lengths[i] * lengths[j])
            return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

        return readonly_array([angle(1, 2), angle(0, 2), angle(0, 1)])

    @property
    def volume(self) -> float:
        """The cell volume, the absolute determinant of the basis.

        :return: The volume as a floating-point value.
        :raises ValueError: If the cell is not periodic in all three directions.
        """
        import numpy

        if self._periodicity != (True, True, True):
            raise ValueError(
                "volume is defined only for a fully 3D-periodic cell; this one is periodic in "
                f"{self.nperiodic_dimensions} of 3 directions ({self._periodicity})."
            )
        return abs(float(numpy.linalg.det(numpy.asarray(self._basis))))

    def metric(self) -> NumericVector:
        """The Gram matrix ``basis * basis^T``.

        :return: The Gram matrix as a read-only float array.
        """
        import numpy

        basis = numpy.asarray(self._basis)
        return readonly_array((basis @ basis.T).tolist())

    def _key(self) -> tuple[Any, ...]:
        return (self._basis, self._precision, self._periodicity)

    def __eq__(self, other: object) -> bool:
        """Equality of the basis, precision and periodicity.

        :param other: The object to compare with.
        :return: Whether the float values match.
        """
        if not isinstance(other, NumericCell):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        parts = [repr([list(row) for row in self._basis])]
        if self._precision is not None:
            parts.append(f"precision={self._precision!r}")
        if self._periodicity != (True, True, True):
            parts.append(f"periodicity={self._periodicity!r}")
        return f"NumericCell({', '.join(parts)})"
