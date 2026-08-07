"""
The NumericCell presentation: a Cell exposed as plain numpy numbers.
"""

from httk.core import NumericVector, to_numeric, to_numeric_scalar

from httk.atomistic.models._vector_guards import require_numpy
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.cell.view import CellView


class NumericCell:
    """
    A plain-numpy presentation of a :class:`~httk.atomistic.Cell`.

    Where a ``Cell`` holds its geometry **exactly** (a :class:`~httk.core.SurdVector` basis, exact
    ``SurdScalar`` lengths/volume, exact ``Fraction`` angles), a ``NumericCell`` mirrors that
    interface but returns plain numpy numbers — a ``float64`` :class:`numpy.ndarray` for every vector
    and a plain :class:`float` for every scalar — for callers who do not need exact arithmetic and
    just want numpy arrays.

    The presentation is numpy-backed, so constructing a ``NumericCell`` **requires numpy** (the
    ``httk-atomistic[numpy]`` extra) and raises :class:`ImportError` eagerly when it is unavailable.
    The exact object is always one hop away via :attr:`exact`.

    :param cell: The cell or cell-like object to present.
    """

    _cell: Cell

    def __init__(self, cell: CellLike) -> None:
        require_numpy()
        self._cell = cell if isinstance(cell, Cell) else CellView(cell)

    def _vector(self, values: tuple[object, ...]) -> NumericVector:
        """Present a tuple of exact scalars as a plain ``float64`` numpy vector."""
        return to_numeric(tuple(to_numeric_scalar(value) for value in values))

    @property
    def scale(self) -> float:
        """The overall length factor.

        :return: The scale as a floating-point value.
        """
        return to_numeric_scalar(self._cell.scale)

    @property
    def precision(self) -> float | None:
        """The cell precision, or ``None`` if unknown.

        :return: The absolute precision as a floating-point value.
        """
        return None if self._cell.precision is None else float(self._cell.precision)

    @property
    def periodicity(self) -> tuple[bool, bool, bool]:
        """Which of the three basis rows is a genuine lattice translation.

        :return: Flags identifying the periodic basis rows.
        """
        return self._cell.periodicity

    @property
    def nperiodic_dimensions(self) -> int:
        """How many of the three directions are periodic.

        :return: The number of periodic directions.
        """
        return self._cell.nperiodic_dimensions

    @property
    def unscaled_basis(self) -> NumericVector:
        """The 3x3 cell vectors before applying ``scale``.

        :return: The unscaled lattice vectors as floating-point values.
        """
        return to_numeric(self._cell.unscaled_basis)

    @property
    def basis(self) -> NumericVector:
        """The 3x3 lattice vectors ``scale * unscaled_basis``.

        :return: The scaled lattice vectors as floating-point values.
        """
        return to_numeric(self._cell.basis)

    @property
    def lengths(self) -> NumericVector:
        """The three cell-vector lengths.

        :return: The lengths as floating-point values.
        """
        return self._vector(self._cell.lengths)

    @property
    def angles(self) -> NumericVector:
        """The cell angles ``(alpha, beta, gamma)`` in degrees.

        :return: The angles as floating-point values.
        """
        return self._vector(self._cell.angles)

    @property
    def volume(self) -> float:
        """The cell volume.

        :return: The volume as a floating-point value.
        :raises ValueError: If the exact cell is not periodic in all three directions.
        """
        return to_numeric_scalar(self._cell.volume)

    def metric(self) -> NumericVector:
        """The Gram matrix ``basis * basis^T``.

        :return: The Gram matrix as floating-point values.
        """
        return to_numeric(self._cell.metric())

    @property
    def exact(self) -> Cell:
        """The exact cell this presentation wraps.

        :return: The exact cell.
        """
        return self._cell

    def __repr__(self) -> str:
        return f"NumericCell(basis={self.basis!r}, scale={self.scale!r})"
