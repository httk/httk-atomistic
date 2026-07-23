"""
The NumericCell presentation: a Cell exposed as plain numpy numbers.
"""

from httk.core import NumericVector, to_numeric, to_numeric_scalar

from ._vector_guards import require_numpy
from .cell import Cell
from .cell_class_view import CellClassView
from .cell_like import CellLike


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
    """

    _cell: Cell

    def __init__(self, cell: CellLike) -> None:
        require_numpy()
        self._cell = cell if isinstance(cell, Cell) else CellClassView(cell)

    def _vector(self, values: tuple[object, ...]) -> NumericVector:
        """Present a tuple of exact scalars as a plain ``float64`` numpy vector."""
        return to_numeric(tuple(to_numeric_scalar(value) for value in values))

    @property
    def scale(self) -> float:
        """The overall length factor as a plain ``float``."""
        return to_numeric_scalar(self._cell.scale)

    @property
    def unscaled_basis(self) -> NumericVector:
        """The 3x3 cell vectors before applying ``scale`` as a ``float64`` numpy array."""
        return to_numeric(self._cell.unscaled_basis)

    @property
    def basis(self) -> NumericVector:
        """The 3x3 lattice vectors ``scale * unscaled_basis`` as a ``float64`` numpy array."""
        return to_numeric(self._cell.basis)

    @property
    def lengths(self) -> NumericVector:
        """The three cell-vector lengths as a ``(3,)`` ``float64`` numpy array."""
        return self._vector(self._cell.lengths)

    @property
    def angles(self) -> NumericVector:
        """The cell angles ``(alpha, beta, gamma)`` in degrees as a ``(3,)`` ``float64`` numpy array."""
        return self._vector(self._cell.angles)

    @property
    def volume(self) -> float:
        """The cell volume as a plain ``float``."""
        return to_numeric_scalar(self._cell.volume)

    def metric(self) -> NumericVector:
        """The Gram matrix ``basis * basis^T`` as a ``float64`` numpy array."""
        return to_numeric(self._cell.metric())

    @property
    def exact(self) -> Cell:
        """The exact :class:`~httk.atomistic.Cell` this presentation wraps."""
        return self._cell

    def __repr__(self) -> str:
        return f"NumericCell(basis={self.basis!r}, scale={self.scale!r})"
