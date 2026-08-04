"""Crystal-axis per-site magnetic-moment backend and exact frame conversions."""

import fractions
from typing import Any

from httk.core import SurdVector, VectorLike

from httk.atomistic.models._vector_guards import to_precision, to_surdvector
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.cell.view import CellView
from httk.atomistic.models.moments.backend import SiteMomentsBackend


def _unit_axes(cell: Cell) -> SurdVector:
    """Return the exact matrix whose rows are the cell's unit lattice axes."""
    basis = cell.basis
    lengths = cell.lengths
    rows = [[(basis._element((row, column)) / lengths[row])._as_scalar() for column in range(3)] for row in range(3)]
    return SurdVector._from_scalar_grid(rows, (3, 3))


def _cartesian_to_crystalaxis(moments: SurdVector, cell: Cell) -> SurdVector:
    """Convert Cartesian rows to crystal-axis rows as ``moments * B^-1 * diag(lengths)``."""
    scaled = moments * cell.basis.inv()
    rows = [
        [(scaled._element((row, column)) * cell.lengths[column])._as_scalar() for column in range(3)]
        for row in range(scaled.dim[0])
    ]
    return SurdVector._from_scalar_grid(rows, scaled.dim)


class CrystalAxisSiteMoments(SiteMomentsBackend):
    """Moments along the unit lattice axes ``â``, ``b̂``, ``ĉ``, in Bohr magnetons."""

    kind = "crystalaxis"
    _crystalaxis_moments: SurdVector
    _cell: Cell

    def __init__(self, moments: VectorLike, cell: CellLike, precision: Any = None) -> None:
        value = to_surdvector(moments)
        if len(value.dim) != 2 or value.dim[1] != 3:
            raise ValueError("CrystalAxisSiteMoments moments must be an Nx3 vector-like")
        self._crystalaxis_moments = value
        self._cell = cell if isinstance(cell, Cell) else CellView(cell)
        self._precision = to_precision(precision)

    @property
    def crystalaxis_moments(self) -> SurdVector:
        """The exact Nx3 moments along the cell's unit lattice axes."""
        return self._crystalaxis_moments

    @property
    def cell(self) -> Cell:
        """The cell defining the crystal-axis frame."""
        return self._cell

    @property
    def cartesian_moments(self) -> SurdVector:
        """The exact Cartesian moments, using rows ``moments * U``."""
        return self._crystalaxis_moments * _unit_axes(self._cell)

    @property
    def precision(self) -> fractions.Fraction | None:
        return self._precision

    def __len__(self) -> int:
        return self._crystalaxis_moments.dim[0]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CrystalAxisSiteMoments):
            return NotImplemented
        return self._crystalaxis_moments == other._crystalaxis_moments and self._cell == other._cell

    def __repr__(self) -> str:
        return f"CrystalAxisSiteMoments(moments={self._crystalaxis_moments!r}, cell={self._cell!r})"
