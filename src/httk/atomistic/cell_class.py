"""
Backend wrapping a Cell in the class representation.
"""

from typing import Any

from httk.core import SurdScalar, SurdVector

from .cell import Cell
from .cell_backend import CellBackend


class CellClass(CellBackend):
    """
    Backend for a cell backed by an actual ``Cell`` object.

    Its exact accessors delegate to the wrapped Cell (preserving its ``scale``/``unscaled_basis``
    split), and ``unwrap`` returns that Cell.
    """

    _cell: Cell

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if not isinstance(obj, Cell):
            return None
        if hints and hints.get("kind", "class") != "class":
            return None
        return super().__new__(cls)

    def __init__(self, obj: Cell, **hints: Any) -> None:
        self._cell = obj

    @property
    def basis(self) -> SurdVector:
        return self._cell.basis

    @property
    def scale(self) -> SurdScalar:
        return self._cell.scale

    @property
    def unscaled_basis(self) -> SurdVector:
        return self._cell.unscaled_basis

    def unwrap(self) -> Any:
        return self._cell
