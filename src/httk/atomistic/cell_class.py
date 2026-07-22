"""
Backend wrapping a Cell in the class representation.
"""

from typing import Any

from .cell import Cell
from .cell_backend import CellBackend


class CellClass(CellBackend):
    """
    Backend for a cell backed by an actual ``Cell`` object.

    Its ``matrix`` accessor delegates to the wrapped Cell, and ``unwrap`` returns that
    Cell.
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
    def matrix(self) -> tuple[tuple[float, ...], ...]:
        return self._cell.matrix

    def unwrap(self) -> Any:
        return self._cell
