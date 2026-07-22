"""
Backend wrapping a raw 3x3 cell-vector matrix.
"""

from typing import Any

from .cell_backend import CellBackend


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_3x3(matrix: Any) -> bool:
    if not isinstance(matrix, (list, tuple)) or len(matrix) != 3:
        return False
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            return False
        if not all(_is_number(x) for x in row):
            return False
    return True


class CellPrimitive(CellBackend):
    """
    Backend for a cell backed by a raw 3x3 list or tuple of numbers.

    The native representation is a 3x3 nested list or tuple of cell vectors (one vector
    per row). The ``matrix`` is derived lazily and cached, and ``unwrap`` returns the
    original raw object.
    """

    _raw: Any
    _matrix_cache: tuple[tuple[float, ...], ...] | None

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "primitive") != "primitive":
            return None
        if not _is_3x3(obj):
            return None
        return super().__new__(cls)

    def __init__(self, obj: Any, **hints: Any) -> None:
        self._raw = obj
        self._matrix_cache = None

    @property
    def matrix(self) -> tuple[tuple[float, ...], ...]:
        if self._matrix_cache is None:
            self._matrix_cache = tuple(tuple(float(x) for x in row) for row in self._raw)
        return self._matrix_cache

    def unwrap(self) -> Any:
        return self._raw
