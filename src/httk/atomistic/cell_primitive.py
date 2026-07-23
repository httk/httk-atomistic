"""
Backend wrapping a raw 3x3 cell-vector matrix.
"""

from typing import Any

from httk.core import SurdScalar, SurdVector

from ._vector_guards import is_matrix_3x3, to_surdvector
from .cell_backend import CellBackend


class CellPrimitive(CellBackend):
    """
    Backend for a cell backed by a raw 3x3 list or tuple of numbers (or any 3x3 vector-like).

    The native representation is preserved verbatim (one cell vector per row); the exact
    :class:`~httk.core.SurdVector` ``matrix`` is built lazily and cached. This representation carries
    no separate length factor, so ``scale`` is the exact ``1`` and ``unscaled_matrix == matrix``.
    ``unwrap`` returns the original raw object.
    """

    _raw: Any
    _matrix_cache: SurdVector | None

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "primitive") != "primitive":
            return None
        if not is_matrix_3x3(obj):
            return None
        return super().__new__(cls)

    def __init__(self, obj: Any, **hints: Any) -> None:
        self._raw = obj
        self._matrix_cache = None

    @property
    def matrix(self) -> SurdVector:
        if self._matrix_cache is None:
            self._matrix_cache = to_surdvector(self._raw)
        return self._matrix_cache

    @property
    def scale(self) -> SurdScalar:
        return SurdVector.one()

    @property
    def unscaled_matrix(self) -> SurdVector:
        return self.matrix

    def unwrap(self) -> Any:
        return self._raw
