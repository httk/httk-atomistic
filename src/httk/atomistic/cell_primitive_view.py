"""
A view presenting any cell backend as a raw 3-row tuple of cell vectors.
"""

from typing import Any, Self

from httk.core import unwrap

from .cell_backend import CellBackend
from .cell_like import CellLike
from .cell_view import CellView


class CellPrimitiveView(CellView, tuple):
    """
    A view presenting an underlying cell backend as a raw 3x3 matrix.

    This view is a genuine tuple of three cell-vector rows, built eagerly and immutable.
    """

    _backend: CellBackend

    def __new__(cls, obj: CellLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls, backend.matrix)
        instance._backend = backend
        return instance

    def __init__(self, obj: CellLike, **hints: Any) -> None:
        super().__init__()

    def unwrap(self) -> Any:
        return unwrap(self._backend)
