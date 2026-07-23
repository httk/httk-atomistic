"""
A view presenting any cell backend as a raw 3-row tuple of cell vectors.
"""

from typing import Any, Self

from httk.core import unwrap

from ._vector_guards import to_float_tuples
from .cell_backend import CellBackend
from .cell_like import CellLike
from .cell_view import CellView


class CellPrimitiveView(CellView, tuple):
    """
    A view presenting an underlying cell backend as the raw 3x3 basis matrix of floats.

    This view is a genuine tuple of three cell-vector rows (the scaled lattice vectors rendered to
    floats from the exact ``basis``), built eagerly and immutable.
    """

    _backend: CellBackend

    def __new__(cls, obj: CellLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls, to_float_tuples(backend.basis))
        instance._backend = backend
        return instance

    def __init__(self, obj: CellLike, **hints: Any) -> None:
        super().__init__()

    def unwrap(self) -> Any:
        return unwrap(self._backend)
