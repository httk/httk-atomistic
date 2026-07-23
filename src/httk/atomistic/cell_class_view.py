"""
A view presenting any cell backend as a Cell (the class representation).
"""

from typing import Any, Self

from httk.core import unwrap

from .cell import Cell
from .cell_backend import CellBackend
from .cell_like import CellLike
from .cell_view import CellView


class CellClassView(CellView, Cell):
    """
    A view presenting an underlying cell backend as a ``Cell``.

    This view is a genuine ``Cell``, so it can be passed anywhere a Cell is accepted.
    Its matrix is built eagerly from the backend on construction.
    """

    _backend: CellBackend

    def __new__(cls, obj: CellLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        # Cell is mutable, so its state is initialized here in __new__ (keeping __init__ a no-op),
        # so that rewrapping an existing view via cls(view) does not re-initialize it. The
        # scale/unscaled split is preserved so a scaled cell stays exactly factored.
        Cell.__init__(instance, backend.unscaled_matrix, backend.scale)
        instance._backend = backend
        return instance

    def __init__(self, obj: CellLike, **hints: Any) -> None:
        pass

    def unwrap(self) -> Any:
        return unwrap(self._backend)
