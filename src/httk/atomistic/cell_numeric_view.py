"""
A view presenting any cell backend as a NumericCell (the plain-numpy presentation).
"""

from typing import Any, Self

from httk.core import unwrap

from .cell import Cell
from .cell_backend import CellBackend
from .cell_like import CellLike
from .cell_view import CellView
from .numeric_cell import NumericCell


class CellNumericView(CellView, NumericCell):
    """
    A view presenting an underlying cell backend as a ``NumericCell``.

    This view is a genuine ``NumericCell``, so it can be passed anywhere one is accepted. Its exact
    ``Cell`` is built eagerly from the backend on construction, preserving the scale/unscaled split.
    Like a ``NumericCell`` it requires numpy (raising :class:`ImportError` otherwise).
    """

    _backend: CellBackend

    def __new__(cls, obj: CellLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        # NumericCell is mutable, so its state is initialized here in __new__ (keeping __init__ a
        # no-op), so that rewrapping an existing view via cls(view) does not re-initialize it. The
        # exact Cell is rebuilt from the backend's scale/unscaled_basis to preserve the factoring.
        NumericCell.__init__(instance, Cell(backend.unscaled_basis, backend.scale))
        instance._backend = backend
        return instance

    def __init__(self, obj: CellLike, **hints: Any) -> None:
        pass

    def unwrap(self) -> Any:
        return unwrap(self._backend)
