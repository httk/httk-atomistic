"""
A view presenting any cell backend as a NumericCell (the plain-numpy presentation).
"""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from ._vector_guards import require_numpy
from .cell import Cell
from .cell_backend import CellBackend
from .cell_like import CellLike
from .cell_view_base import CellViewBase
from .numeric_cell import NumericCell


class CellNumericView(CellViewBase, NumericCell):
    """
    A view presenting an underlying cell backend as a ``NumericCell``.

    This view is a genuine ``NumericCell``, so it can be passed anywhere one is accepted. Its exact
    ``Cell`` is built lazily from the backend on first access, preserving the scale/unscaled split.
    Like a ``NumericCell`` it requires numpy (raising :class:`ImportError` otherwise).
    """

    _backend: CellBackend

    def __new__(cls, obj: CellLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        require_numpy()
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        instance._backend = backend
        return instance

    def __init__(self, obj: CellLike, **hints: Any) -> None:
        pass

    # Build then assign so a failed exact presentation leaves the shadow unmaterialized.
    def _fill_cell(self) -> None:
        NumericCell.__init__(
            self,
            Cell(
                self._backend.unscaled_basis,
                self._backend.scale,
                self._backend.precision,
                self._backend.periodicity,
            ),
        )

    @cached_property
    def _cell(self) -> Cell:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_cell()
        return self.__dict__["_cell"]

    def unwrap(self) -> Any:
        return unwrap(self._backend)
