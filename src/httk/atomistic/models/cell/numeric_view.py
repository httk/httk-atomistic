"""
A view presenting any cell backend as a NumericCell (the plain-numpy presentation).
"""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models._vector_guards import require_numpy
from httk.atomistic.models.cell.backend import CellBackend
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.cell.numeric import NumericCell
from httk.atomistic.models.cell.view_base import CellViewBase


class CellNumericView(CellViewBase, NumericCell):
    r"""
    A view presenting an underlying cell backend as a ``NumericCell``.

    This view is a genuine ``NumericCell``, so it can be passed anywhere one is accepted. Its exact
    ``Cell`` is built lazily from the backend on first access, preserving the scale/unscaled split.
    Like a ``NumericCell`` it requires numpy (raising :class:`ImportError` otherwise).

    :param obj: The cell-like object to present.
    :param \**hints: Backend-selection hints.
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
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> NumericCell:
        """Return this presentation as a standalone numeric cell.

        :return: The plain-numpy presentation.
        """
        # A genuine NumericCell backend is exactly the presented value: reuse it.
        backend = self._backend
        if type(backend) is NumericCell:
            return backend
        return NumericCell(self._cell)
