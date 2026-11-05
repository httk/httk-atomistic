"""
A view presenting any cell backend as a NumericCell (the plain-numpy presentation).
"""

from functools import cached_property
from typing import Any, Self

from httk.core import NumericVector, unwrap

from httk.atomistic.models._vector_guards import require_numpy
from httk.atomistic.models.cell.backend import CellBackend
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.cell.numeric import FloatRows, NumericCell, float_precision, float_rows, readonly_array
from httk.atomistic.models.cell.view import CellView
from httk.atomistic.models.cell.view_base import CellViewBase


class CellNumericView(CellViewBase, NumericCell):
    r"""
    A view presenting an underlying cell backend as a ``NumericCell``.

    This view is a genuine ``NumericCell``, so it can be passed anywhere one is accepted, but it is
    a lossless presentation: the exact cell stays one hop away via :attr:`exact`, and the float
    basis, precision and periodicity are computed lazily from the backend on first access.
    :meth:`unview` returns the standalone ``NumericCell`` value. Like a ``NumericCell`` it requires
    numpy (raising :class:`ImportError` otherwise).

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

    @cached_property
    def exact(self) -> Cell:
        """The exact cell this view presents.

        :return: The backend itself when it is a ``Cell``, else a lazy ``CellView`` of it.
        """
        backend = self._backend
        return backend if isinstance(backend, Cell) else CellView(backend)

    @cached_property
    def _basis(self) -> FloatRows:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        return float_rows(self.exact.basis_floats(), "NumericCell basis")

    @cached_property
    def _precision(self) -> float | None:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        return float_precision(self.exact.precision)

    @cached_property
    def _periodicity(self) -> tuple[bool, bool, bool]:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.exact.periodicity

    # Derived geometry renders the exact values, so a presentation of 120 degrees reads 120.0.
    @property
    def lengths(self) -> NumericVector:
        """The three cell-vector lengths, rendered from the exact cell.

        :return: The lengths as a read-only float array.
        """
        return readonly_array([float(length) for length in self.exact.lengths])

    @property
    def angles(self) -> NumericVector:
        """The cell angles ``(alpha, beta, gamma)`` in degrees, rendered from the exact cell.

        :return: The angles as a read-only float array.
        """
        return readonly_array([float(angle) for angle in self.exact.angles])

    @property
    def volume(self) -> float:
        """The cell volume, rendered from the exact cell.

        :return: The volume as a floating-point value.
        :raises ValueError: If the cell is not periodic in all three directions.
        """
        return float(self.exact.volume)

    def metric(self) -> NumericVector:
        """The Gram matrix ``basis * basis^T``, rendered from the exact cell.

        :return: The Gram matrix as a read-only float array.
        """
        return readonly_array(self.exact.metric().to_floats())

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> NumericCell:
        """Return this presentation as a standalone numeric cell value.

        :return: The float-owning cell.
        """
        return NumericCell(self)
