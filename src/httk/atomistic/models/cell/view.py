"""
A view presenting any cell backend as a Cell (the class representation).
"""

import fractions
from functools import cached_property
from typing import Any, Self

from httk.core import SurdScalar, SurdVector, unwrap

from httk.atomistic.models._vector_guards import to_periodicity, to_precision, to_surdscalar, to_surdvector
from httk.atomistic.models.cell.backend import CellBackend
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.cell.view_base import CellViewBase


class CellView(CellViewBase, Cell):
    r"""
    A view presenting an underlying cell backend as a ``Cell``.

    This view is a genuine ``Cell``, so it can be passed anywhere a Cell is accepted.
    Its state is built lazily on first access from the backend.

    :param obj: The cell-like object to present.
    :param \**hints: Backend-selection hints.
    """

    _backend: CellBackend

    def __new__(cls, obj: CellLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        instance._backend = backend
        # Derived-value caches are checked by Cell's ordinary properties, so initialize them
        # eagerly without materializing any backend presentation state.
        instance._basis_cache = None
        instance._metric_cache = None
        instance._lengths_cache = None
        instance._angles_cache = None
        instance._volume_cache = None
        return instance

    def __init__(self, obj: CellLike, **hints: Any) -> None:
        pass

    # Validate then assign: failed fills leave no partial presentation state, and fills must not
    # read shadowed attributes or they recurse.
    def _fill_unscaled_basis(self) -> None:
        unscaled = to_surdvector(self._backend.unscaled_basis)
        if unscaled.dim != (3, 3):
            raise ValueError("Cell basis must be a 3x3 vector-like")
        if unscaled.det().sign() == 0:
            raise ValueError(
                "Cell basis must be non-degenerate (its three vectors must span three "
                "dimensions); a zero or linearly dependent row cannot be a cell, and for a "
                "non-periodic direction it should be a unit vector rather than a zero one"
            )
        object.__setattr__(self, "_unscaled_basis", unscaled)

    def _fill_scale(self) -> None:
        scale = to_surdscalar(self._backend.scale)
        if scale.sign() <= 0:
            raise ValueError("Cell scale must be strictly positive")
        object.__setattr__(self, "_scale", scale)

    def _fill_precision(self) -> None:
        object.__setattr__(self, "_precision", to_precision(self._backend.precision))

    def _fill_periodicity(self) -> None:
        object.__setattr__(self, "_periodicity", to_periodicity(self._backend.periodicity))

    @cached_property
    def _unscaled_basis(self) -> SurdVector:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_unscaled_basis()
        return self.__dict__["_unscaled_basis"]

    @cached_property
    def _scale(self) -> SurdScalar:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_scale()
        return self.__dict__["_scale"]

    @cached_property
    def _precision(self) -> fractions.Fraction | None:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_precision()
        return self.__dict__["_precision"]

    @cached_property
    def _periodicity(self) -> tuple[bool, bool, bool]:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_periodicity()
        return self.__dict__["_periodicity"]

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> Cell:
        """Return this presentation as a standalone cell.

        :return: The exact cell representation.
        """
        # The folded design makes a genuine Cell backend exactly the presented value: reuse it.
        backend = self._backend
        if type(backend) is Cell:
            return backend
        return Cell(self._unscaled_basis, self._scale, self._precision, self._periodicity)
