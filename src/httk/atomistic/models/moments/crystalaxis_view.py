"""Crystal-axis class view for site moments."""

import fractions
from functools import cached_property
from typing import Any, Self

from httk.core import SurdVector

from httk.atomistic.models._vector_guards import to_precision, to_surdvector
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.cell.view import CellView
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.crystalaxis import (
    CrystalAxisSiteMoments,
    _cartesian_to_crystalaxis,
)
from httk.atomistic.models.moments.like import SiteMomentsLike
from httk.atomistic.models.moments.view_base import SiteMomentsViewBase


class CrystalAxisSiteMomentsView(SiteMomentsViewBase, CrystalAxisSiteMoments):
    """A lazy crystal-axis site-moments presentation with an eagerly checked frame hint."""

    _backend: SiteMomentsBackend
    _cell_hint: Cell | None

    def __new__(cls, obj: SiteMomentsLike, *, cell: CellLike | None = None, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        cell_hint = None if cell is None else (cell if isinstance(cell, Cell) else CellView(cell))
        if isinstance(backend, CrystalAxisSiteMoments):
            if cell_hint is not None and cell_hint != backend.cell:
                raise ValueError("CrystalAxisSiteMomentsView cell hint conflicts with the backend cell")
        elif isinstance(backend, CartesianSiteMoments) and cell_hint is None:
            raise ValueError("CrystalAxisSiteMomentsView requires a cell hint for Cartesian moments")
        instance = super().__new__(cls)
        instance._backend = backend
        instance._cell_hint = cell_hint
        return instance

    def __init__(self, obj: SiteMomentsLike, *, cell: CellLike | None = None, **hints: Any) -> None:
        pass

    def _fill_cell(self) -> None:
        if self._cell_hint is not None:
            value = self._cell_hint
        elif isinstance(self._backend, CrystalAxisSiteMoments):
            value = self._backend.cell
        else:
            raise ValueError("CrystalAxisSiteMomentsView needs a cell to define the crystal-axis frame")
        object.__setattr__(self, "_cell", value)

    def _fill_crystalaxis_moments(self) -> None:
        if isinstance(self._backend, CrystalAxisSiteMoments):
            moments = to_surdvector(self._backend.crystalaxis_moments)
        else:
            # Access the canonical value before the cell: collinear backends must fail at fill
            # because a scalar moment has no axis to invert into.
            cartesian = to_surdvector(self._backend.cartesian_moments)
            moments = _cartesian_to_crystalaxis(cartesian, self.cell)
        if len(moments.dim) != 2 or moments.dim[1] != 3:
            raise ValueError("CrystalAxisSiteMoments moments must be an Nx3 vector-like")
        object.__setattr__(self, "_crystalaxis_moments", moments)

    def _fill_precision(self) -> None:
        object.__setattr__(self, "_precision", to_precision(self._backend.precision))

    @cached_property
    def _cell(self) -> Cell:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_cell()
        return self.__dict__["_cell"]

    @cached_property
    def _crystalaxis_moments(self) -> SurdVector:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_crystalaxis_moments()
        return self.__dict__["_crystalaxis_moments"]

    @cached_property
    def _precision(self) -> fractions.Fraction | None:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_precision()
        return self.__dict__["_precision"]
