"""A plain-numpy presentation of a structure backend."""

from functools import cached_property
from typing import Any, Self

from httk.core import NumericVector, to_numeric, unwrap

from ._vector_guards import require_numpy
from .numeric_cell import NumericCell
from .numeric_sites import NumericSites
from .species import Species
from .structure import Structure
from .structure_backend import StructureBackend
from .structure_like import StructureLike
from .structure_view import StructureView


class NumericUnitcellStructureView(StructureView):
    """A plain-numpy presentation of a :class:`~httk.atomistic.Structure`.

    Where a ``Structure`` holds its geometry exactly (a surd ``cell`` basis, rational reduced
    coordinates, and an exact Cartesian frame), this view mirrors that interface but returns
    plain numpy numbers: its :attr:`cell` is a :class:`~httk.atomistic.NumericCell`, its
    :attr:`sites` a :class:`~httk.atomistic.NumericSites`, and :meth:`cartesian_sites` a
    ``float64`` numpy array. The ``species``/``species_at_sites`` are passed through unchanged.
    It is for callers who do not need exact arithmetic and just want numpy arrays.

    The presentation is numpy-backed, so constructing it **requires numpy** (the
    ``httk-atomistic[numpy]`` extra) and raises :class:`ImportError` eagerly when it is unavailable.
    The exact object is always one hop away via :attr:`exact`.

    This is a view, not a ``Structure`` subclass. Its exact ``Structure`` is built lazily on
    first access to exact geometry.
    """

    _backend: StructureBackend

    def __new__(cls, obj: StructureLike, **hints: Any) -> Self:
        require_numpy()
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        instance._backend = backend
        return instance

    def __init__(self, obj: StructureLike, **hints: Any) -> None:
        pass

    @cached_property
    def _exact(self) -> Structure:
        return Structure(self._backend.cell, self._backend.sites, self._backend.species, self._backend.species_at_sites)

    @property
    def cell(self) -> NumericCell:
        """The cell as a :class:`~httk.atomistic.NumericCell`."""
        return NumericCell(self._exact.cell)

    @property
    def sites(self) -> NumericSites:
        """The sites as a :class:`~httk.atomistic.NumericSites`."""
        return NumericSites(self._exact.sites)

    @property
    def species(self) -> tuple[Species, ...]:
        """The distinct species, passed through unchanged."""
        return self._backend.species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """The species name occupying each site, passed through unchanged."""
        return self._backend.species_at_sites

    def cartesian_sites(self) -> NumericVector:
        """The Cartesian site positions as an ``(N, 3)`` ``float64`` numpy array."""
        return to_numeric(self._exact.cartesian_sites())

    @property
    def exact(self) -> Structure:
        """The exact :class:`~httk.atomistic.Structure` this view presents."""
        return self._exact

    def __repr__(self) -> str:
        return (
            f"NumericUnitcellStructureView(cell={self.cell!r}, sites={self.sites!r}, "
            f"species={self.species!r}, species_at_sites={self.species_at_sites!r})"
        )

    def unwrap(self) -> Any:
        """Return the raw object wrapped by the backend."""
        return unwrap(self._backend)
