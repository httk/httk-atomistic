"""Lazy anonymous-structure presentation view."""

from functools import cached_property
from typing import TYPE_CHECKING, Any, Self

from httk.core import unwrap

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.prototype.anonymous import AnonymousStructure
from httk.atomistic.models.prototype.backend import AnonymousStructureBackend
from httk.atomistic.models.prototype.view_base import AnonymousStructureViewBase
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species

if TYPE_CHECKING:
    from httk.atomistic.models.prototype.like import AnonymousStructureLike
    from httk.atomistic.models.structure.like import StructureLike


class AnonymousStructureView(AnonymousStructureViewBase, AnonymousStructure):
    """A lazy view presenting an anonymous structure or ordinary structure."""

    _backend: AnonymousStructureBackend

    def __new__(cls, obj: "AnonymousStructureLike | StructureLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        instance._backend = backend
        return instance

    def __init__(self, obj: Any, **hints: Any) -> None:
        pass

    def _fill_cell(self) -> None:
        object.__setattr__(self, "_cell", self._backend.cell)

    def _fill_sites(self) -> None:
        object.__setattr__(self, "_sites", self._backend.sites)

    def _fill_species(self) -> None:
        object.__setattr__(self, "_species", self._backend.species)

    def _fill_species_at_sites(self) -> None:
        object.__setattr__(self, "_species_at_sites", self._backend.species_at_sites)

    @cached_property
    def _cell(self) -> Cell:  # type: ignore[override]
        self._fill_cell()
        return self.__dict__["_cell"]

    @cached_property
    def _sites(self) -> Sites:  # type: ignore[override]
        self._fill_sites()
        return self.__dict__["_sites"]

    @cached_property
    def _species(self) -> tuple[Species, ...]:  # type: ignore[override]
        self._fill_species()
        return self.__dict__["_species"]

    @cached_property
    def _species_at_sites(self) -> tuple[str, ...]:  # type: ignore[override]
        self._fill_species_at_sites()
        return self.__dict__["_species_at_sites"]

    @property
    def periodicity(self):
        return self.cell.periodicity

    @property
    def nperiodic_dimensions(self):
        return self.cell.nperiodic_dimensions

    @property
    def nsites(self):
        return len(self.sites)

    def cartesian_sites(self):
        from httk.core import SurdVector

        return SurdVector.create(self.sites.reduced_coords) * self.cell.basis

    @property
    def coordinate_precision(self):
        return self.sites.precision

    @property
    def basis_precision(self):
        return self.cell.precision

    def unwrap(self) -> Any:
        return unwrap(self._backend)

    def unview(self) -> AnonymousStructure:
        if type(self._backend) is AnonymousStructure:
            return self._backend
        return AnonymousStructure(self.cell, self.sites, self.species, self.species_at_sites)
