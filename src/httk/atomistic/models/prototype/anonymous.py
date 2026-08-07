"""The dummy-species unit-cell value."""

from collections.abc import Sequence
from typing import Any, ClassVar

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.prototype.anonymize import dummy_species, is_dummy_species
from httk.atomistic.models.prototype.backend import AnonymousStructureBackend
from httk.atomistic.models.sites.like import SitesLike
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.like import SpeciesLike
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.unitcell import (
    _check_sites_length,
    _norm_cell,
    _norm_sites,
    _norm_species,
    _norm_species_at_sites,
)


class AnonymousStructure(AnonymousStructureBackend):
    """A unit cell whose site identities are consecutive dummy labels."""

    _cell: Cell
    _sites: Sites
    _species: tuple[Species, ...]
    _species_at_sites: tuple[str, ...]
    kind: ClassVar[str] = "anonymous"

    def __init__(
        self,
        cell: CellLike,
        sites: SitesLike,
        species: Sequence[SpeciesLike] | None = None,
        species_at_sites: Sequence[str] | None = None,
    ) -> None:
        if species_at_sites is None:
            raise TypeError("AnonymousStructure species_at_sites is required")
        norm_cell = _norm_cell(cell)
        norm_sites = _norm_sites(sites)
        norm_species_at_sites = _norm_species_at_sites(species_at_sites)
        if species is None:
            norm_species = tuple(dummy_species(label) for label in dict.fromkeys(norm_species_at_sites))
        else:
            norm_species = _norm_species(species)
        _check_sites_length(norm_sites, norm_species_at_sites)
        if len({value.name for value in norm_species}) != len(norm_species):
            raise ValueError("AnonymousStructure species names must be unique")
        if any(not is_dummy_species(value) for value in norm_species):
            raise ValueError("AnonymousStructure species must be dummy species")
        known = {value.name for value in norm_species}
        for label in norm_species_at_sites:
            if label not in known:
                raise ValueError(f"AnonymousStructure species_at_sites references unknown species name: {label!r}")
        expected = {anonymous_symbol(index) for index in range(len(norm_species))}
        if {value.name for value in norm_species} != expected:
            raise ValueError("AnonymousStructure species labels must be consecutive anonymous symbols from 'A'")
        self._cell = norm_cell
        self._sites = norm_sites
        self._species = norm_species
        self._species_at_sites = norm_species_at_sites

    @property
    def cell(self) -> Cell:
        return self._cell

    @property
    def sites(self) -> Sites:
        return self._sites

    @property
    def species(self) -> tuple[Species, ...]:
        return self._species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return self._species_at_sites

    @property
    def periodicity(self) -> tuple[bool, bool, bool]:
        return self._cell.periodicity

    @property
    def nperiodic_dimensions(self) -> int:
        return self._cell.nperiodic_dimensions

    @property
    def nsites(self) -> int:
        return len(self._sites)

    def cartesian_sites(self) -> Any:
        from httk.core import SurdVector

        return SurdVector.create(self._sites.reduced_coords) * self._cell.basis

    @property
    def coordinate_precision(self):
        return self._sites.precision

    @property
    def basis_precision(self):
        return self._cell.precision

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnonymousStructure):
            return NotImplemented
        return (
            self._cell == other._cell
            and self.basis_precision == other.basis_precision
            and self._sites == other._sites
            and self.coordinate_precision == other.coordinate_precision
            and self._species == other._species
            and self._species_at_sites == other._species_at_sites
        )

    def __repr__(self) -> str:
        return f"AnonymousStructure(cell={self._cell!r}, sites={self._sites!r}, species_at_sites={self._species_at_sites!r})"
