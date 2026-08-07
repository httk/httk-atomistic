"""The standard-setting dummy-species fundamental-domain value."""

from collections.abc import Sequence
from functools import cached_property
from typing import Any, ClassVar, Self

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.cell.view import CellView
from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.prototype.anonymize import dummy_species, is_dummy_species
from httk.atomistic.models.prototype.backend import AnonymousStructureBackend
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.like import SpeciesLike
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.species.view import SpeciesView
from httk.atomistic.models.structure.asu import FundamentalDomainStructure, WyckoffSite
from httk.atomistic.symmetry._periodicity_guard import require_full_periodicity
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup


class Prototype(AnonymousStructureBackend):
    """Store a standard-setting fundamental domain with dummy species labels.

    :param cell: The standard-setting cell geometry.
    :param spacegroup: The standard-setting space group.
    :param wyckoff_sites: The symmetry-distinct site definitions.
    :param species: The distinct dummy species definitions.
    :param coordinate_precision: The precision recorded for the reduced coordinates.
    """

    _cell: Cell
    _spacegroup: Spacegroup
    _wyckoff_sites: tuple[WyckoffSite, ...]
    _species: tuple[Species, ...]
    _coordinate_precision: Any
    kind: ClassVar[str] = "prototype"

    def __init__(
        self,
        cell: CellLike,
        spacegroup: Spacegroup | int,
        wyckoff_sites: Sequence[WyckoffSite],
        species: Sequence[SpeciesLike] | None = None,
        coordinate_precision: Any = None,
    ) -> None:
        self._cell = cell if isinstance(cell, Cell) else CellView(cell)
        require_full_periodicity(self._cell, "Prototype")
        self._spacegroup = spacegroup if isinstance(spacegroup, Spacegroup) else Spacegroup.standard(spacegroup)
        if not self._spacegroup.is_standard_setting:
            raise ValueError(
                f"Prototype records Wyckoff data in the IT standard setting, but was given "
                f"{self._spacegroup.setting}; pass Spacegroup.standard({self._spacegroup.it_number}) "
                "and express the difference as the transform instead"
            )
        raw_sites = tuple(wyckoff_sites)
        labels = tuple(site.species for site in raw_sites)
        if species is None:
            norm_species = tuple(dummy_species(label) for label in dict.fromkeys(labels))
        else:
            norm_species = tuple(value if isinstance(value, Species) else SpeciesView(value) for value in species)
        if len({value.name for value in norm_species}) != len(norm_species):
            raise ValueError("Prototype species names must be unique")
        if any(not is_dummy_species(value) for value in norm_species):
            raise ValueError("Prototype species must be dummy species")
        known = {value.name for value in norm_species}
        for site in raw_sites:
            if site.representative is not None:
                raise ValueError("Prototype WyckoffSite representative is not supported")
            if site.moment is not None:
                raise ValueError("Prototype WyckoffSite moment is not supported")
            if site.species not in known:
                raise ValueError(f"Prototype WyckoffSite references unknown species name: {site.species!r}")
            position = self._spacegroup.wyckoff_position(site.wyckoff)
            if site.free_count != position.free_count:
                raise ValueError(
                    f"Wyckoff position {position.multiplicity}{position.letter} of {self._spacegroup.setting} "
                    f"takes {position.free_count} free parameter(s), but the site supplies {site.free_count}"
                )
        expected = {anonymous_symbol(index) for index in range(len(norm_species))}
        if {value.name for value in norm_species} != expected:
            raise ValueError("Prototype species labels must be consecutive anonymous symbols from 'A'")
        self._wyckoff_sites = tuple(
            sorted(raw_sites, key=lambda site: (site.species, site.wyckoff, site.free_params.to_fractions()))
        )
        self._species = norm_species
        from httk.atomistic.models._vector_guards import to_precision

        self._coordinate_precision = to_precision(coordinate_precision)

    @property
    def cell(self) -> Cell:
        """Return the standard-setting cell."""
        return self._cell

    @property
    def spacegroup(self) -> Spacegroup:
        """Return the standard-setting space group."""
        return self._spacegroup

    @property
    def wyckoff_sites(self) -> tuple[WyckoffSite, ...]:
        """Return the symmetry-distinct site definitions."""
        return self._wyckoff_sites

    @property
    def species(self) -> tuple[Species, ...]:
        """Return the distinct dummy species."""
        return self._species

    @property
    def coordinate_precision(self):
        """Return the reduced-coordinate precision."""
        return self._coordinate_precision

    @property
    def basis_precision(self):
        """Return the cell-basis precision."""
        return self._cell.precision

    @cached_property
    def _domain(self) -> FundamentalDomainStructure:
        return FundamentalDomainStructure(
            self._cell,
            self._spacegroup,
            self._wyckoff_sites,
            self._species,
            transform=SettingTransform.identity(),
            coordinate_precision=self._coordinate_precision,
        )

    @property
    def sites(self) -> Sites:
        """Return the expanded standard-setting sites."""
        return self._domain.sites

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Return dummy species names in expanded site order."""
        return self._domain.species_at_sites

    @property
    def periodicity(self) -> tuple[bool, bool, bool]:
        """Return the periodic directions."""
        return self._cell.periodicity

    @property
    def nperiodic_dimensions(self) -> int:
        """Return the number of periodic directions."""
        return self._cell.nperiodic_dimensions

    @property
    def nsites(self) -> int:
        """Return the number of expanded sites."""
        return len(self.sites)

    def cartesian_sites(self) -> Any:
        """Return the exact Cartesian expanded site positions."""
        return self._domain.cartesian_sites()

    def multiplicities(self) -> tuple[int, ...]:
        """Return the standard-setting multiplicity for each Wyckoff site."""
        return self._domain.multiplicities()

    @property
    def nsites_conventional(self) -> int:
        """Return the number of sites in the conventional cell."""
        return sum(self.multiplicities())

    @property
    def prototype(self) -> Self:
        """Return this prototype value."""
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Prototype):
            return NotImplemented
        return (
            self._cell == other._cell
            and self.basis_precision == other.basis_precision
            and self._spacegroup == other._spacegroup
            and self._wyckoff_sites == other._wyckoff_sites
            and self._species == other._species
            and self.coordinate_precision == other.coordinate_precision
        )

    def __repr__(self) -> str:
        return f"Prototype({self._spacegroup.setting!r}, wyckoff_sites={self._wyckoff_sites!r})"
