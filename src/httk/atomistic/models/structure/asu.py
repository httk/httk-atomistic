"""A crystal structure held as its asymmetric unit.

An :class:`ASUStructure` records only the symmetry-distinct sites — the asymmetric unit —
plus the space group needed to regenerate the rest. Where a :class:`~httk.atomistic.UnitcellStructure`
lists every atom in the cell, this lists one representative per orbit as a Wyckoff letter
and the values of that position's free parameters.

**Any setting, including non-standard ones.** The Wyckoff data is always recorded against
the International Tables *standard* setting, and a :class:`~httk.atomistic.SettingTransform`
carries the change of basis from there to whatever setting the structure is actually in.
A setting that appears in no table is representable just as well as a tabulated one: the
transform is stored, not looked up. That pairing is what makes the representation lossless
for arbitrary settings.

**Expansion is exact and needs no tolerance.** Reduced coordinates, symmetry operations,
Wyckoff parameters, and the setting transform are all exact rationals, and the vendored
orbits are complete and pre-deduplicated. So generating the full cell is affine arithmetic
over the rationals with an exact equality test at the end — no coordinate grid, no
snapping, no neighbour search. Tolerance enters this class only where a *measured*
structure is first recognized as symmetric, never in expansion.
"""

import datetime
import fractions
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

from httk.core import FracVector

from httk.atomistic import data
from httk.atomistic.composition import Assembly
from httk.atomistic.models._vector_guards import to_precision
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.cell.view import CellView
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.like import SpeciesLike
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.species.view import SpeciesView
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.semantics import StructureSemanticsMixin, initialize_semantics
from httk.atomistic.symmetry._periodicity_guard import require_full_periodicity
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup, wyckoff_letter_map

__all__ = ["ASUStructure", "FundamentalDomainStructure", "WyckoffSite"]


@dataclass(frozen=True)
class WyckoffSite:
    """One symmetry-distinct site: a Wyckoff position, its free parameters, and a species.

    ``wyckoff`` is a bare letter (``"e"``, not ``"4e"``) naming a position of the
    **standard setting**, and ``free_params`` holds one exact value per degree of freedom
    of that position — none at all for a fixed position such as an inversion centre.
    ``species`` names one of the owning structure's species.

    Partial occupancy needs nothing special here: it lives in the referenced
    :class:`~httk.atomistic.Species`, which already carries a composition.
    """

    wyckoff: str
    free_params: FracVector
    species: str
    representative: FracVector | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "wyckoff", str(self.wyckoff))
        object.__setattr__(self, "species", str(self.species))
        object.__setattr__(self, "free_params", FracVector.create(self.free_params))
        if self.representative is not None:
            representative = FracVector.create(self.representative)
            if representative.dim != (3,):
                raise ValueError("WyckoffSite representative must be a three-dimensional coordinate")
            object.__setattr__(self, "representative", representative)

    def __repr__(self) -> str:
        values = ", ".join(str(value) for value in self.free_params.to_fractions()) if self.free_count else ""
        return f"WyckoffSite({self.species!r} at {self.wyckoff}{f'({values})' if values else ''})"

    @property
    def free_count(self) -> int:
        """How many free parameters this site carries."""
        return 0 if self.free_params.dim in ((), (0,)) else self.free_params.dim[0]


class FundamentalDomainStructure(StructureBackend, StructureSemanticsMixin):
    """A crystal structure represented by one exact site per symmetry orbit.

    Holds the cell in the structure's own setting, the space group as its **standard**
    setting, a transform from that standard setting to the structure's own, one
    :class:`WyckoffSite` per symmetry-distinct site, and the species they name.
    """

    _cell: Cell
    _spacegroup: Spacegroup
    _transform: SettingTransform
    _wyckoff_sites: tuple[WyckoffSite, ...]
    _species: tuple[Species, ...]
    _coordinate_precision: fractions.Fraction | None
    kind: ClassVar[str] = "asu"

    def __init__(
        self,
        cell: CellLike,
        spacegroup: Spacegroup | int,
        wyckoff_sites: Sequence[WyckoffSite],
        species: Sequence[SpeciesLike],
        transform: SettingTransform | None = None,
        coordinate_precision: Any = None,
        *,
        molecular: bool = False,
        assemblies: Sequence[Any] | None = None,
        chemical_composition: Any = None,
        chemical_formula_descriptive: str | None = None,
        chemical_formula_hill: str | None = None,
        optimization_type: str | None = None,
        immutable_id: str | None = None,
        last_modified: datetime.datetime | None = None,
    ) -> None:
        self._cell = cell if isinstance(cell, Cell) else CellView(cell)
        require_full_periodicity(self._cell, "ASUStructure")
        self._spacegroup = spacegroup if isinstance(spacegroup, Spacegroup) else Spacegroup.standard(spacegroup)
        if not self._spacegroup.is_standard_setting:
            raise ValueError(
                f"ASUStructure records Wyckoff data in the IT standard setting, but was given "
                f"{self._spacegroup.setting}; pass Spacegroup.standard({self._spacegroup.it_number}) and "
                f"express the difference as the transform instead"
            )
        self._transform = SettingTransform.identity() if transform is None else transform
        self._coordinate_precision = to_precision(coordinate_precision)
        self._wyckoff_sites = tuple(wyckoff_sites)
        self._species = tuple(item if isinstance(item, Species) else SpeciesView(item) for item in species)

        names = [item.name for item in self._species]
        if len(names) != len(set(names)):
            raise ValueError("ASUStructure species names must be unique")
        known = set(names)
        for site in self._wyckoff_sites:
            if site.species not in known:
                raise ValueError(f"ASUStructure site references unknown species name: {site.species!r}")
            position = self._spacegroup.wyckoff_position(site.wyckoff)
            if site.free_count != position.free_count:
                raise ValueError(
                    f"Wyckoff position {position.multiplicity}{position.letter} of "
                    f"{self._spacegroup.setting} takes {position.free_count} free parameter(s), "
                    f"but the site supplies {site.free_count}"
                )
            if site.representative is not None and not self._representative_matches_orbit(site):
                raise ValueError(
                    f"representative coordinate for Wyckoff site {site.wyckoff!r} does not match its orbit"
                )
        initialize_semantics(
            self,
            nsites=len(self._wyckoff_sites),
            molecular=molecular,
            assemblies=None if assemblies is None else tuple(assemblies),
            symmetry=None,
            chemical_composition=chemical_composition,
            chemical_formula_descriptive=chemical_formula_descriptive,
            chemical_formula_hill=chemical_formula_hill,
            optimization_type=optimization_type,
            immutable_id=immutable_id,
            last_modified=last_modified,
        )

    # --- accessors ---

    @property
    def cell(self) -> Cell:
        """The cell, in the structure's own setting."""
        return self._cell

    @property
    def spacegroup(self) -> Spacegroup:
        """The space group, as its IT standard setting."""
        return self._spacegroup

    @property
    def transform(self) -> SettingTransform:
        """The change of basis from the standard setting to this structure's own."""
        return self._transform

    @property
    def wyckoff_sites(self) -> tuple[WyckoffSite, ...]:
        """The symmetry-distinct sites."""
        return self._wyckoff_sites

    @property
    def domain_sites(self) -> tuple[WyckoffSite, ...]:
        """Representation-neutral name for the directly stored fundamental-domain sites."""
        return self._wyckoff_sites

    @property
    def species(self) -> tuple[Species, ...]:
        """The species referenced by the sites."""
        return self._species

    @property
    def coordinate_precision(self) -> fractions.Fraction | None:
        """How precisely the coordinates behind this structure were stated, or ``None``.

        Fractional, and expressed in **this structure's own setting** — the frame the data
        arrived in — so it needs no transforming on the way to the expanded sites. Recording
        it here is what lets an asymmetric unit say how good the data behind it was, rather
        than leaving that to be guessed again downstream.

        It is provenance, never an operating parameter: expansion remains exact and uses no
        tolerance at all.
        """
        return self._coordinate_precision

    @property
    def asu(self) -> "FundamentalDomainStructure":
        """Self, so a view can recognize a backend that already holds an ASU and pass it through."""
        return self

    @property
    def periodicity(self) -> tuple[bool, bool, bool]:
        return self._cell.periodicity

    @property
    def nperiodic_dimensions(self) -> int:
        return self._cell.nperiodic_dimensions

    @property
    def molecular(self) -> bool:
        return self._molecular

    @property
    def domain_species_at_sites(self) -> tuple[str, ...]:
        """Species names of the directly represented domain sites."""
        return tuple(site.species for site in self._wyckoff_sites)

    def _representatives_for_site(self, site: WyckoffSite) -> tuple[FracVector, ...]:
        position = self._spacegroup.wyckoff_position(site.wyckoff)
        values: list[FracVector] = []
        for standard_point in position.coordinates(site.free_params):
            own_point = self._transform.to_setting(standard_point)
            values.extend((own_point + coset).normalize() for coset in self._transform.lattice_cosets())
        return tuple(values)

    def _representative_matches_orbit(self, site: WyckoffSite) -> bool:
        assert site.representative is not None
        stated = site.representative.normalize().to_fractions()
        # A source representative is retained before symmetry snapping. Coordinate
        # precision is a per-component statement, while recognition compares two rounded
        # positions and may change axes; use the same conservative three-component bound.
        tolerance = (self._coordinate_precision or fractions.Fraction()) * 3
        for candidate in self._representatives_for_site(site):
            expected = candidate.to_fractions()
            differences = []
            for left, right in zip(stated, expected):
                difference = abs(left - right) % 1
                differences.append(min(difference, 1 - difference))
            if all(value <= tolerance for value in differences):
                return True
        return False

    def _representative_sites(self) -> Sites:
        """Exact representative positions retained by the fundamental-domain representation."""
        coordinates = [
            site.representative.normalize()
            if site.representative is not None
            else self._representatives_for_site(site)[0]
            for site in self._wyckoff_sites
        ]
        return Sites(
            FracVector.create([list(value.to_fractions()) for value in coordinates]), self._coordinate_precision
        )

    def cartesian_sites(self) -> Any:
        from httk.core import SurdVector

        return SurdVector.create(self._representative_sites().reduced_coords) * self._cell.basis

    @property
    def fractional_site_positions(self) -> list[list[float]]:
        return self._representative_sites().reduced_coords.to_floats()

    @property
    def nsites(self) -> int:
        return len(self._wyckoff_sites)

    @property
    def site_coordinate_span(self) -> str:
        return "molecular_fundamental_domain" if self._molecular else "fundamental_domain"

    @property
    def space_group_it_number(self) -> int:
        return self._spacegroup.it_number

    @property
    def space_group_symbol_hall(self) -> str | None:
        setting = self.setting()
        return None if setting is None else setting.hall_symbol

    @property
    def space_group_symbol_hermann_mauguin(self) -> str | None:
        setting = self.setting()
        return None if setting is None else setting.hermann_mauguin

    @property
    def space_group_symbol_hermann_mauguin_extended(self) -> str | None:
        setting = self.setting()
        if setting is None:
            return None
        value = setting.record.get("hm_extended")
        return None if not value else " ".join(part.strip() for part in str(value).split("\n") if part.strip())

    @property
    def space_group_symmetry_operations_xyz(self) -> tuple[str, ...]:
        setting = self.setting()
        operations = (
            tuple(self._transform.symop_to_setting(value) for value in self._spacegroup.symmetry_operations)
            if setting is None
            else setting.symmetry_operations
        )
        return tuple(operation.wrapped().to_xyz() for operation in operations)

    @property
    def wyckoff_positions(self) -> tuple[str, ...] | None:
        setting = self.setting()
        if setting is None:
            return None
        letters = wyckoff_letter_map(self._spacegroup, setting)
        return tuple(setting.wyckoff_position(letters[site.wyckoff]).letter for site in self._wyckoff_sites)

    @property
    def is_standard_setting(self) -> bool:
        """Whether the structure is written in the IT standard setting of its space group."""
        return self._transform.is_identity()

    def setting(self) -> Spacegroup | None:
        """The tabulated setting this structure is written in, or ``None`` if untabulated.

        A structure in an arbitrary setting is perfectly representable but has no tabulated
        name; that is the point of storing the transform rather than a setting label.

        A transform looked up from the tables remembers which setting it came from, but one
        that was constructed directly does not, so an equal transform is also matched
        against the group's tabulated settings. An identity transform means the structure is
        in the standard setting, which is of course tabulated — reporting it as nameless
        would be simply wrong.
        """
        if self._transform.is_identity():
            return self._spacegroup

        hall_entry = self._transform.hall_entry
        if hall_entry is not None:
            return Spacegroup.for_hall_entry(hall_entry)

        for record in data.spacegroup_settings():
            if record["it_number"] != self._spacegroup.it_number:
                continue
            candidate = Spacegroup(record)
            if candidate.transform_from_standard == self._transform:
                return candidate
        return None

    # --- expansion ---

    def _expanded_offsets(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        counts = self.multiplicities()
        offsets: list[int] = []
        offset = 0
        for count in counts:
            offsets.append(offset)
            offset += count
        return counts, tuple(offsets)

    def _expanded_assemblies(self) -> tuple[Assembly, ...] | None:
        assemblies = self._assemblies
        if assemblies is None or not assemblies:
            return assemblies
        counts, offsets = self._expanded_offsets()
        expanded: list[Assembly] = []
        for assembly in assemblies:
            groups: list[tuple[int, ...]] = []
            for group in assembly.sites_in_groups:
                if any(counts[index] != 1 for index in group):
                    raise ValueError(
                        "symmetry-reduced expansion cannot map assembly correlations "
                        "when a correlated domain site has multiple unit-cell images"
                    )
                groups.append(tuple(offsets[index] for index in group))
            expanded.append(
                Assembly(
                    tuple(groups),
                    assembly.group_probabilities,
                    assembly.group_probabilities_precision,
                )
            )
        return tuple(expanded)

    def _validate_expansion_semantics(self) -> None:
        self._expanded_assemblies()
        if not self.molecular:
            return
        counts = self.multiplicities()
        if any(count != 1 for count in counts) or any(site.representative is None for site in self.wyckoff_sites):
            raise ValueError(
                "symmetry-reduced molecular expansion requires one retained representative "
                "for every one-to-one domain site"
            )

    @cached_property
    def _expansion(self) -> tuple[FracVector, tuple[str, ...], tuple[int, ...]]:
        """The full cell: coordinates, the species at each, and the per-site counts.

        Computed once. Sites generated by *different* asymmetric-unit sites are deduplicated
        against each other too, not only within an orbit, so two sites that name the same
        point cannot silently produce a doubled atom.
        """
        transform = self._transform
        cosets = transform.lattice_cosets()

        coordinates: list[tuple[fractions.Fraction, ...]] = []
        species_at_sites: list[str] = []
        counts: list[int] = []
        seen: set[tuple[fractions.Fraction, ...]] = set()

        for site in self._wyckoff_sites:
            position = self._spacegroup.wyckoff_position(site.wyckoff)
            # The tabulated orbit is the complete, already-deduplicated set of equivalent
            # points in the standard setting, so the group's operations never need to be
            # applied one by one here.
            generated: list[tuple[fractions.Fraction, ...]] = []
            for standard_point in position.coordinates(site.free_params):
                own_point = transform.to_setting(standard_point)
                for coset in cosets:
                    key = tuple((own_point + coset).normalize().to_fractions())
                    if key not in seen:
                        seen.add(key)
                        generated.append(key)

            # Deterministic order, so an expansion is reproducible run to run.
            counts.append(len(generated))
            for key in sorted(generated):
                coordinates.append(key)
                species_at_sites.append(site.species)

        if not coordinates:
            return FracVector.create(()), (), tuple(counts)
        return FracVector.create([list(point) for point in coordinates]), tuple(species_at_sites), tuple(counts)

    def expand_sites(self) -> Sites:
        """Every site of the unit cell, as exact reduced coordinates in this structure's setting.

        The orbit of each asymmetric-unit site is generated in the standard setting, mapped
        through the setting transform, wrapped into ``[0, 1)``, and deduplicated by exact
        equality. Deduplication is not a formality: when the transform shrinks the cell — as
        it does for the seven rhombohedral-axes settings, where ``det M == 3`` — the standard
        setting's orbit is three times too large and the surplus points coincide exactly.
        The opposite case, a transform onto a larger cell, is covered by
        :meth:`~httk.atomistic.SettingTransform.lattice_cosets`.
        """
        return Sites(self._expansion[0], self._coordinate_precision)

    def expand_species_at_sites(self) -> tuple[str, ...]:
        """The species name occupying each site produced by :meth:`expand_sites`, in order."""
        return self._expansion[1]

    def multiplicities(self) -> tuple[int, ...]:
        """How many cell sites each asymmetric-unit site generates, in order.

        Usually the Wyckoff position's tabulated multiplicity, but not always: a setting
        transform that changes the cell volume changes the count too, by a factor of three
        for the rhombohedral-axes settings.
        """
        return self._expansion[2]

    @property
    def sites(self) -> Sites:
        self._validate_expansion_semantics()
        return self._representative_sites() if self.molecular else self.expand_sites()

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        self._validate_expansion_semantics()
        return self.domain_species_at_sites if self.molecular else self.expand_species_at_sites()

    @property
    def assemblies(self) -> tuple[Assembly, ...] | None:
        return self._assemblies

    # --- identity ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FundamentalDomainStructure) or (
            type(self) is not type(other) and not (isinstance(self, ASUStructure) and isinstance(other, ASUStructure))
        ):
            return NotImplemented
        return (
            self._cell == other._cell
            and self._cell.precision == other._cell.precision
            and self._spacegroup == other._spacegroup
            and self._transform == other._transform
            and self._wyckoff_sites == other._wyckoff_sites
            and self._species == other._species
            and self._coordinate_precision == other._coordinate_precision
            and self._molecular == other._molecular
            and self._assemblies == other._assemblies
            and self._chemical_composition == other._chemical_composition
            and self._chemical_formula_descriptive == other._chemical_formula_descriptive
            and self._chemical_formula_hill == other._chemical_formula_hill
            and self._optimization_type == other._optimization_type
        )

    def __repr__(self) -> str:
        setting = self.setting()
        where = (
            "standard setting"
            if self.is_standard_setting
            else f"setting {setting.setting if setting else '(untabulated)'}"
        )
        return (
            f"{type(self).__name__}({self._spacegroup.hermann_mauguin!r}, {len(self._wyckoff_sites)} site(s), {where})"
        )


class ASUStructure(FundamentalDomainStructure):
    """A fundamental domain asserted by its creator to be a true asymmetric unit."""

    @property
    def site_coordinate_span(self) -> str:
        return "molecular_asymmetric_unit" if self._molecular else "asymmetric_unit"
