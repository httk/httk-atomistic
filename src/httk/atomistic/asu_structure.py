"""A crystal structure held as its asymmetric unit.

An :class:`ASUStructure` records only the symmetry-distinct sites — the asymmetric unit —
plus the space group needed to regenerate the rest. Where a :class:`~httk.atomistic.Structure`
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

import fractions
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Sequence

from httk.core import FracVector

from .cell import Cell
from .cell_class_view import CellClassView
from .cell_like import CellLike
from .setting_transform import SettingTransform
from .sites import Sites
from .spacegroup import Spacegroup
from .species import Species
from .species_class_view import SpeciesClassView
from .species_like import SpeciesLike

__all__ = ["ASUSite", "ASUStructure"]


@dataclass(frozen=True)
class ASUSite:
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "wyckoff", str(self.wyckoff))
        object.__setattr__(self, "species", str(self.species))
        object.__setattr__(self, "free_params", FracVector.create(self.free_params))

    def __repr__(self) -> str:
        values = ", ".join(str(value) for value in self.free_params.to_fractions()) if self.free_count else ""
        return f"ASUSite({self.species!r} at {self.wyckoff}{f'({values})' if values else ''})"

    @property
    def free_count(self) -> int:
        """How many free parameters this site carries."""
        return 0 if self.free_params.dim in ((), (0,)) else self.free_params.dim[0]


class ASUStructure:
    """A crystal structure represented by its asymmetric unit.

    Holds the cell in the structure's own setting, the space group as its **standard**
    setting, a transform from that standard setting to the structure's own, one
    :class:`ASUSite` per symmetry-distinct site, and the species they name.
    """

    _cell: Cell
    _spacegroup: Spacegroup
    _transform: SettingTransform
    _asu_sites: tuple[ASUSite, ...]
    _species: tuple[Species, ...]

    def __init__(
        self,
        cell: CellLike,
        spacegroup: Spacegroup | int,
        asu_sites: Sequence[ASUSite],
        species: Sequence[SpeciesLike],
        transform: SettingTransform | None = None,
    ) -> None:
        self._cell = cell if isinstance(cell, Cell) else CellClassView(cell)
        self._spacegroup = spacegroup if isinstance(spacegroup, Spacegroup) else Spacegroup.standard(spacegroup)
        if not self._spacegroup.is_standard_setting:
            raise ValueError(
                f"ASUStructure records Wyckoff data in the IT standard setting, but was given "
                f"{self._spacegroup.setting}; pass Spacegroup.standard({self._spacegroup.it_number}) and "
                f"express the difference as the transform instead"
            )
        self._transform = SettingTransform.identity() if transform is None else transform
        self._asu_sites = tuple(asu_sites)
        self._species = tuple(item if isinstance(item, Species) else SpeciesClassView(item) for item in species)

        names = [item.name for item in self._species]
        if len(names) != len(set(names)):
            raise ValueError("ASUStructure species names must be unique")
        known = set(names)
        for site in self._asu_sites:
            if site.species not in known:
                raise ValueError(f"ASUStructure site references unknown species name: {site.species!r}")
            position = self._spacegroup.wyckoff_position(site.wyckoff)
            if site.free_count != position.free_count:
                raise ValueError(
                    f"Wyckoff position {position.multiplicity}{position.letter} of "
                    f"{self._spacegroup.setting} takes {position.free_count} free parameter(s), "
                    f"but the site supplies {site.free_count}"
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
    def asu_sites(self) -> tuple[ASUSite, ...]:
        """The symmetry-distinct sites."""
        return self._asu_sites

    @property
    def species(self) -> tuple[Species, ...]:
        """The species referenced by the sites."""
        return self._species

    @property
    def asu(self) -> "ASUStructure":
        """Self, so a view can recognize a backend that already holds an ASU and pass it through."""
        return self

    @property
    def is_standard_setting(self) -> bool:
        """Whether the structure is written in the IT standard setting of its space group."""
        return self._transform.is_identity()

    def setting(self) -> Spacegroup | None:
        """The tabulated setting this structure is written in, or ``None`` if untabulated.

        A structure in an arbitrary setting is perfectly representable but has no tabulated
        name; that is the point of storing the transform rather than a setting label.
        """
        hall_entry = self._transform.hall_entry
        return None if hall_entry is None else Spacegroup.for_hall_entry(hall_entry)

    # --- expansion ---

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

        for site in self._asu_sites:
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
        return Sites(self._expansion[0])

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

    def to_structure(self) -> Any:
        """The equivalent full-cell :class:`~httk.atomistic.Structure`.

        Equivalent to ``StructureSimpleView(asu_structure)``; provided as a plain method so
        the common case needs no knowledge of the view machinery.
        """
        from .structure_simple_view import StructureSimpleView

        return StructureSimpleView(self)

    # --- identity ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ASUStructure):
            return NotImplemented
        return (
            self._cell == other._cell
            and self._spacegroup == other._spacegroup
            and self._transform == other._transform
            and self._asu_sites == other._asu_sites
            and self._species == other._species
        )

    def __repr__(self) -> str:
        setting = self.setting()
        where = (
            "standard setting"
            if self.is_standard_setting
            else f"setting {setting.setting if setting else '(untabulated)'}"
        )
        return f"ASUStructure({self._spacegroup.hermann_mauguin!r}, {len(self._asu_sites)} site(s), {where})"
