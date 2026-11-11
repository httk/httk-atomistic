"""The immutable assigned-species geometrical-classification value."""

from collections.abc import Sequence
from typing import Any, ClassVar

from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.species.classes import _class_sites, _merges_classes
from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure
from httk.atomistic.symmetry.spacegroup import Spacegroup


class Protostructure(ProtostructureBackend):
    """Store occupied Wyckoff positions with explicit geometrical-class information.

    At least one of an exact representative or a nonempty discriminator is required.
    Both participate in equality and content identity. Use ``BareProtostructure`` for
    the broader Wyckoff-only classification; recognizing a structure yields that level.

    :param spacegroup: The standard-setting space group or its IT number.
    :param occupations: The occupied Wyckoff positions and their species.
    :param representative: An optional exact standard-setting class anchor. It is stored in
        terms of species classes: its sites are renamed to their class names and only the used
        classes are kept, as for the occupations. When two of its species share a class, the
        class structure can have more symmetry than the declared one, and the representative
        becomes the canonical ASU of the class structure
        (:func:`~httk.atomistic.symmetry.canonical_protostructure.canonical_asu_protostructure`).
    :param discriminator: An optional external class discriminator.
    """

    _spacegroup: Spacegroup
    _occupations: tuple[WyckoffOccupation, ...]
    kind: ClassVar[str] = "protostructure"

    def __init__(
        self,
        spacegroup: Spacegroup | int | None = None,
        occupations: Sequence[WyckoffOccupation | tuple[str, Any]] | None = None,
        *,
        representative: FundamentalDomainStructure | None = None,
        discriminator: str | None = None,
    ) -> None:
        base_supplied = spacegroup is not None or occupations is not None
        if representative is not None:
            _validate_representative(representative)
            if _merges_classes(representative):
                # the declared symmetry describes the named species; their classes can have more
                from httk.atomistic.symmetry.canonical_protostructure import canonical_asu_protostructure

                representative = canonical_asu_protostructure(representative)
            representative = _class_representative(representative)
            if not base_supplied:
                spacegroup = representative.spacegroup
                representative_species = {species.name: species for species in representative.species}
                occupations = tuple(
                    (site.wyckoff, representative_species[site.species]) for site in representative.wyckoff_sites
                )
        if spacegroup is None or occupations is None:
            raise ValueError("Protostructure needs spacegroup and occupations or a representative")
        if discriminator is not None and (not isinstance(discriminator, str) or not discriminator):
            raise ValueError("Protostructure discriminator must be a non-empty string when given")
        from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure

        bare = BareProtostructure(spacegroup, occupations)
        self._spacegroup, self._occupations = bare.spacegroup, bare.occupations
        if representative is not None and base_supplied:
            expected = Protostructure(representative=representative)
            if (self._spacegroup, self._occupations) != (expected.spacegroup, expected.occupations):
                raise ValueError("Protostructure base disagrees with its representative")
        if representative is None and discriminator is None:
            raise ValueError(
                "Protostructure requires a representative or discriminator; use BareProtostructure for Wyckoff-only values"
            )
        self._representative = representative
        self._discriminator = discriminator

    @property
    def spacegroup(self) -> Spacegroup:
        """Return the standard-setting space group."""
        return self._spacegroup

    @property
    def occupations(self) -> tuple[WyckoffOccupation, ...]:
        """Return the canonical occupied Wyckoff positions."""
        return self._occupations

    @property
    def representative(self) -> FundamentalDomainStructure | None:
        return self._representative

    @property
    def discriminator(self) -> str | None:
        return self._discriminator

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Protostructure):
            return NotImplemented
        return (self._spacegroup, self._occupations, self._representative, self._discriminator) == (
            other._spacegroup,
            other._occupations,
            other._representative,
            other._discriminator,
        )

    def __hash__(self) -> int:
        return hash((self._spacegroup, self._occupations, self._discriminator))

    def __repr__(self) -> str:
        pairs = ", ".join(f"{occupation.wyckoff}:{occupation.species.name}" for occupation in self._occupations)
        parts = [f"{self._spacegroup.setting!r}, {pairs}"]
        if self._representative is not None:
            parts.append("representative=...")
        if self._discriminator is not None:
            parts.append(f"discriminator={self._discriminator!r}")
        return f"Protostructure({', '.join(parts)})"


def _class_representative(representative: FundamentalDomainStructure) -> FundamentalDomainStructure:
    """Return the representative in terms of species classes, of the same kind.

    Sites are renamed to their class names and only the used classes are kept (see
    :func:`~httk.atomistic.models.species.classes._class_sites`); a representative that already
    is in that form is returned unchanged.
    """
    sites, species = _class_sites(representative)
    if sites == representative.wyckoff_sites and species == representative.species:
        return representative
    kind = ASUStructure if isinstance(representative, ASUStructure) else FundamentalDomainStructure
    return kind(
        representative.cell,
        representative.spacegroup,
        sites,
        species,
        transform=representative.transform,
        coordinate_precision=representative.coordinate_precision,
        chemical_composition=representative.chemical_composition,
        chemical_formula_descriptive=representative.chemical_formula_descriptive,
        chemical_formula_hill=representative.chemical_formula_hill,
        optimization_type=representative.optimization_type,
        immutable_id=representative.immutable_id,
        last_modified=representative.last_modified,
        charge=representative.charge,
    )


def _validate_representative(representative: FundamentalDomainStructure) -> None:
    if not isinstance(representative, FundamentalDomainStructure):
        raise TypeError("Protostructure representative must be a FundamentalDomainStructure")
    if not representative.spacegroup.is_standard_setting:
        raise ValueError("Protostructure representative must record Wyckoff data in the IT standard setting")
    if not representative.transform.is_identity():
        raise ValueError("Protostructure representative must use an identity setting transform")
    if representative.assemblies is not None:
        raise ValueError("Protostructure representative cannot carry assemblies")
    if representative.molecular:
        raise ValueError("Protostructure representative cannot be molecular")
    if any(site.moment is not None for site in representative.wyckoff_sites):
        raise ValueError("Protostructure representative cannot carry site moments")
