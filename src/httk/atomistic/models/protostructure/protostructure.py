"""The immutable assigned-species geometrical-classification value."""

from collections.abc import Sequence
from typing import Any, ClassVar

from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.structure.asu import FundamentalDomainStructure
from httk.atomistic.symmetry.spacegroup import Spacegroup


class Protostructure(ProtostructureBackend):
    """Store a standard-setting space group and its occupied Wyckoff positions.

    Multiplicities, composition, and formula derivations are defined at the
    standard-setting conventional-cell scale, even when the source used to recognize it
    was stored in a volume-scaled setting.

    ``Protostructure`` is the assigned-species Wyckoff value. It may carry an optional
    geometrical representative and/or discriminator in the same value. Equality (and
    therefore content identity) covers the space group, the occupations, the
    discriminator, and the representative when present; two values sharing space group,
    occupations, and discriminator but differing in representative are not equal.

    :param spacegroup: The standard-setting space group or its IT number.
    :param occupations: The occupied Wyckoff positions and their species.
    :param representative: An optional exact standard-setting class anchor.
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
        self._spacegroup = spacegroup if isinstance(spacegroup, Spacegroup) else Spacegroup.standard(spacegroup)
        if not self._spacegroup.is_standard_setting:
            raise ValueError(
                f"Protostructure records Wyckoff data in the IT standard setting, but was given "
                f"{self._spacegroup.setting}; pass Spacegroup.standard({self._spacegroup.it_number}) "
                "instead"
            )
        raw = tuple(
            occupation if isinstance(occupation, WyckoffOccupation) else WyckoffOccupation(occupation[0], occupation[1])
            for occupation in occupations
        )
        if not raw:
            raise ValueError("Protostructure occupations must be non-empty")
        species_by_name: dict[str, Any] = {}
        for occupation in raw:
            try:
                self._spacegroup.wyckoff_position(occupation.wyckoff)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            previous = species_by_name.get(occupation.species.name)
            if previous is not None and previous != occupation.species:
                raise ValueError(
                    f"Protostructure occupations naming species {occupation.species.name!r} must carry equal Species"
                )
            species_by_name[occupation.species.name] = occupation.species
        self._occupations = tuple(sorted(raw, key=lambda value: (value.species.name, value.wyckoff)))
        if representative is not None and base_supplied:
            expected = Protostructure(representative=representative)
            if (self._spacegroup, self._occupations) != (expected.spacegroup, expected.occupations):
                raise ValueError("Protostructure base disagrees with its representative")
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
