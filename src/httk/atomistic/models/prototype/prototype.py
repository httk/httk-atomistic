"""The immutable anonymous geometrical-class prototype value."""

from collections.abc import Sequence
from typing import ClassVar

from httk.atomistic.models.bareprototype.bareprototype import BarePrototype
from httk.atomistic.models.prototype.backend import PrototypeBackend
from httk.atomistic.models.prototype.occupation import PrototypeOccupation
from httk.atomistic.models.structuretype.fundamental import FundamentalDomainTemplate
from httk.atomistic.symmetry.spacegroup import Spacegroup


class Prototype(PrototypeBackend):
    """Store occupied Wyckoff positions with explicit geometrical-class information.

    At least one of an exact representative or a nonempty discriminator is required.
    Both participate in equality and content identity. Use ``BarePrototype`` for
    the broader Wyckoff-only classification; recognizing a structure yields that level.

    :param spacegroup: The standard-setting space group or its IT number.
    :param occupations: The occupied Wyckoff positions and canonical anonymous labels.
    :param representative: An optional exact anonymous class anchor.
    :param discriminator: An optional external class discriminator.
    :param prototype: An optional existing prototype whose base and unspecified optional
        fields are copied.
    """

    kind: ClassVar[str] = "prototype"

    def __init__(
        self,
        spacegroup: Spacegroup | int | None = None,
        occupations: Sequence[PrototypeOccupation | tuple[str, str]] | None = None,
        *,
        representative: FundamentalDomainTemplate | None = None,
        discriminator: str | None = None,
        prototype: "Prototype | BarePrototype | None" = None,
    ) -> None:
        if (spacegroup is None) != (occupations is None):
            raise ValueError("Prototype spacegroup and occupations must be supplied together")
        base_supplied = spacegroup is not None
        if representative is not None:
            if not isinstance(representative, FundamentalDomainTemplate):
                raise TypeError("Prototype representative must be a FundamentalDomainTemplate")
            unview = getattr(representative, "unview", None)
            representative = unview() if unview is not None else representative
        if prototype is not None:
            if not isinstance(prototype, (Prototype, BarePrototype)):
                from httk.atomistic.models.prototype.view import PrototypeView

                prototype = PrototypeView(prototype).unview()
            if base_supplied:
                raise TypeError("Prototype accepts either prototype or spacegroup/occupations")
            spacegroup, occupations = prototype.spacegroup, prototype.occupations
            if representative is None:
                representative = getattr(prototype, "representative", None)
            if discriminator is None:
                discriminator = getattr(prototype, "discriminator", None)
            base_supplied = True
        if discriminator is not None and (not isinstance(discriminator, str) or not discriminator):
            raise ValueError("Prototype discriminator must be a non-empty string when given")
        if spacegroup is None or occupations is None:
            if representative is None:
                raise ValueError("Prototype needs spacegroup and occupations or a representative")
            spacegroup, occupations = (
                representative.spacegroup,
                [(site.wyckoff, site.species) for site in representative.wyckoff_sites],
            )
        bare = BarePrototype(spacegroup, occupations)
        self._spacegroup, self._occupations = bare.spacegroup, bare.occupations
        if representative is not None and base_supplied:
            expected_base = Prototype(representative=representative)
            if (self._spacegroup, self._occupations) != (expected_base.spacegroup, expected_base.occupations):
                raise ValueError("Prototype base disagrees with its representative")
        if representative is None and discriminator is None:
            raise ValueError(
                "Prototype requires a representative or discriminator; use BarePrototype for Wyckoff-only values"
            )
        self._representative = representative
        self._discriminator = discriminator

    @property
    def spacegroup(self) -> Spacegroup:
        return self._spacegroup

    @property
    def occupations(self) -> tuple[PrototypeOccupation, ...]:
        return self._occupations

    @property
    def representative(self) -> FundamentalDomainTemplate | None:
        return self._representative

    @property
    def discriminator(self) -> str | None:
        return self._discriminator

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Prototype):
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
        pairs = ", ".join(f"{value.wyckoff}:{value.label}" for value in self._occupations)
        parts = [f"{self._spacegroup.setting!r}, {pairs}"]
        if self._representative is not None:
            parts.append("representative=...")
        if self._discriminator is not None:
            parts.append(f"discriminator={self._discriminator!r}")
        return f"Prototype({', '.join(parts)})"
