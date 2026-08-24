"""The immutable anonymous geometrical-class prototype value."""

from collections.abc import Sequence
from typing import ClassVar

from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.prototype.backend import PrototypeBackend
from httk.atomistic.models.prototype.notation import canonical_label_map
from httk.atomistic.models.prototype.occupation import PrototypeOccupation
from httk.atomistic.models.structuretype.fundamental import FundamentalDomainTemplate
from httk.atomistic.symmetry.spacegroup import Spacegroup


class Prototype(PrototypeBackend):
    """Store anonymous occupied Wyckoff positions and optional class information.

    The base value is provenance-independent: recognition and derivation return a base
    ``Prototype``, so a value recognized from a structure compares equal to one built by hand
    or parsed from a label. A geometrical representative and/or a discriminator are optional
    refinements, present only when the user constructs the value with them; recognition never
    attaches them. They participate in equality and content identity.

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
        prototype: "Prototype | None" = None,
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
            if not isinstance(prototype, Prototype):
                from httk.atomistic.models.prototype.view import PrototypeView

                prototype = PrototypeView(prototype).unview()
            if base_supplied:
                raise TypeError("Prototype accepts either prototype or spacegroup/occupations")
            spacegroup, occupations = prototype.spacegroup, prototype.occupations
            if representative is None:
                representative = prototype.representative
            if discriminator is None:
                discriminator = prototype.discriminator
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
        self._spacegroup = spacegroup if isinstance(spacegroup, Spacegroup) else Spacegroup.standard(spacegroup)
        if not self._spacegroup.is_standard_setting:
            raise ValueError("Prototype records Wyckoff data in the IT standard setting")
        raw = tuple(
            value if isinstance(value, PrototypeOccupation) else PrototypeOccupation(*value) for value in occupations
        )
        if not raw:
            raise ValueError("Prototype occupations must be non-empty")
        letters_by_label: dict[str, list[str]] = {}
        for value in raw:
            try:
                self._spacegroup.wyckoff_position(value.wyckoff)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            letters_by_label.setdefault(value.label, []).append(value.wyckoff)
        expected = {anonymous_symbol(index) for index in range(len(letters_by_label))}
        if set(letters_by_label) != expected:
            raise ValueError("Prototype class labels must be consecutive anonymous symbols from 'A'")
        relabel = canonical_label_map({label: tuple(sorted(letters)) for label, letters in letters_by_label.items()})
        self._occupations = tuple(
            sorted(
                (PrototypeOccupation(value.wyckoff, relabel[value.label]) for value in raw),
                key=lambda value: (value.label, value.wyckoff),
            )
        )
        if representative is not None and base_supplied:
            expected_base = Prototype(representative=representative)
            if (self._spacegroup, self._occupations) != (expected_base.spacegroup, expected_base.occupations):
                raise ValueError("Prototype base disagrees with its representative")
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
