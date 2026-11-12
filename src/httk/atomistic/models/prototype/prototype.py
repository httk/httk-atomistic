"""The immutable anonymous geometrical-class prototype value."""

from collections.abc import Sequence
from typing import ClassVar

from httk.atomistic.models.bareprototype.bareprototype import BarePrototype
from httk.atomistic.models.prototype.backend import PrototypeBackend
from httk.atomistic.models.prototype.notation import _known, _parse_undetermined, _undetermined
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
    :param representative: An optional exact anonymous class anchor. A left-handed one is stored
        as the same crystal in a right-handed basis, so its Wyckoff letters, and in an
        enantiomorphic pair its group, read the crystal's hand.
    :param discriminator: An optional external class discriminator.
    :param prototype: An optional existing prototype whose base and unspecified optional
        fields are copied.
    """

    kind: ClassVar[str] = "prototype"
    _undetermined: str | None = None

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
            from httk.atomistic.symmetry.canonical_protostructure import _right_handed_template

            unview = getattr(representative, "unview", None)
            representative = _right_handed_template(unview() if unview is not None else representative)
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

    @classmethod
    def undetermined(cls, label: str) -> "Prototype":
        """Return the undetermined prototype of a label in which ``X`` marks what is not determined.

        The canonicalisers return such a value when a part did not finish within its time budget; it
        holds only its label (no representative, no discriminator): ``<Formulatype>_X``,
        ``<Formulatype>_<Pearson>_<number>_X``, or the determined bare label when only the geometry
        was not determined. Two undetermined values are equal when their labels are; that does not
        mean their structures are the same.

        :param label: The undetermined label.
        :return: The undetermined value, with ``determined`` false.
        :raises ValueError: If ``label`` is not an undetermined prototype label.
        """
        _parse_undetermined(label, cls.kind)
        return Prototype._undetermined_of(label)

    @classmethod
    def _undetermined_of(cls, label: str) -> "Prototype":
        """Build an undetermined value from a trusted label, without parsing it."""
        instance = object.__new__(Prototype)
        instance._undetermined = label
        instance._representative = instance._discriminator = None
        return instance

    @property
    def spacegroup(self) -> Spacegroup:
        if (undetermined := _undetermined(self)) is not None:
            return _known(self, "spacegroup", undetermined.spacegroup)
        return self._spacegroup

    @property
    def occupations(self) -> tuple[PrototypeOccupation, ...]:
        if (undetermined := _undetermined(self)) is not None:
            return _known(self, "occupations", undetermined.occupations)
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
        if self._undetermined is not None or other._undetermined is not None:
            return self._undetermined == other._undetermined
        return (self._spacegroup, self._occupations, self._representative, self._discriminator) == (
            other._spacegroup,
            other._occupations,
            other._representative,
            other._discriminator,
        )

    def __hash__(self) -> int:
        if self._undetermined is not None:
            return hash(self._undetermined)
        return hash((self._spacegroup, self._occupations, self._discriminator))

    def __repr__(self) -> str:
        if self._undetermined is not None:
            return f"Prototype.undetermined({self._undetermined!r})"
        pairs = ", ".join(f"{value.wyckoff}:{value.label}" for value in self._occupations)
        parts = [f"{self._spacegroup.setting!r}, {pairs}"]
        if self._representative is not None:
            parts.append("representative=...")
        if self._discriminator is not None:
            parts.append(f"discriminator={self._discriminator!r}")
        return f"Prototype({', '.join(parts)})"
