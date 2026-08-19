"""The immutable assigned geometrical-class structuretype value."""

from typing import TYPE_CHECKING, ClassVar

from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.structure.asu import FundamentalDomainStructure
from httk.atomistic.models.structuretype.backend import StructuretypeBackend

if TYPE_CHECKING:
    from httk.atomistic.models.protostructure.like import ProtostructureLike


def _protostructure_from_representative(representative: FundamentalDomainStructure) -> Protostructure:
    """Validate a class representative and derive its geometry-free protostructure.

    The representative must be a standard-setting fundamental domain with an identity
    transform and no assemblies or site moments, so its Wyckoff letters can be read directly
    and its species resolved by name.

    :param representative: The candidate class representative.
    :return: The derived protostructure.
    :raises TypeError: If ``representative`` is not a fundamental-domain structure.
    :raises ValueError: If the representative is not a clean standard-setting class anchor.
    """
    if not isinstance(representative, FundamentalDomainStructure):
        raise TypeError("Structuretype representative must be a FundamentalDomainStructure")
    if not representative.spacegroup.is_standard_setting:
        raise ValueError("Structuretype representative must record Wyckoff data in the IT standard setting")
    if not representative.transform.is_identity():
        raise ValueError("Structuretype representative must use an identity setting transform")
    if representative.assemblies is not None:
        raise ValueError("Structuretype representative cannot carry assemblies")
    if any(site.moment is not None for site in representative.wyckoff_sites):
        raise ValueError("Structuretype representative cannot carry site moments")
    species_by_name = {species.name: species for species in representative.species}
    occupations = tuple(
        WyckoffOccupation(site.wyckoff, species_by_name[site.species]) for site in representative.wyckoff_sites
    )
    return Protostructure(representative.spacegroup, occupations)


class Structuretype(StructuretypeBackend):
    """Store a protostructure refined by an assigned geometrical-class distinction.

    The class distinction is pinned by a canonical *representative* (a standard-setting
    :class:`~httk.atomistic.models.structure.asu.FundamentalDomainStructure` holding one exact
    realization used only as the class anchor, never as exact-structure data) and/or an
    externally assigned *discriminator* string. At least one is required; both may be given.
    Equality compares exactly the information present, so a representative-only value is never
    equal to a discriminator-only value. The discriminator names a species-independent class
    and is not part of the label.

    ``Structuretype`` is the assigned-species, geometrical-class cell of the
    material-information matrix:

    ======================  ==============  ==============
    Geometrical info        Anonymous       Assigned
    ======================  ==============  ==============
    Wyckoff positions only  Protopattern    Protostructure
    Geometrical class       Prototype       Structuretype
    Exact geometry          CrystalPattern  Structure
    ======================  ==============  ==============

    :param protostructure: The geometry-free protostructure, as any protostructure-like value.
        When omitted it is derived from ``representative``; it is required when only
        ``discriminator`` is given.
    :param representative: The canonical class representative, if one is known.
    :param discriminator: The externally assigned class discriminator, if one is known.
    """

    _protostructure: Protostructure
    _representative: FundamentalDomainStructure | None
    _discriminator: str | None
    kind: ClassVar[str] = "structuretype"

    def __init__(
        self,
        protostructure: "ProtostructureLike | None" = None,
        *,
        representative: FundamentalDomainStructure | None = None,
        discriminator: str | None = None,
    ) -> None:
        if representative is None and discriminator is None:
            raise ValueError("Structuretype requires at least one of representative or discriminator")
        if discriminator is not None and (not isinstance(discriminator, str) or not discriminator):
            raise ValueError("Structuretype discriminator must be a non-empty string when given")
        derived = None if representative is None else _protostructure_from_representative(representative)
        if protostructure is not None:
            from httk.atomistic.models.protostructure.view import ProtostructureView

            held = ProtostructureView(protostructure).unview()
            if derived is not None and held != derived:
                raise ValueError("Structuretype protostructure disagrees with its representative's occupations")
            result = held
        elif derived is not None:
            result = derived
        else:
            raise ValueError("Structuretype needs a protostructure when only a discriminator is given")
        self._protostructure = result
        self._representative = representative
        self._discriminator = discriminator

    @property
    def protostructure(self) -> Protostructure:
        """Return the geometry-free protostructure this structuretype refines."""
        return self._protostructure

    @property
    def representative(self) -> FundamentalDomainStructure | None:
        """Return the canonical class representative, if one is held."""
        return self._representative

    @property
    def discriminator(self) -> str | None:
        """Return the externally assigned class discriminator, if one is held."""
        return self._discriminator

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Structuretype):
            return NotImplemented
        return (self._protostructure, self._representative, self._discriminator) == (
            other._protostructure,
            other._representative,
            other._discriminator,
        )

    __hash__ = None  # type: ignore[assignment]  # representative is unhashable; a structuretype is a value key by equality only

    def __repr__(self) -> str:
        parts = [repr(self._protostructure.label)]
        if self._representative is not None:
            parts.append("representative=...")
        if self._discriminator is not None:
            parts.append(f"discriminator={self._discriminator!r}")
        return f"Structuretype({', '.join(parts)})"
