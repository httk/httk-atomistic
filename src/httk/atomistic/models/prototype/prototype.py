"""The immutable anonymous geometrical-class prototype value."""

from typing import TYPE_CHECKING, ClassVar

from httk.atomistic.models.crystalpattern.fundamental import FundamentalDomainPattern
from httk.atomistic.models.prototype.backend import PrototypeBackend

if TYPE_CHECKING:
    from httk.atomistic.models.protopattern.like import ProtopatternLike
    from httk.atomistic.models.protopattern.protopattern import Protopattern


class Prototype(PrototypeBackend):
    """Store a protopattern refined by an anonymous geometrical-class distinction.

    The class distinction is pinned by a canonical *representative* (a standard-setting
    :class:`~httk.atomistic.models.crystalpattern.fundamental.FundamentalDomainPattern`
    holding one exact realization used only as the class anchor, never as exact-structure
    data) and/or an externally assigned *discriminator* string. At least one is required;
    both may be given. Equality compares exactly the information present, so a
    representative-only value is never equal to a discriminator-only value. The discriminator
    names a species-independent class and is not part of the label.

    ``Prototype`` is the anonymous-species, geometrical-class cell of the
    material-information matrix:

    ======================  ==============  ==============
    Geometrical info        Anonymous       Assigned
    ======================  ==============  ==============
    Wyckoff positions only  Protopattern    Protostructure
    Geometrical class       Prototype       Structuretype
    Exact geometry          CrystalPattern  Structure
    ======================  ==============  ==============

    :param protopattern: The anonymous protopattern, as any protopattern-like value. When
        omitted it is derived from ``representative``; it is required when only
        ``discriminator`` is given.
    :param representative: The canonical class representative, if one is known.
    :param discriminator: The externally assigned class discriminator, if one is known.
    """

    _protopattern: "Protopattern"
    _representative: FundamentalDomainPattern | None
    _discriminator: str | None
    kind: ClassVar[str] = "prototype"

    def __init__(
        self,
        protopattern: "ProtopatternLike | None" = None,
        *,
        representative: FundamentalDomainPattern | None = None,
        discriminator: str | None = None,
    ) -> None:
        if representative is None and discriminator is None:
            raise ValueError("Prototype requires at least one of representative or discriminator")
        if discriminator is not None and (not isinstance(discriminator, str) or not discriminator):
            raise ValueError("Prototype discriminator must be a non-empty string when given")
        if representative is not None:
            if not isinstance(representative, FundamentalDomainPattern):
                raise TypeError("Prototype representative must be a FundamentalDomainPattern")
            unview = getattr(representative, "unview", None)
            if unview is not None:  # normalize a lazy FundamentalDomainPatternView to its plain value
                representative = unview()
        if protopattern is not None:
            from httk.atomistic.models.protopattern.view import ProtopatternView

            pattern = ProtopatternView(protopattern).unview()
            if representative is not None and pattern != representative.protopattern:
                raise ValueError("Prototype protopattern disagrees with its representative's protopattern")
        elif representative is not None:
            pattern = representative.protopattern
        else:
            raise ValueError("Prototype needs a protopattern when only a discriminator is given")
        self._protopattern = pattern
        self._representative = representative
        self._discriminator = discriminator

    @property
    def protopattern(self) -> "Protopattern":
        """Return the anonymous protopattern this prototype refines."""
        return self._protopattern

    @property
    def representative(self) -> FundamentalDomainPattern | None:
        """Return the canonical class representative, if one is held."""
        return self._representative

    @property
    def discriminator(self) -> str | None:
        """Return the externally assigned class discriminator, if one is held."""
        return self._discriminator

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Prototype):
            return NotImplemented
        return (self._protopattern, self._representative, self._discriminator) == (
            other._protopattern,
            other._representative,
            other._discriminator,
        )

    __hash__ = None  # type: ignore[assignment]  # representative is unhashable; a prototype is a value key by equality only

    def __repr__(self) -> str:
        parts = [repr(self._protopattern.label)]
        if self._representative is not None:
            parts.append("representative=...")
        if self._discriminator is not None:
            parts.append(f"discriminator={self._discriminator!r}")
        return f"Prototype({', '.join(parts)})"
