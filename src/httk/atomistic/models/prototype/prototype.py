"""The immutable anonymous geometrical-class prototype value."""

from typing import TYPE_CHECKING, ClassVar

from httk.atomistic.models.crystaltemplate.fundamental import FundamentalDomainTemplate
from httk.atomistic.models.prototype.backend import PrototypeBackend

if TYPE_CHECKING:
    from httk.atomistic.models.prototemplate.like import PrototemplateLike
    from httk.atomistic.models.prototemplate.prototemplate import Prototemplate


class Prototype(PrototypeBackend):
    """Store a prototemplate refined by an anonymous geometrical-class distinction.

    The class distinction is pinned by a canonical *representative* (a standard-setting
    :class:`~httk.atomistic.models.crystaltemplate.fundamental.FundamentalDomainTemplate`
    holding one exact realization used only as the class anchor, never as exact-structure
    data) and/or an externally assigned *discriminator* string. At least one is required;
    both may be given. Equality compares exactly the information present, so a
    representative-only value is never equal to a discriminator-only value. The discriminator
    names a species-independent class and is not part of the label.

    ``Prototype`` is the anonymous-species, geometrical-class cell of the
    material-information matrix:

    ======================  ===============  ==============
    Geometrical info        Anonymous        Assigned
    ======================  ===============  ==============
    Wyckoff positions only  Prototemplate    Protostructure
    Geometrical class       Prototype        Structuretype
    Exact geometry          CrystalTemplate  Structure
    ======================  ===============  ==============

    :param prototemplate: The anonymous prototemplate, as any prototemplate-like value. When
        omitted it is derived from ``representative``; it is required when only
        ``discriminator`` is given.
    :param representative: The canonical class representative, if one is known.
    :param discriminator: The externally assigned class discriminator, if one is known.
    """

    _prototemplate: "Prototemplate"
    _representative: FundamentalDomainTemplate | None
    _discriminator: str | None
    kind: ClassVar[str] = "prototype"

    def __init__(
        self,
        prototemplate: "PrototemplateLike | None" = None,
        *,
        representative: FundamentalDomainTemplate | None = None,
        discriminator: str | None = None,
    ) -> None:
        if representative is None and discriminator is None:
            raise ValueError("Prototype requires at least one of representative or discriminator")
        if discriminator is not None and (not isinstance(discriminator, str) or not discriminator):
            raise ValueError("Prototype discriminator must be a non-empty string when given")
        if representative is not None:
            if not isinstance(representative, FundamentalDomainTemplate):
                raise TypeError("Prototype representative must be a FundamentalDomainTemplate")
            unview = getattr(representative, "unview", None)
            if unview is not None:  # normalize a lazy FundamentalDomainTemplateView to its plain value
                representative = unview()
        if prototemplate is not None:
            from httk.atomistic.models.prototemplate.view import PrototemplateView

            template = PrototemplateView(prototemplate).unview()
            if representative is not None and template != representative.prototemplate:
                raise ValueError("Prototype prototemplate disagrees with its representative's prototemplate")
        elif representative is not None:
            template = representative.prototemplate
        else:
            raise ValueError("Prototype needs a prototemplate when only a discriminator is given")
        self._prototemplate = template
        self._representative = representative
        self._discriminator = discriminator

    @property
    def prototemplate(self) -> "Prototemplate":
        """Return the anonymous prototemplate this prototype refines."""
        return self._prototemplate

    @property
    def representative(self) -> FundamentalDomainTemplate | None:
        """Return the canonical class representative, if one is held."""
        return self._representative

    @property
    def discriminator(self) -> str | None:
        """Return the externally assigned class discriminator, if one is held."""
        return self._discriminator

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Prototype):
            return NotImplemented
        return (self._prototemplate, self._representative, self._discriminator) == (
            other._prototemplate,
            other._representative,
            other._discriminator,
        )

    __hash__ = None  # type: ignore[assignment]  # representative is unhashable; a prototype is a value key by equality only

    def __repr__(self) -> str:
        parts = [repr(self._prototemplate.label)]
        if self._representative is not None:
            parts.append("representative=...")
        if self._discriminator is not None:
            parts.append(f"discriminator={self._discriminator!r}")
        return f"Prototype({', '.join(parts)})"
