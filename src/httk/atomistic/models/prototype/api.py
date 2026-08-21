"""The anonymous geometrical-class prototype interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, cast

if TYPE_CHECKING:
    from httk.atomistic.models.chromastructure.fundamental import FundamentalDomainPattern
    from httk.atomistic.models.formula.chromaformula_view import ChromaformulaView
    from httk.atomistic.models.protochroma.backend import ProtochromaBackend
    from httk.atomistic.models.protochroma.label import ProtochromaLabel
    from httk.atomistic.models.protochroma.protochroma import Protochroma
    from httk.atomistic.symmetry.spacegroup import Spacegroup


class PrototypeAPI(ABC):
    """The common interface for a protochroma refined by a geometrical class.

    A prototype is a :class:`~httk.atomistic.models.protochroma.protochroma.Protochroma`
    plus a geometrical-class distinction. The class is pinned by a canonical *representative*
    (a standard-setting :class:`~httk.atomistic.models.chromastructure.fundamental.FundamentalDomainPattern`
    used only as the class anchor, never as exact-structure data) and/or an externally assigned
    *discriminator* string. At least one is present; equality compares exactly the information
    present. The discriminator is species-independent and is not part of the label. All
    label, Pearson, and formula derivations delegate to the protochroma.
    """

    @property
    @abstractmethod
    def protochroma(self) -> "Protochroma":
        """Return the anonymous protochroma this prototype refines."""
        raise NotImplementedError

    @property
    @abstractmethod
    def representative(self) -> "FundamentalDomainPattern | None":
        """Return the canonical class representative, if one is held."""
        raise NotImplementedError

    @property
    @abstractmethod
    def discriminator(self) -> str | None:
        """Return the externally assigned class discriminator, if one is held."""
        raise NotImplementedError

    @property
    def spacegroup(self) -> "Spacegroup":
        """Return the standard-setting space group of the protochroma.

        :return: The standard-setting space group.
        """
        return self.protochroma.spacegroup

    @property
    def label(self) -> "ProtochromaLabel":
        """Return the httk protochroma label of this prototype's protochroma.

        The discriminator names the geometrical class and is not part of the label. Any
        faithful render is the protochroma label; the *canonical* protochroma label comes
        from a normalizer-canonical protochroma.

        :return: The protochroma label view.
        """
        from httk.atomistic.models.protochroma.label import ProtochromaLabel

        return ProtochromaLabel(cast("ProtochromaBackend", self.protochroma))

    @property
    def pearson_symbol(self) -> str:
        """Return the Pearson symbol at the standard conventional-cell scale.

        :return: The Pearson symbol, such as ``"cF8"``.
        """
        return self.protochroma.pearson_symbol

    @property
    def nsites_conventional(self) -> int:
        """Return the total number of sites in the standard conventional cell.

        :return: The conventional-cell site count.
        """
        return self.protochroma.nsites_conventional

    @property
    def anonymous_formula(self) -> "ChromaformulaView":
        """Return the reduced anonymous formula at the standard conventional-cell scale.

        :return: The conventional-cell anonymous formula view.
        """
        return self.protochroma.anonymous_formula

    @property
    def prototype(self) -> Self:
        """Return this prototype value."""
        return self
