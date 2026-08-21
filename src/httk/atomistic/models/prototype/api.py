"""The anonymous geometrical-class prototype interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, cast

if TYPE_CHECKING:
    from httk.atomistic.models.crystaltemplate.fundamental import FundamentalDomainTemplate
    from httk.atomistic.models.formula.formulatemplate_view import FormulatemplateView
    from httk.atomistic.models.prototemplate.backend import PrototemplateBackend
    from httk.atomistic.models.prototemplate.label import PrototemplateLabel
    from httk.atomistic.models.prototemplate.prototemplate import Prototemplate
    from httk.atomistic.symmetry.spacegroup import Spacegroup


class PrototypeAPI(ABC):
    """The common interface for a prototemplate refined by a geometrical class.

    A prototype is a :class:`~httk.atomistic.models.prototemplate.prototemplate.Prototemplate`
    plus a geometrical-class distinction. The class is pinned by a canonical *representative*
    (a standard-setting :class:`~httk.atomistic.models.crystaltemplate.fundamental.FundamentalDomainTemplate`
    used only as the class anchor, never as exact-structure data) and/or an externally assigned
    *discriminator* string. At least one is present; equality compares exactly the information
    present. The discriminator is species-independent and is not part of the label. All
    label, Pearson, and formula derivations delegate to the prototemplate.
    """

    @property
    @abstractmethod
    def prototemplate(self) -> "Prototemplate":
        """Return the anonymous prototemplate this prototype refines."""
        raise NotImplementedError

    @property
    @abstractmethod
    def representative(self) -> "FundamentalDomainTemplate | None":
        """Return the canonical class representative, if one is held."""
        raise NotImplementedError

    @property
    @abstractmethod
    def discriminator(self) -> str | None:
        """Return the externally assigned class discriminator, if one is held."""
        raise NotImplementedError

    @property
    def spacegroup(self) -> "Spacegroup":
        """Return the standard-setting space group of the prototemplate.

        :return: The standard-setting space group.
        """
        return self.prototemplate.spacegroup

    @property
    def label(self) -> "PrototemplateLabel":
        """Return the httk prototemplate label of this prototype's template.

        The discriminator names the geometrical class and is not part of the label. Any
        faithful render is the prototemplate label; the *canonical* prototemplate label comes
        from a normalizer-canonical template.

        :return: The prototemplate label view.
        """
        from httk.atomistic.models.prototemplate.label import PrototemplateLabel

        return PrototemplateLabel(cast("PrototemplateBackend", self.prototemplate))

    @property
    def pearson_symbol(self) -> str:
        """Return the Pearson symbol at the standard conventional-cell scale.

        :return: The Pearson symbol, such as ``"cF8"``.
        """
        return self.prototemplate.pearson_symbol

    @property
    def nsites_conventional(self) -> int:
        """Return the total number of sites in the standard conventional cell.

        :return: The conventional-cell site count.
        """
        return self.prototemplate.nsites_conventional

    @property
    def anonymous_formula(self) -> "FormulatemplateView":
        """Return the reduced anonymous formula at the standard conventional-cell scale.

        :return: The conventional-cell anonymous formula view.
        """
        return self.prototemplate.anonymous_formula

    @property
    def prototype(self) -> Self:
        """Return this prototype value."""
        return self
