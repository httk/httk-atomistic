"""The anonymous geometrical-class prototype interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, cast

if TYPE_CHECKING:
    from httk.atomistic.models.crystalpattern.fundamental import FundamentalDomainPattern
    from httk.atomistic.models.formula.formulapattern_view import FormulapatternView
    from httk.atomistic.models.protopattern.backend import ProtopatternBackend
    from httk.atomistic.models.protopattern.label import ProtopatternLabel
    from httk.atomistic.models.protopattern.protopattern import Protopattern
    from httk.atomistic.symmetry.spacegroup import Spacegroup


class PrototypeAPI(ABC):
    """The common interface for a protopattern refined by a geometrical class.

    A prototype is a :class:`~httk.atomistic.models.protopattern.protopattern.Protopattern`
    plus a geometrical-class distinction. The class is pinned by a canonical *representative*
    (a standard-setting :class:`~httk.atomistic.models.crystalpattern.fundamental.FundamentalDomainPattern`
    used only as the class anchor, never as exact-structure data) and/or an externally assigned
    *discriminator* string. At least one is present; equality compares exactly the information
    present. The discriminator is species-independent and is not part of the label. All
    label, Pearson, and formula derivations delegate to the protopattern.
    """

    @property
    @abstractmethod
    def protopattern(self) -> "Protopattern":
        """Return the anonymous protopattern this prototype refines."""
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
        """Return the standard-setting space group of the protopattern.

        :return: The standard-setting space group.
        """
        return self.protopattern.spacegroup

    @property
    def label(self) -> "ProtopatternLabel":
        """Return the httk protopattern label of this prototype's pattern.

        The discriminator names the geometrical class and is not part of the label. Any
        faithful render is the protopattern label; the *canonical* protopattern label comes
        from a normalizer-canonical pattern.

        :return: The protopattern label view.
        """
        from httk.atomistic.models.protopattern.label import ProtopatternLabel

        return ProtopatternLabel(cast("ProtopatternBackend", self.protopattern))

    @property
    def pearson_symbol(self) -> str:
        """Return the Pearson symbol at the standard conventional-cell scale.

        :return: The Pearson symbol, such as ``"cF8"``.
        """
        return self.protopattern.pearson_symbol

    @property
    def nsites_conventional(self) -> int:
        """Return the total number of sites in the standard conventional cell.

        :return: The conventional-cell site count.
        """
        return self.protopattern.nsites_conventional

    @property
    def anonymous_formula(self) -> "FormulapatternView":
        """Return the reduced anonymous formula at the standard conventional-cell scale.

        :return: The conventional-cell anonymous formula view.
        """
        return self.protopattern.anonymous_formula

    @property
    def prototype(self) -> Self:
        """Return this prototype value."""
        return self
