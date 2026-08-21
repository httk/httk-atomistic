"""The assigned geometrical-class crystallotype interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, cast

if TYPE_CHECKING:
    from httk.atomistic.models.protochroma.protochroma import Protochroma
    from httk.atomistic.models.protostructure.backend import ProtostructureBackend
    from httk.atomistic.models.protostructure.label import ProtostructureLabel
    from httk.atomistic.models.protostructure.protostructure import Protostructure
    from httk.atomistic.models.structure.asu import FundamentalDomainStructure
    from httk.atomistic.symmetry.spacegroup import Spacegroup


class CrystallotypeAPI(ABC):
    """The common interface for a protostructure refined by a geometrical class.

    A crystallotype is a :class:`~httk.atomistic.models.protostructure.protostructure.Protostructure`
    plus a geometrical-class distinction. The class is pinned by a canonical *representative*
    (a standard-setting :class:`~httk.atomistic.models.structure.asu.FundamentalDomainStructure`
    used only as the class anchor, never as exact-structure data) and/or an externally assigned
    *discriminator* string. At least one is present; equality compares exactly the information
    present. The discriminator is species-independent and is not part of the label. Label and
    ``aflow_label`` derivations delegate to the protostructure; ``pearson_symbol`` and
    ``protochroma`` come from the erased anonymous form.
    """

    @property
    @abstractmethod
    def protostructure(self) -> "Protostructure":
        """Return the geometry-free protostructure this crystallotype refines."""
        raise NotImplementedError

    @property
    @abstractmethod
    def representative(self) -> "FundamentalDomainStructure | None":
        """Return the canonical class representative, if one is held."""
        raise NotImplementedError

    @property
    @abstractmethod
    def discriminator(self) -> str | None:
        """Return the externally assigned class discriminator, if one is held."""
        raise NotImplementedError

    @property
    def spacegroup(self) -> "Spacegroup":
        """Return the standard-setting space group of the protostructure.

        :return: The standard-setting space group.
        """
        return self.protostructure.spacegroup

    @property
    def protochroma(self) -> "Protochroma":
        """Return the anonymous protochroma the protostructure erases to.

        :return: The erased protochroma value.
        """
        from httk.atomistic.models.protochroma.derived import DerivedProtochroma

        return DerivedProtochroma(cast("ProtostructureBackend", self.protostructure)).resolve()

    @property
    def label(self) -> "ProtostructureLabel":
        """Return the httk protostructure label of this crystallotype.

        The discriminator names the geometrical class and is not part of the label. Any
        faithful render is the protostructure label; the *canonical* protostructure label
        comes from a normalizer-canonical protostructure.

        :return: The protostructure label view.
        """
        from httk.atomistic.models.protostructure.label import ProtostructureLabel

        return ProtostructureLabel(cast("ProtostructureBackend", self.protostructure))

    @property
    def aflow_label(self) -> str:
        """Return the AFLOW-style label of this crystallotype's protostructure.

        :return: The AFLOW-style label text.
        """
        return self.protostructure.aflow_label

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
        return self.protostructure.nsites_conventional

    @property
    def crystallotype(self) -> Self:
        """Return this crystallotype value."""
        return self
