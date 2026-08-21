"""The geometry-free prototemplate interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, cast

from httk.atomistic.models.formula.formulatemplate_view import FormulatemplateView
from httk.atomistic.models.prototemplate.notation import pearson_symbol, render_prototemplate_label

if TYPE_CHECKING:
    from httk.atomistic.models.prototemplate.backend import PrototemplateBackend
    from httk.atomistic.models.prototemplate.label import PrototemplateLabel
    from httk.atomistic.models.prototemplate.occupation import PrototemplateOccupation
    from httk.atomistic.symmetry.spacegroup import Spacegroup


class PrototemplateAPI(ABC):
    """The common interface for a space group with class-partitioned Wyckoff letters.

    A prototemplate is the information content of an unsuffixed AFLOW-style label: a space
    group, its occupied Wyckoff letters, and the partition of those occupations into
    anonymous species classes. It carries no chemical elements and no continuous degrees
    of freedom. All derivations use the standard-setting conventional-cell scale.
    """

    @property
    @abstractmethod
    def spacegroup(self) -> "Spacegroup":
        """Return the standard-setting space group."""
        raise NotImplementedError

    @property
    @abstractmethod
    def occupations(self) -> tuple["PrototemplateOccupation", ...]:
        """Return the occupied Wyckoff positions in canonical order."""
        raise NotImplementedError

    def multiplicities(self) -> tuple[int, ...]:
        """Return tabulated standard-setting multiplicities for each occupation.

        :return: The standard-setting multiplicity for each occupation.
        """
        return tuple(
            self.spacegroup.wyckoff_position(occupation.wyckoff).multiplicity for occupation in self.occupations
        )

    @property
    def nsites_conventional(self) -> int:
        """Return the total number of sites in the standard conventional cell.

        :return: The conventional-cell site count.
        """
        return sum(self.multiplicities())

    @property
    def pearson_symbol(self) -> str:
        """Return the Pearson symbol at the standard conventional-cell scale.

        :return: The Pearson symbol, such as ``"cF8"``.
        """
        return pearson_symbol(self.spacegroup, self.nsites_conventional)

    @property
    def label(self) -> "PrototemplateLabel":
        """Return the httk prototemplate label of this template.

        Any faithful render is the prototemplate label; the *canonical* prototemplate label
        is the one obtained from a normalizer-canonical template.

        :return: The prototemplate label view.
        """
        from httk.atomistic.models.prototemplate.label import PrototemplateLabel

        return PrototemplateLabel(cast("PrototemplateBackend", self))

    @property
    def anonymous_formula(self) -> FormulatemplateView:
        """Return the reduced anonymous formula at the standard conventional-cell scale.

        The anonymous amounts are the per-class conventional multiplicities. This differs
        from the label's own anonymous prefix, which follows group order rather than the
        canonical non-increasing formula order.

        :return: The conventional-cell anonymous formula view.
        """
        return FormulatemplateView(cast("PrototemplateBackend", self))

    @property
    def prototemplate(self) -> Self:
        """Return this prototemplate value."""
        return self

    def _template_label_text(self) -> str:
        """Return the raw prototemplate label string, used by the label view."""
        return render_prototemplate_label(
            self.spacegroup, [(occupation.wyckoff, occupation.label) for occupation in self.occupations]
        )
