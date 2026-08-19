"""The geometry-free protopattern interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, cast

from httk.atomistic.models.formula.formulapattern_view import FormulapatternView
from httk.atomistic.models.protopattern.notation import pearson_symbol, render_protopattern_label

if TYPE_CHECKING:
    from httk.atomistic.models.protopattern.backend import ProtopatternBackend
    from httk.atomistic.models.protopattern.label import ProtopatternLabel
    from httk.atomistic.models.protopattern.occupation import ProtopatternOccupation
    from httk.atomistic.symmetry.spacegroup import Spacegroup


class ProtopatternAPI(ABC):
    """The common interface for a space group with class-partitioned Wyckoff letters.

    A protopattern is the information content of an unsuffixed AFLOW-style label: a space
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
    def occupations(self) -> tuple["ProtopatternOccupation", ...]:
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
    def label(self) -> "ProtopatternLabel":
        """Return the httk protopattern label of this pattern.

        Any faithful render is the protopattern label; the *canonical* protopattern label
        is the one obtained from a normalizer-canonical pattern.

        :return: The protopattern label view.
        """
        from httk.atomistic.models.protopattern.label import ProtopatternLabel

        return ProtopatternLabel(cast("ProtopatternBackend", self))

    @property
    def anonymous_formula(self) -> FormulapatternView:
        """Return the reduced anonymous formula at the standard conventional-cell scale.

        The anonymous amounts are the per-class conventional multiplicities. This differs
        from the label's own anonymous prefix, which follows group order rather than the
        canonical non-increasing formula order.

        :return: The conventional-cell anonymous formula view.
        """
        return FormulapatternView(cast("ProtopatternBackend", self))

    @property
    def protopattern(self) -> Self:
        """Return this protopattern value."""
        return self

    def _pattern_label_text(self) -> str:
        """Return the raw protopattern label string, used by the label view."""
        return render_protopattern_label(
            self.spacegroup, [(occupation.wyckoff, occupation.label) for occupation in self.occupations]
        )
