"""The geometry-free protochroma interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, cast

from httk.atomistic.models.formula.chromaformula_view import ChromaformulaView
from httk.atomistic.models.protochroma.notation import pearson_symbol, render_protochroma_label

if TYPE_CHECKING:
    from httk.atomistic.models.protochroma.backend import ProtochromaBackend
    from httk.atomistic.models.protochroma.label import ProtochromaLabel
    from httk.atomistic.models.protochroma.occupation import ProtochromaOccupation
    from httk.atomistic.symmetry.spacegroup import Spacegroup


class ProtochromaAPI(ABC):
    """The common interface for a space group with class-partitioned Wyckoff letters.

    A protochroma is the information content of an unsuffixed AFLOW-style label: a space
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
    def occupations(self) -> tuple["ProtochromaOccupation", ...]:
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
    def label(self) -> "ProtochromaLabel":
        """Return the httk protochroma label of this protochroma.

        Any faithful render is the protochroma label; the *canonical* protochroma label
        is the one obtained from a normalizer-canonical protochroma.

        :return: The protochroma label view.
        """
        from httk.atomistic.models.protochroma.label import ProtochromaLabel

        return ProtochromaLabel(cast("ProtochromaBackend", self))

    @property
    def anonymous_formula(self) -> ChromaformulaView:
        """Return the reduced anonymous formula at the standard conventional-cell scale.

        The anonymous amounts are the per-class conventional multiplicities. This differs
        from the label's own anonymous prefix, which follows group order rather than the
        canonical non-increasing formula order.

        :return: The conventional-cell anonymous formula view.
        """
        return ChromaformulaView(cast("ProtochromaBackend", self))

    @property
    def protochroma(self) -> Self:
        """Return this protochroma value."""
        return self

    def _pattern_label_text(self) -> str:
        """Return the raw protochroma label string, used by the label view."""
        return render_protochroma_label(
            self.spacegroup, [(occupation.wyckoff, occupation.label) for occupation in self.occupations]
        )
