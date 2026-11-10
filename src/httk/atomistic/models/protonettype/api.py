"""The anonymous protonettype interface."""

from abc import ABC, abstractmethod
from itertools import groupby
from typing import TYPE_CHECKING, Self, cast

from httk.atomistic.models.formula.formulatype_view import FormulatypeView
from httk.atomistic.models.prototype.notation import pearson_symbol, render_protonet

if TYPE_CHECKING:
    from httk.atomistic.models.bareprototype.bareprototype import BarePrototype
    from httk.atomistic.models.protonettype.backend import ProtonettypeBackend
    from httk.atomistic.models.protonettype.label import ProtonettypeLabel
    from httk.atomistic.models.prototype.occupation import PrototypeOccupation
    from httk.atomistic.symmetry.spacegroup import Spacegroup


class ProtonettypeAPI(ABC):
    """Common interface for anonymous standard-setting Wyckoff prototypes with a bonding net.

    Site ``n`` of the net is ``occupations[n-1]``.
    """

    @property
    @abstractmethod
    def spacegroup(self) -> "Spacegroup":
        """Return the standard-setting space group."""
        raise NotImplementedError

    @property
    @abstractmethod
    def occupations(self) -> tuple["PrototypeOccupation", ...]:
        """Return the anonymous Wyckoff occupations in site order."""
        raise NotImplementedError

    @property
    @abstractmethod
    def net(self) -> tuple[tuple[int, int, int, int, int, int], ...]:
        """Return the net rows ``(i, j, k, n1, n2, n3)`` in label order."""
        raise NotImplementedError

    def multiplicities(self) -> tuple[int, ...]:
        """Return tabulated standard-setting multiplicities in site order."""
        return tuple(self.spacegroup.wyckoff_position(value.wyckoff).multiplicity for value in self.occupations)

    @property
    def nsites_conventional(self) -> int:
        """Return the number of sites in the conventional cell."""
        return sum(self.multiplicities())

    @property
    def pearson_symbol(self) -> str:
        """Return the Pearson symbol for the conventional cell."""
        return pearson_symbol(self.spacegroup, self.nsites_conventional)

    @property
    def anonymous_formula(self) -> FormulatypeView:
        """Return the reduced anonymous formula view."""
        return FormulatypeView(cast("ProtonettypeBackend", self))

    @property
    def label(self) -> "ProtonettypeLabel":
        """Return the protonettype label presentation."""
        from httk.atomistic.models.protonettype.label import ProtonettypeLabel

        return ProtonettypeLabel(cast("ProtonettypeBackend", self))

    @property
    def bare_prototype(self) -> "BarePrototype":
        """Return the bare prototype of these occupations, without the net."""
        from httk.atomistic.models.bareprototype.bareprototype import BarePrototype

        return BarePrototype(self.spacegroup, self.occupations)

    @property
    def protonettype(self) -> Self:
        """Return this protonettype value."""
        return self

    def _label_text(self) -> str:
        letters = [tuple(o.wyckoff for o in run) for _, run in groupby(self.occupations, key=lambda o: o.label)]
        return render_protonet(self.spacegroup, letters, self.net)
