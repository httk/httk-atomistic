"""The anonymous Wyckoff-only prototype interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, cast

from httk.atomistic.models.formula.formulatype_view import FormulatypeView
from httk.atomistic.models.prototype.notation import pearson_symbol, render_prototype_label

if TYPE_CHECKING:
    from httk.atomistic.models.bareprototype.backend import BarePrototypeBackend
    from httk.atomistic.models.bareprototype.label import BarePrototypeLabel
    from httk.atomistic.models.prototype.occupation import PrototypeOccupation
    from httk.atomistic.symmetry.spacegroup import Spacegroup


class BarePrototypeAPI(ABC):
    """Common interface for anonymous standard-setting Wyckoff prototypes."""

    @property
    @abstractmethod
    def spacegroup(self) -> "Spacegroup":
        """Return the standard-setting space group."""
        raise NotImplementedError

    @property
    @abstractmethod
    def occupations(self) -> tuple["PrototypeOccupation", ...]:
        """Return canonical anonymous Wyckoff occupations."""
        raise NotImplementedError

    def multiplicities(self) -> tuple[int, ...]:
        """Return tabulated standard-setting multiplicities."""
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
        return FormulatypeView(cast("BarePrototypeBackend", self))

    @property
    def label(self) -> "BarePrototypeLabel":
        """Return the canonical Wyckoff label presentation."""
        from httk.atomistic.models.bareprototype.label import BarePrototypeLabel

        return BarePrototypeLabel(cast("BarePrototypeBackend", self))

    @property
    def bare_prototype(self) -> Self:
        """Return this bare prototype value."""
        return self

    def _prototype_label_text(self) -> str:
        return render_prototype_label(self.spacegroup, [(value.wyckoff, value.label) for value in self.occupations])
