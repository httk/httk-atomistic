"""The assigned-species protonetstructure interface."""

from abc import ABC, abstractmethod
from itertools import groupby
from typing import TYPE_CHECKING, Self, cast

from httk.atomistic.models.formula.formula_view import ChemicalFormulaView
from httk.atomistic.models.formula.formulatype_view import FormulatypeView
from httk.atomistic.models.prototype.notation import pearson_symbol, render_protonet

if TYPE_CHECKING:
    from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure
    from httk.atomistic.models.protonetstructure.backend import ProtonetstructureBackend
    from httk.atomistic.models.protonetstructure.label import ProtonetstructureLabel
    from httk.atomistic.models.protonettype.protonettype import Protonettype
    from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
    from httk.atomistic.symmetry.spacegroup import Spacegroup


class ProtonetstructureAPI(ABC):
    """Common interface for standard-setting occupied Wyckoff positions with a bonding net.

    Site ``n`` of the net is ``occupations[n-1]``. Composition and formula derivations use
    the standard-setting conventional-cell scale.
    """

    @property
    @abstractmethod
    def spacegroup(self) -> "Spacegroup":
        """Return the standard-setting space group."""
        raise NotImplementedError

    @property
    @abstractmethod
    def occupations(self) -> tuple["WyckoffOccupation", ...]:
        """Return the occupied Wyckoff positions in site order."""
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
    def formula(self) -> ChemicalFormulaView:
        """Return a reduced formula at the standard conventional-cell scale."""
        return ChemicalFormulaView(cast("ProtonetstructureBackend", self))

    @property
    def anonymous_formula(self) -> FormulatypeView:
        """Return the Formulatype of the species-class site counts, the label prefix."""
        return FormulatypeView(cast("ProtonetstructureBackend", self))

    @property
    def label(self) -> "ProtonetstructureLabel":
        """Return the protonetstructure label presentation.

        Classes keep this value's order, which the net's site numbering depends on, so the
        unsuffixed bare part can differ from the canonical bare protostructure label when
        classes tie.
        """
        from httk.atomistic.models.protonetstructure.label import ProtonetstructureLabel

        return ProtonetstructureLabel(cast("ProtonetstructureBackend", self))

    @property
    def bare_protostructure(self) -> "BareProtostructure":
        """Return the canonical bare protostructure of these occupations, without the net.

        This is not a string cut of :attr:`label`: classes there follow the net, so its bare
        part can differ from the canonical bare protostructure label when classes tie.
        """
        from httk.atomistic.models.bareprotostructure.view import BareProtostructureView

        return BareProtostructureView(self).unview()

    @property
    def protonettype(self) -> "Protonettype":
        """Return the anonymous erasure, labelling classes by position."""
        from httk.atomistic.models.protonettype.derived import DerivedProtonettype

        return DerivedProtonettype(self).resolve()

    @property
    def protonetstructure(self) -> Self:
        """Return this protonetstructure value."""
        return self

    def _label_text(self) -> str:
        runs = [
            (name, tuple(o.wyckoff for o in run))
            for name, run in groupby(self.occupations, key=lambda o: o.species.name)
        ]
        return render_protonet(self.spacegroup, [letters for _, letters in runs], self.net, [name for name, _ in runs])
