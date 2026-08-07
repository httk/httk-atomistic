"""The geometry-free protostructure interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, cast

from httk.atomistic.models.formula.anonymous_view import AnonymousFormulaView
from httk.atomistic.models.formula.formula_view import ChemicalFormulaView

if TYPE_CHECKING:
    from httk.atomistic.models.protostructure.backend import ProtostructureBackend
    from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
    from httk.atomistic.symmetry.spacegroup import Spacegroup


class ProtostructureAPI(ABC):
    """The common interface for standard-setting occupied Wyckoff positions."""

    @property
    @abstractmethod
    def spacegroup(self) -> "Spacegroup":
        raise NotImplementedError

    @property
    @abstractmethod
    def occupations(self) -> tuple["WyckoffOccupation", ...]:
        raise NotImplementedError

    def multiplicities(self) -> tuple[int, ...]:
        return tuple(
            self.spacegroup.wyckoff_position(occupation.wyckoff).multiplicity for occupation in self.occupations
        )

    @property
    def nsites_conventional(self) -> int:
        return sum(self.multiplicities())

    @property
    def formula(self) -> ChemicalFormulaView:
        return ChemicalFormulaView(cast("ProtostructureBackend", self))

    @property
    def anonymous_formula(self) -> AnonymousFormulaView:
        return AnonymousFormulaView(cast("ProtostructureBackend", self))

    @property
    def protostructure(self) -> Self:
        return self
