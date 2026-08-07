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
    """The common interface for standard-setting occupied Wyckoff positions.

    Composition and formula derivations from this interface use the standard-setting
    conventional-cell scale, independently of the setting or transform of a source
    structure from which a protostructure was recognized.
    """

    @property
    @abstractmethod
    def spacegroup(self) -> "Spacegroup":
        raise NotImplementedError

    @property
    @abstractmethod
    def occupations(self) -> tuple["WyckoffOccupation", ...]:
        raise NotImplementedError

    def multiplicities(self) -> tuple[int, ...]:
        """Return tabulated standard-setting multiplicities for each occupation.

        These are deliberately not multiplicities from a source structure's transform;
        they define the provenance-independent conventional-cell scale of this value.
        """
        return tuple(
            self.spacegroup.wyckoff_position(occupation.wyckoff).multiplicity for occupation in self.occupations
        )

    @property
    def nsites_conventional(self) -> int:
        """Return the total number of sites in the standard conventional cell."""
        return sum(self.multiplicities())

    @property
    def formula(self) -> ChemicalFormulaView:
        """Return a reduced formula at the standard conventional-cell scale."""
        return ChemicalFormulaView(cast("ProtostructureBackend", self))

    @property
    def anonymous_formula(self) -> AnonymousFormulaView:
        """Return a reduced anonymous formula at the standard conventional-cell scale."""
        return AnonymousFormulaView(cast("ProtostructureBackend", self))

    @property
    def protostructure(self) -> Self:
        return self
