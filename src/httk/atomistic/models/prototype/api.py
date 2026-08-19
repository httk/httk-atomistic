"""The minimal canonical anonymous-structure interface."""

from abc import ABC, abstractmethod
from fractions import Fraction
from typing import TYPE_CHECKING, cast

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.formula.formulapattern_view import FormulapatternView
from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species

if TYPE_CHECKING:
    from httk.atomistic.models.prototype.backend import AnonymousStructureBackend


class AnonymousStructureAPI(ABC):
    """The common interface for dummy-species structures and prototypes."""

    @property
    @abstractmethod
    def cell(self) -> Cell:
        """Return the structure cell."""
        raise NotImplementedError

    @property
    @abstractmethod
    def sites(self) -> Sites:
        """Return the reduced sites."""
        raise NotImplementedError

    @property
    @abstractmethod
    def species(self) -> tuple[Species, ...]:
        """Return the distinct dummy species."""
        raise NotImplementedError

    @property
    @abstractmethod
    def species_at_sites(self) -> tuple[str, ...]:
        """Return dummy species names in site order."""
        raise NotImplementedError

    @property
    def coordinate_precision(self) -> Fraction | None:
        """Return the reduced-coordinate precision, if known."""
        return None

    @property
    def basis_precision(self) -> Fraction | None:
        """Return the cell-basis precision, if known."""
        return None

    @property
    def anonymous_formula(self) -> FormulapatternView:
        """Return the canonical anonymous formula at the represented site scale."""
        # The API root is only ever hosted by a backend in this family; the formula bridge
        # accepts that backend root rather than the abstract API protocol itself.
        return FormulapatternView(cast("AnonymousStructureBackend", self))

    @property
    def is_canonical(self) -> bool:
        """Return whether site counts assign consecutive canonical anonymous labels."""
        counts: dict[str, int] = {}
        for label in self.species_at_sites:
            counts[label] = counts.get(label, 0) + 1
        ordered = sorted(counts, key=lambda label: (-counts[label], label))
        return all(label == anonymous_symbol(index) for index, label in enumerate(ordered))
