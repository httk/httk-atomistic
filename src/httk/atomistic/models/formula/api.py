"""
The minimal canonical chemical-formula interface for httk-atomistic.
"""

from abc import ABC, abstractmethod
from fractions import Fraction

from httk.atomistic.models.formula.diagnostics import CompositionDiagnostic


class ChemicalFormulaAPI(ABC):
    """
    Abstract base class for the canonical chemical-formula interface.

    The concrete defaults keep backends that do not have provenance for optional
    formula metadata interoperable without requiring them to implement every accessor.
    """

    @property
    @abstractmethod
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:
        """Ordered ``(label, exact amount)`` pairs for this chemical formula.

        Labels are element symbols in alphabetical order unless ``is_anonymous``, in
        which case they are OPTIMADE anonymous symbols in canonical
        (descending-coefficient) order.
        """
        raise NotImplementedError

    @property
    def is_anonymous(self) -> bool:
        """Whether the formula labels are OPTIMADE anonymous symbols.

        A backend that carries anonymous notation overrides this; a backend without
        that distinction inherits ``False``.
        """
        return False

    @property
    def uncertainties(self) -> tuple[tuple[str, Fraction | None], ...]:
        """Per-amount absolute uncertainties, or ``None`` where exactness is unstated."""
        return tuple((label, None) for label, _ in self.amounts)

    @property
    def complete(self) -> bool:
        """Whether the formula accounts for all represented elemental material."""
        return True

    @property
    def exact(self) -> bool:
        """Whether all formula amounts are exact rather than precision-bounded."""
        return all(uncertainty is None for _, uncertainty in self.uncertainties)

    @property
    def normalized(self) -> bool:
        """Whether the formula is normalized within the stated precision."""
        return True

    @property
    def normalization_status(self) -> str:
        """The formula's normalization status derived from exactness and normalization."""
        if self.exact and self.normalized:
            return "exact"
        if self.normalized:
            return "within_precision"
        return "outside_precision"

    @property
    def diagnostics(self) -> tuple[CompositionDiagnostic, ...]:
        """Non-fatal structured diagnostics associated with the formula."""
        return ()
