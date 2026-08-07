"""Backend wrapping a raw elemental-composition mapping."""

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from httk.atomistic.elements import SYMBOLS
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.composition import Composition
from httk.atomistic.models.formula.diagnostics import CompositionDiagnostic

_ELEMENTS = frozenset(SYMBOLS)


class PlainComposition(ChemicalFormulaBackend):
    r"""Wrap a mapping of real element symbols to composition amounts.

    :param obj: The elemental amount mapping.
    :param \*\*hints: Backend-selection hints.
    """

    _raw: Mapping[str, Any]
    _composition: Composition

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "plain") != "plain":
            return None
        if not isinstance(obj, Mapping) or not all(isinstance(key, str) and key in _ELEMENTS for key in obj):
            return None
        return super().__new__(cls)

    def __init__(self, obj: Mapping[str, Any], **hints: Any) -> None:
        self._raw = obj
        self._composition = Composition(obj)

    @property
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:
        """Return the elemental amounts in canonical symbol order."""
        return self._composition.amounts

    @property
    def uncertainties(self) -> tuple[tuple[str, Fraction | None], ...]:
        """Return the amount precisions in canonical symbol order."""
        return self._composition.uncertainties

    @property
    def complete(self) -> bool:
        """Return whether all represented elemental material is known."""
        return self._composition.complete

    @property
    def exact(self) -> bool:
        """Return whether all elemental amounts are exact."""
        return self._composition.exact

    @property
    def normalized(self) -> bool:
        """Return whether the composition is normalized within precision."""
        return self._composition.normalized

    @property
    def normalization_status(self) -> str:
        """Return the composition's normalization status."""
        return self._composition.normalization_status

    @property
    def diagnostics(self) -> tuple[CompositionDiagnostic, ...]:
        """Return non-fatal diagnostics associated with the composition."""
        return self._composition.diagnostics

    def unwrap(self) -> Mapping[str, Any]:
        """Return the original elemental amount mapping."""
        return self._raw
