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
    """Backend for a mapping of real element symbols to composition amounts."""

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
        return self._composition.amounts

    @property
    def uncertainties(self) -> tuple[tuple[str, Fraction | None], ...]:
        return self._composition.uncertainties

    @property
    def complete(self) -> bool:
        return self._composition.complete

    @property
    def exact(self) -> bool:
        return self._composition.exact

    @property
    def normalized(self) -> bool:
        return self._composition.normalized

    @property
    def normalization_status(self) -> str:
        return self._composition.normalization_status

    @property
    def diagnostics(self) -> tuple[CompositionDiagnostic, ...]:
        return self._composition.diagnostics

    def unwrap(self) -> Mapping[str, Any]:
        return self._raw
