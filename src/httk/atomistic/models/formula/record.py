"""Backend projecting a normalized composition storage record."""

from fractions import Fraction
from functools import cached_property
from typing import Any

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.composition import Composition
from httk.atomistic.models.formula.diagnostics import CompositionDiagnostic
from httk.atomistic.storage.records import (
    NormalizedCompositionRecord,
    _composition_from_record,
)


class RecordComposition(ChemicalFormulaBackend):
    """Backend for a :class:`NormalizedCompositionRecord`."""

    _record: NormalizedCompositionRecord

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, NormalizedCompositionRecord):
            return None
        return super().__new__(cls)

    def __init__(self, obj: NormalizedCompositionRecord, **hints: Any) -> None:
        self._record = obj

    @cached_property
    def _composition(self) -> Composition:
        return _composition_from_record(self._record)

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

    def unwrap(self) -> NormalizedCompositionRecord:
        return self._record
