"""Backend projecting a normalized composition storage record."""

from fractions import Fraction
from functools import cached_property
from typing import Any, Self

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.composition import Composition
from httk.atomistic.models.formula.diagnostics import CompositionDiagnostic
from httk.atomistic.storage.records import (
    NormalizedCompositionRecord,
    _composition_from_record,
)


class RecordComposition(ChemicalFormulaBackend):
    r"""Read a composition from a :class:`NormalizedCompositionRecord`.

    :param obj: The normalized-composition record to wrap.
    :param \*\*hints: Backend-selection hints.
    """

    _record: NormalizedCompositionRecord

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a normalized composition record.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, NormalizedCompositionRecord):
            return None
        return cls(obj, **hints)

    def __init__(self, obj: NormalizedCompositionRecord, **hints: Any) -> None:
        self._record = obj

    @cached_property
    def _composition(self) -> Composition:
        return _composition_from_record(self._record)

    @property
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:
        """Return the stored elemental amounts."""
        return self._composition.amounts

    @property
    def uncertainties(self) -> tuple[tuple[str, Fraction | None], ...]:
        """Return the stored amount precisions."""
        return self._composition.uncertainties

    @property
    def complete(self) -> bool:
        """Return whether the stored composition is complete."""
        return self._composition.complete

    @property
    def exact(self) -> bool:
        """Return whether the stored amounts are exact."""
        return self._composition.exact

    @property
    def normalized(self) -> bool:
        """Return whether the stored composition is normalized."""
        return self._composition.normalized

    @property
    def normalization_status(self) -> str:
        """Return the stored composition's normalization status."""
        return self._composition.normalization_status

    @property
    def diagnostics(self) -> tuple[CompositionDiagnostic, ...]:
        """Return diagnostics stored with the composition."""
        return self._composition.diagnostics

    def unwrap(self) -> NormalizedCompositionRecord:
        """Return the original normalized-composition record."""
        return self._record
