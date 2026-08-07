"""The immutable canonical elemental composition value."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any

from httk.atomistic._composition_values import as_fraction, as_precision
from httk.atomistic.elements import SYMBOLS
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.diagnostics import CompositionDiagnostic
from httk.atomistic.models.formula.notation import (
    reduced_coefficients,
    render_anonymous,
    render_reduced,
)

_ELEMENTS = frozenset(SYMBOLS)


@dataclass(frozen=True, init=False, eq=False)
class Composition(ChemicalFormulaBackend):
    """The immutable projected elemental composition and formula diagnostics."""

    amounts: tuple[tuple[str, Fraction], ...] = ()  # pyright: ignore[reportIncompatibleMethodOverride]
    uncertainties: tuple[tuple[str, Fraction | None], ...] = ()  # pyright: ignore[reportIncompatibleMethodOverride]
    complete: bool = True  # pyright: ignore[reportIncompatibleMethodOverride]
    exact: bool = True  # pyright: ignore[reportIncompatibleMethodOverride]
    normalized: bool = True  # pyright: ignore[reportIncompatibleMethodOverride]
    normalization_status: str = "exact"  # pyright: ignore[reportIncompatibleMethodOverride]
    diagnostics: tuple[CompositionDiagnostic, ...] = ()  # pyright: ignore[reportIncompatibleMethodOverride]

    def __init__(
        self,
        amounts: Mapping[str, Any] | Iterable[tuple[str, Any]],
        uncertainties: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
        complete: bool = True,
        exact: bool | None = None,
        normalized: bool = True,
        normalization_status: str | None = None,
        diagnostics: Iterable[CompositionDiagnostic] = (),
    ) -> None:
        raw_amounts = dict(amounts)
        converted: dict[str, Fraction] = {}
        inferred: dict[str, Fraction | None] = {}
        for element, value in raw_amounts.items():
            if element not in _ELEMENTS:
                raise ValueError(f"Composition amount is not a real element: {element!r}")
            central, width = as_fraction(value, field=f"Composition amount for {element}")
            if central <= 0:
                raise ValueError("Composition amounts must be positive")
            converted[element] = central
            inferred[element] = width
        ordered = tuple((element, converted[element]) for element in sorted(converted))

        if uncertainties is None:
            ordered_uncertainties = tuple((element, inferred[element]) for element, _ in ordered)
        else:
            supplied = dict(uncertainties)
            if set(supplied) != set(converted):
                raise ValueError("Composition uncertainty keys must match amount keys")
            ordered_uncertainties = tuple(
                (element, as_precision(supplied[element], field=f"Composition uncertainty for {element}"))
                for element, _ in ordered
            )
        derived_exact = all(width is None for _, width in ordered_uncertainties)
        stated_exact = derived_exact if exact is None else exact
        stated_status = (
            ("exact" if stated_exact and normalized else "within_precision" if normalized else "outside_precision")
            if normalization_status is None
            else normalization_status
        )
        object.__setattr__(self, "amounts", ordered)
        object.__setattr__(self, "uncertainties", ordered_uncertainties)
        object.__setattr__(self, "complete", complete)
        object.__setattr__(self, "exact", stated_exact)
        object.__setattr__(self, "normalized", normalized)
        object.__setattr__(self, "normalization_status", stated_status)
        object.__setattr__(self, "diagnostics", tuple(diagnostics))

    @property
    def amount_mapping(self) -> Mapping[str, Fraction]:
        return MappingProxyType(dict(self.amounts))

    @property
    def uncertainty_mapping(self) -> Mapping[str, Fraction | None]:
        return MappingProxyType(dict(self.uncertainties))

    @property
    def elements(self) -> tuple[str, ...]:
        return tuple(element for element, _ in self.amounts)

    @property
    def nelements(self) -> int:
        return len(self.amounts)

    @property
    def elements_ratios(self) -> tuple[Fraction, ...]:
        total = sum((amount for _, amount in self.amounts), Fraction())
        return () if not total else tuple(amount / total for _, amount in self.amounts)

    def _formula_coefficients(self) -> tuple[tuple[str, int], ...] | None:
        if not self.complete or not self.amounts:
            return None
        central = reduced_coefficients(self.elements_ratios)
        return (
            None if central is None else tuple((element, amount) for (element, _), amount in zip(self.amounts, central))
        )

    @property
    def chemical_formula_reduced(self) -> str | None:
        coefficients = self._formula_coefficients()
        return None if coefficients is None else render_reduced(coefficients)

    @property
    def chemical_formula_anonymous(self) -> str | None:
        coefficients = self._formula_coefficients()
        if coefficients is None:
            return None
        ordered = sorted(coefficients, key=lambda item: (-item[1], item[0]))
        return render_anonymous(tuple(amount for _, amount in ordered))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Composition):
            return NotImplemented
        return self._identity_tuple() == other._identity_tuple()

    def __hash__(self) -> int:
        return hash(self._identity_tuple())

    def _identity_tuple(self) -> tuple[object, ...]:
        return (
            self.amounts,
            self.uncertainties,
            self.complete,
            self.exact,
            self.normalized,
            self.normalization_status,
            self.diagnostics,
        )
