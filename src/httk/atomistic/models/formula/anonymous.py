"""The canonical anonymous chemical-formula value class."""

from fractions import Fraction
from typing import Self

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.notation import parse_anonymous_formula


class AnonymousFormula(ChemicalFormulaBackend, str):
    """A strictly canonical OPTIMADE anonymous chemical formula."""

    _coefficients: tuple[tuple[str, int], ...]

    def __new__(cls, formula: str) -> Self:
        if isinstance(formula, cls):
            return formula
        coefficients = parse_anonymous_formula(formula)
        instance = super().__new__(cls, formula)
        instance._coefficients = coefficients
        return instance

    @property
    def is_anonymous(self) -> bool:
        return True

    @property
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:
        return tuple((label, Fraction(count)) for label, count in self._coefficients)
