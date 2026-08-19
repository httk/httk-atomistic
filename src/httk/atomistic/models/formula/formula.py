"""The canonical reduced chemical-formula value class."""

from fractions import Fraction
from typing import Self

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.notation import parse_reduced_formula


class ChemicalFormula(ChemicalFormulaBackend, str):
    """Store a strictly canonical reduced chemical formula.

    :param formula: The alphabetical formula text with greatest common divisor one.
    """

    _coefficients: tuple[tuple[str, int], ...]

    def __new__(cls, formula: str) -> Self:
        if isinstance(formula, cls):
            return formula
        coefficients = parse_reduced_formula(formula)
        instance = super().__new__(cls, formula)
        instance._coefficients = coefficients
        return instance

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str.__repr__(self)})"

    @property
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:
        """Return the formula coefficients as exact amounts."""
        return tuple((element, Fraction(count)) for element, count in self._coefficients)
