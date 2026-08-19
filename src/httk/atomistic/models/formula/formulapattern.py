"""The canonical anonymous chemical-formula value class."""

from fractions import Fraction
from typing import Self

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.notation import parse_anonymous_formula


class Formulapattern(ChemicalFormulaBackend, str):
    """Store a strictly canonical OPTIMADE anonymous chemical formula.

    The canonical class name is ``Formulapattern``; the legacy
    :class:`AnonymousFormula` name remains available as an alias.

    :param formula: The canonical anonymous formula text.
    """

    _coefficients: tuple[tuple[str, int], ...]

    def __new__(cls, formula: str) -> Self:
        if isinstance(formula, cls):
            return formula
        coefficients = parse_anonymous_formula(formula)
        instance = super().__new__(cls, formula)
        instance._coefficients = coefficients
        return instance

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str.__repr__(self)})"

    @property
    def is_anonymous(self) -> bool:
        """Return whether the formula uses anonymous labels."""
        return True

    @property
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:
        """Return the anonymous coefficients as exact amounts."""
        return tuple((label, Fraction(count)) for label, count in self._coefficients)
