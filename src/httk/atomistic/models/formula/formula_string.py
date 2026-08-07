"""Backend wrapping a raw canonical reduced formula string."""

from fractions import Fraction
from typing import Any, cast

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.notation import try_parse_reduced


class FormulaString(ChemicalFormulaBackend):
    """Backend for a canonical reduced formula held as a plain string."""

    _raw: str
    _coefficients: tuple[tuple[str, int], ...]

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "formula") != "formula":
            return None
        if not isinstance(obj, str):
            return None
        if try_parse_reduced(obj) is None:
            return None
        return super().__new__(cls)

    def __init__(self, obj: str, **hints: Any) -> None:
        self._raw = obj
        self._coefficients = cast(tuple[tuple[str, int], ...], try_parse_reduced(obj))

    @property
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:
        return tuple((element, Fraction(count)) for element, count in self._coefficients)

    def unwrap(self) -> str:
        return self._raw
