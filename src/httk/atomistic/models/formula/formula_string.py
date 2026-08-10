"""Backend wrapping a raw canonical reduced formula string."""

from fractions import Fraction
from typing import Any, Self, cast

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.notation import try_parse_reduced


class FormulaString(ChemicalFormulaBackend):
    r"""Wrap a canonical reduced formula held as a plain string.

    :param obj: The canonical reduced formula text.
    :param \*\*hints: Backend-selection hints.
    """

    _raw: str
    _coefficients: tuple[tuple[str, int], ...]

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a reduced formula string.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "formula") != "formula":
            return None
        if not isinstance(obj, str):
            return None
        if try_parse_reduced(obj) is None:
            return None
        return cls(obj, **hints)

    def __init__(self, obj: str, **hints: Any) -> None:
        self._raw = obj
        self._coefficients = cast(tuple[tuple[str, int], ...], try_parse_reduced(obj))

    @property
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:
        """Return the formula coefficients as exact amounts."""
        return tuple((element, Fraction(count)) for element, count in self._coefficients)

    def unwrap(self) -> str:
        """Return the original formula text."""
        return self._raw
