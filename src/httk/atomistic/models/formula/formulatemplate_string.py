"""Backend wrapping a raw canonical anonymous formula string."""

from fractions import Fraction
from typing import Any, Self, cast

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.notation import try_parse_anonymous


class FormulatemplateString(ChemicalFormulaBackend):
    r"""Wrap a canonical anonymous formula held as a plain string.

    The canonical class name is ``FormulatemplateString``; the legacy
    ``AnonymousFormulaString`` name remains available as a package-internal
    alias.

    :param obj: The canonical anonymous formula text.
    :param \*\*hints: Backend-selection hints.
    """

    _raw: str
    _coefficients: tuple[tuple[str, int], ...]

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt an anonymous formula string.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "anonymous") != "anonymous":
            return None
        if not isinstance(obj, str):
            return None
        if try_parse_anonymous(obj) is None:
            return None
        return cls(obj, **hints)

    def __init__(self, obj: str, **hints: Any) -> None:
        self._raw = obj
        self._coefficients = cast(tuple[tuple[str, int], ...], try_parse_anonymous(obj))

    @property
    def is_anonymous(self) -> bool:
        """Return whether the formula uses anonymous labels."""
        return True

    @property
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:
        """Return the anonymous coefficients as exact amounts."""
        return tuple((label, Fraction(count)) for label, count in self._coefficients)

    def unwrap(self) -> str:
        """Return the original formula text."""
        return self._raw
