"""Formula bridge for anonymous structures and prototypes."""

from collections import Counter
from fractions import Fraction
from functools import cached_property
from typing import Any

from httk.core import unwrap

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.prototype.backend import AnonymousStructureBackend
from httk.atomistic.models.prototype.view_base import AnonymousStructureViewBase


class PrototypeComposition(ChemicalFormulaBackend):
    """The canonical anonymous composition of an anonymous-structure backend."""

    _prototype: AnonymousStructureBackend
    kind = "prototype"

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if isinstance(obj, (AnonymousStructureBackend, AnonymousStructureViewBase)):
            return super().__new__(cls)
        return None

    def __init__(self, obj: Any, **hints: Any) -> None:
        self._prototype = obj._backend if isinstance(obj, AnonymousStructureViewBase) else obj

    @property
    def is_anonymous(self) -> bool:
        return True

    @cached_property
    def _amounts(self) -> tuple[tuple[str, Fraction], ...]:
        counts = Counter(self._prototype.species_at_sites)
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return tuple((anonymous_symbol(index), Fraction(count)) for index, (_, count) in enumerate(ordered))

    @property
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:
        return self._amounts

    @property
    def complete(self) -> bool:
        return True

    def unwrap(self) -> Any:
        return unwrap(self._prototype)
