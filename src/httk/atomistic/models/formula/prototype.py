"""Formula bridge for anonymous structures and prototypes."""

from collections import Counter
from fractions import Fraction
from functools import cached_property
from types import SimpleNamespace
from typing import Any

from httk.core import unwrap

import httk.atomistic.models.protostructure.backend
import httk.atomistic.models.protostructure.view_base
from httk.atomistic.composition import project_composition
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.composition import Composition
from httk.atomistic.models.formula.notation import anonymous_symbol
from httk.atomistic.models.prototype.backend import AnonymousStructureBackend
from httk.atomistic.models.prototype.view_base import AnonymousStructureViewBase


class PrototypeComposition(ChemicalFormulaBackend):
    """The canonical anonymous composition of an anonymous-structure backend."""

    _prototype: Any
    kind = "prototype"

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if isinstance(
            obj,
            (
                AnonymousStructureBackend,
                AnonymousStructureViewBase,
                httk.atomistic.models.protostructure.backend.ProtostructureBackend,
                httk.atomistic.models.protostructure.view_base.ProtostructureViewBase,
            ),
        ):
            return super().__new__(cls)
        return None

    def __init__(self, obj: Any, **hints: Any) -> None:
        if isinstance(
            obj, (AnonymousStructureViewBase, httk.atomistic.models.protostructure.view_base.ProtostructureViewBase)
        ):
            self._prototype = obj._backend
        else:
            self._prototype = obj

    @property
    def _is_protostructure(self) -> bool:
        return isinstance(self._prototype, httk.atomistic.models.protostructure.backend.ProtostructureBackend)

    @cached_property
    def _projected(self) -> Composition:
        source = self._prototype
        species_by_name: dict[str, Any] = {}
        for occupation in source.occupations:
            species_by_name.setdefault(occupation.species.name, occupation.species)
        proxy = SimpleNamespace(
            species=tuple(species_by_name.values()),
            wyckoff_sites=tuple(SimpleNamespace(species=occupation.species.name) for occupation in source.occupations),
            multiplicities=source.multiplicities,
            assemblies=None,
            chemical_composition=None,
        )
        return project_composition(proxy)

    @property
    def is_anonymous(self) -> bool:
        return not self._is_protostructure

    @cached_property
    def _amounts(self) -> tuple[tuple[str, Fraction], ...]:
        if self._is_protostructure:
            return self._projected.amounts
        counts = Counter(self._prototype.species_at_sites)
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return tuple((anonymous_symbol(index), Fraction(count)) for index, (_, count) in enumerate(ordered))

    @property
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:
        return self._amounts

    @property
    def uncertainties(self) -> tuple[tuple[str, Fraction | None], ...]:
        if self._is_protostructure:
            return self._projected.uncertainties
        return super().uncertainties

    @property
    def complete(self) -> bool:
        return self._projected.complete if self._is_protostructure else True

    @property
    def exact(self) -> bool:
        return self._projected.exact if self._is_protostructure else super().exact

    @property
    def normalized(self) -> bool:
        return self._projected.normalized if self._is_protostructure else super().normalized

    @property
    def normalization_status(self) -> str:
        return self._projected.normalization_status if self._is_protostructure else super().normalization_status

    @property
    def diagnostics(self):
        return self._projected.diagnostics if self._is_protostructure else super().diagnostics

    def unwrap(self) -> Any:
        return unwrap(self._prototype)
