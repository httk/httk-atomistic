"""Backend projecting a structure into its elemental composition."""

from fractions import Fraction
from functools import cached_property
from typing import Any

from httk.core import unwrap

from httk.atomistic.composition import project_composition
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.composition import Composition
from httk.atomistic.models.formula.diagnostics import CompositionDiagnostic
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.view import StructureView


class StructureComposition(ChemicalFormulaBackend):
    """Backend for a structure, projected only when composition data is accessed."""

    _structure: StructureBackend

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "structure") != "structure":
            return None
        if isinstance(obj, StructureBackend):
            return super().__new__(cls)
        if isinstance(obj, StructureView):
            return super().__new__(cls) if getattr(obj, "_backend", None) is not None else None
        return None

    def __init__(self, obj: Any, **hints: Any) -> None:
        self._structure = obj if isinstance(obj, StructureBackend) else obj._backend

    @cached_property
    def _composition(self) -> Composition:
        return project_composition(self._structure)

    @property
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:
        return self._composition.amounts

    @property
    def uncertainties(self) -> tuple[tuple[str, Fraction | None], ...]:
        return self._composition.uncertainties

    @property
    def complete(self) -> bool:
        return self._composition.complete

    @property
    def exact(self) -> bool:
        return self._composition.exact

    @property
    def normalized(self) -> bool:
        return self._composition.normalized

    @property
    def normalization_status(self) -> str:
        return self._composition.normalization_status

    @property
    def diagnostics(self) -> tuple[CompositionDiagnostic, ...]:
        return self._composition.diagnostics

    def unwrap(self) -> Any:
        return unwrap(self._structure)
