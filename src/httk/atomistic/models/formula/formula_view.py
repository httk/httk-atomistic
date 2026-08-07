"""Eager reduced-formula presentation view."""

from typing import TYPE_CHECKING, Any, Self, cast

from httk.core import unwrap

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.formula import ChemicalFormula
from httk.atomistic.models.formula.notation import reduced_coefficients, render_reduced
from httk.atomistic.models.formula.view_base import ChemicalFormulaViewBase

if TYPE_CHECKING:
    from httk.atomistic.models.formula.like import ChemicalFormulaLike


class ChemicalFormulaView(ChemicalFormulaViewBase, ChemicalFormula):
    """An eager canonical reduced-formula view over a complete composition."""

    _backend: ChemicalFormulaBackend

    def __new__(cls, obj: "ChemicalFormulaLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        if backend.is_anonymous:
            raise ValueError("a chemical formula view cannot invent elements for anonymous labels")
        if not backend.complete:
            raise ValueError(
                'composition is incomplete — species with unknown chemical symbol "X" cannot yield a formula'
            )
        amounts = backend.amounts
        if not amounts:
            raise ValueError("an empty composition cannot yield a formula")
        if type(backend) is ChemicalFormula:
            text = str(backend)
            coefficients = backend._coefficients
        else:
            reduced = cast(
                tuple[int, ...],
                reduced_coefficients(tuple(value for _, value in amounts)),
            )
            coefficients = tuple((element, count) for (element, _), count in zip(amounts, reduced))
            text = render_reduced(coefficients)
        instance = str.__new__(cls, text)
        instance._coefficients = coefficients
        instance._backend = backend
        return instance

    def __init__(self, obj: "ChemicalFormulaLike", **hints: Any) -> None:
        pass

    def unview(self) -> ChemicalFormula:
        backend = self._backend
        if type(backend) is ChemicalFormula:
            return backend
        return ChemicalFormula(str(self))

    def unwrap(self) -> Any:
        return unwrap(self._backend)
