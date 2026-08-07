"""Eager anonymous-formula presentation view."""

from typing import TYPE_CHECKING, Any, Self, cast

from httk.core import unwrap

from httk.atomistic.models.formula.anonymous import AnonymousFormula
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.notation import anonymous_symbol, reduced_coefficients, render_anonymous
from httk.atomistic.models.formula.view_base import ChemicalFormulaViewBase

if TYPE_CHECKING:
    from httk.atomistic.models.formula.like import ChemicalFormulaLike


class AnonymousFormulaView(ChemicalFormulaViewBase, AnonymousFormula):
    """An eager canonical anonymous-formula view over any complete composition."""

    _backend: ChemicalFormulaBackend

    def __new__(cls, obj: "ChemicalFormulaLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        if backend.is_anonymous:
            if type(backend) is AnonymousFormula:
                text = str(backend)
                coefficients = backend._coefficients
            else:
                coefficients = tuple((label, int(count)) for label, count in backend.amounts)
                text = render_anonymous(tuple(count for _, count in coefficients))
        else:
            if not backend.complete:
                raise ValueError(
                    'composition is incomplete — species with unknown chemical symbol "X" cannot yield a formula'
                )
            amounts = backend.amounts
            if not amounts:
                raise ValueError("an empty composition cannot yield a formula")
            reduced = cast(tuple[int, ...], reduced_coefficients(tuple(value for _, value in amounts)))
            ordered = sorted(zip((element for element, _ in amounts), reduced), key=lambda item: (-item[1], item[0]))
            coefficients = tuple((anonymous_symbol(index), count) for index, (_, count) in enumerate(ordered))
            text = render_anonymous(tuple(count for _, count in coefficients))
        instance = str.__new__(cls, text)
        instance._coefficients = coefficients
        instance._backend = backend
        return instance

    def __init__(self, obj: "ChemicalFormulaLike", **hints: Any) -> None:
        pass

    def unview(self) -> AnonymousFormula:
        backend = self._backend
        if type(backend) is AnonymousFormula:
            return backend
        return AnonymousFormula(str(self))

    def unwrap(self) -> Any:
        return unwrap(self._backend)
