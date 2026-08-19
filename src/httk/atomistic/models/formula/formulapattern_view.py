"""Eager anonymous-formula presentation view."""

from typing import TYPE_CHECKING, Any, Self, cast

from httk.core import unwrap

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.formulapattern import Formulapattern
from httk.atomistic.models.formula.notation import (
    anonymous_symbol,
    parse_anonymous_formula,
    reduced_coefficients,
    render_anonymous,
)
from httk.atomistic.models.formula.view_base import ChemicalFormulaViewBase

if TYPE_CHECKING:
    from httk.atomistic.models.formula.like import ChemicalFormulaLike


class FormulapatternView(ChemicalFormulaViewBase, Formulapattern):
    r"""Present a complete composition as an eager canonical anonymous formula.

    The canonical class name is ``FormulapatternView``; the legacy
    ``AnonymousFormulaView`` name remains available as an alias.

    :param obj: The chemical-formula-like object to present.
    :param \*\*hints: Backend-selection hints.
    """

    _backend: ChemicalFormulaBackend

    def __new__(cls, obj: "ChemicalFormulaLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        if backend.is_anonymous:
            if type(backend) is Formulapattern:
                text = str(backend)
                coefficients = backend._coefficients
            else:
                amounts = backend.amounts
                if not amounts:
                    raise ValueError("an empty composition cannot yield an anonymous formula")
                reduced = cast(
                    tuple[int, ...],
                    reduced_coefficients(tuple(value for _, value in amounts)),
                )
                coefficients = tuple((label, count) for (label, _), count in zip(amounts, reduced))
                text = "".join(label + (str(count) if count != 1 else "") for label, count in coefficients)
                coefficients = parse_anonymous_formula(text)
        else:
            if not backend.complete:
                raise ValueError(
                    'composition is incomplete — species with unknown chemical symbol "X" cannot yield a formula'
                )
            amounts = backend.amounts
            if not amounts:
                raise ValueError("an empty composition cannot yield a formula")
            reduced = cast(tuple[int, ...], reduced_coefficients(tuple(value for _, value in amounts)))
            reduced_by_element = dict(zip((element for element, _ in amounts), reduced))
            ordered = sorted(amounts, key=lambda item: (-item[1], item[0]))
            coefficients = tuple(
                (anonymous_symbol(index), reduced_by_element[element]) for index, (element, _) in enumerate(ordered)
            )
            text = render_anonymous(tuple(count for _, count in coefficients))
            coefficients = parse_anonymous_formula(text)
        instance = str.__new__(cls, text)
        instance._coefficients = coefficients
        instance._backend = backend
        return instance

    def __init__(self, obj: "ChemicalFormulaLike", **hints: Any) -> None:
        pass

    @property
    def amounts(self):
        """Return the presented amounts using anonymous labels."""
        backend = self._backend
        if backend.is_anonymous:
            return backend.amounts
        ordered = sorted(backend.amounts, key=lambda item: (-item[1], item[0]))
        return tuple((anonymous_symbol(index), amount) for index, (_, amount) in enumerate(ordered))

    @property
    def uncertainties(self):
        """Return the presented amount precisions using anonymous labels."""
        backend = self._backend
        if backend.is_anonymous:
            return backend.uncertainties
        uncertainties = dict(backend.uncertainties)
        ordered = sorted(backend.amounts, key=lambda item: (-item[1], item[0]))
        return tuple((anonymous_symbol(index), uncertainties[element]) for index, (element, _) in enumerate(ordered))

    @property
    def complete(self):
        """Return whether the presented composition is complete."""
        return self._backend.complete

    @property
    def exact(self):
        """Return whether the presented amounts are exact."""
        return self._backend.exact

    @property
    def normalized(self):
        """Return whether the presented composition is normalized."""
        return self._backend.normalized

    @property
    def normalization_status(self):
        """Return the presented composition's normalization status."""
        return self._backend.normalization_status

    @property
    def diagnostics(self):
        """Return diagnostics associated with the presented composition."""
        return self._backend.diagnostics

    @property
    def is_anonymous(self):
        """Return whether this formula uses anonymous labels."""
        return True

    def unview(self) -> Formulapattern:
        """Return the presented formula as a standalone value.

        :return: The canonical anonymous formula value.
        """
        backend = self._backend
        if type(backend) is Formulapattern:
            return backend
        return Formulapattern(str(self))

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)
