"""Eager reduced-formula presentation view."""

from typing import TYPE_CHECKING, Any, Self, cast

from httk.core import unwrap

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.formula import ChemicalFormula
from httk.atomistic.models.formula.notation import parse_reduced_formula, reduced_coefficients, render_reduced
from httk.atomistic.models.formula.view_base import ChemicalFormulaViewBase

if TYPE_CHECKING:
    from httk.atomistic.models.formula.like import ChemicalFormulaLike


class ChemicalFormulaView(ChemicalFormulaViewBase, ChemicalFormula):
    r"""Present a complete composition as an eager canonical reduced formula.

    :param obj: The chemical-formula-like object to present.
    :param \*\*hints: Backend-selection hints.
    """

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
            coefficients = parse_reduced_formula(text)
        instance = str.__new__(cls, text)
        instance._coefficients = coefficients
        instance._backend = backend
        return instance

    def __init__(self, obj: "ChemicalFormulaLike", **hints: Any) -> None:
        pass

    @property
    def amounts(self):
        """Return the presented elemental amounts."""
        return self._backend.amounts

    @property
    def uncertainties(self):
        """Return the presented amount precisions."""
        return self._backend.uncertainties

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
        return False

    def unview(self) -> ChemicalFormula:
        """Return the presented formula as a standalone value.

        :return: The canonical reduced formula value.
        """
        backend = self._backend
        if type(backend) is ChemicalFormula:
            return backend
        return ChemicalFormula(str(self))

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)
