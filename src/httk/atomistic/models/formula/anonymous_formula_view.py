"""Eager composition-based anonymous-formula presentation view."""

from fractions import Fraction
from typing import TYPE_CHECKING, Any, Self, cast

from httk.core import unwrap

from httk.atomistic.models.formula.anonymous_formula import AnonymousFormula
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.formulatype import Formulatype
from httk.atomistic.models.formula.notation import (
    anonymous_symbol,
    parse_anonymous_formula,
    reduced_coefficients,
    render_anonymous,
)
from httk.atomistic.models.formula.view_base import ChemicalFormulaViewBase

if TYPE_CHECKING:
    from httk.atomistic.models.formula.like import ChemicalFormulaLike


def anonymous_text(backend: ChemicalFormulaBackend) -> str:
    """Return the anonymous formula text of a backend's amounts.

    Anonymous amounts keep their labels and order; named amounts are relabelled by
    decreasing amount, ties by element symbol. Either is reduced by the overall gcd.

    :param backend: The chemical-formula backend to render.
    :return: The anonymous formula text, still to be validated by the parser.
    :raises ValueError: If the composition is empty or incomplete.
    """
    if isinstance(backend, (AnonymousFormula, Formulatype)):
        return str(backend)
    if not backend.is_anonymous and not backend.complete:
        raise ValueError('composition is incomplete — species with unknown chemical symbol "X" cannot yield a formula')
    amounts = backend.amounts
    if not amounts:
        raise ValueError("an empty composition cannot yield an anonymous formula")
    if backend.is_anonymous:
        reduced = cast(tuple[int, ...], reduced_coefficients(tuple(value for _, value in amounts)))
        return "".join(label + (str(count) if count != 1 else "") for (label, _), count in zip(amounts, reduced))
    ordered = sorted(amounts, key=lambda item: (-item[1], item[0]))
    return render_anonymous(cast(tuple[int, ...], reduced_coefficients(tuple(value for _, value in ordered))))


def _anonymous_order(backend: ChemicalFormulaBackend) -> list[str]:
    """Return a named backend's element symbols in anonymous-label order."""
    return [element for element, _ in sorted(backend.amounts, key=lambda item: (-item[1], item[0]))]


def anonymous_amounts(backend: ChemicalFormulaBackend) -> tuple[tuple[str, Fraction], ...]:
    """Return a backend's amounts under anonymous labels.

    :param backend: The chemical-formula backend to present.
    :return: The amounts, relabelled by decreasing amount unless already anonymous.
    """
    if backend.is_anonymous:
        return backend.amounts
    amounts = dict(backend.amounts)
    return tuple((anonymous_symbol(index), amounts[element]) for index, element in enumerate(_anonymous_order(backend)))


def anonymous_uncertainties(backend: ChemicalFormulaBackend) -> tuple[tuple[str, Fraction | None], ...]:
    """Return a backend's amount precisions under anonymous labels.

    :param backend: The chemical-formula backend to present.
    :return: The precisions, relabelled as :func:`anonymous_amounts` relabels amounts.
    """
    if backend.is_anonymous:
        return backend.uncertainties
    uncertainties = dict(backend.uncertainties)
    return tuple(
        (anonymous_symbol(index), uncertainties[element]) for index, element in enumerate(_anonymous_order(backend))
    )


class AnonymousFormulaView(ChemicalFormulaViewBase, AnonymousFormula):
    r"""Present a complete composition as an eager composition-based anonymous formula.

    Element amounts are relabelled ``A``, ``B``, ... by decreasing amount and reduced
    by their gcd, so partial occupancy and mixed sites contribute fractionally. For the
    site-count anonymous formula use :class:`~httk.atomistic.FormulatypeView`; a
    ``FormulatypeView`` of a structural object passed here is re-projected from that
    object by composition.

    :param obj: The chemical-formula-like object to present.
    :param \*\*hints: Backend-selection hints.
    """

    _backend: ChemicalFormulaBackend

    def __new__(cls, obj: "ChemicalFormulaLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        from httk.atomistic.models.formula.sites import SiteCountFormula

        backend = cls._prepare_backend(obj, hints)
        if isinstance(backend, SiteCountFormula):  # a sibling FormulatypeView: recount by composition
            backend = cls._prepare_backend(backend._source, hints)
        text = anonymous_text(backend)
        instance = str.__new__(cls, text)
        instance._coefficients = parse_anonymous_formula(text)
        instance._backend = backend
        return instance

    def __init__(self, obj: "ChemicalFormulaLike", **hints: Any) -> None:
        pass

    @property
    def amounts(self):
        """Return the presented amounts using anonymous labels."""
        return anonymous_amounts(self._backend)

    @property
    def uncertainties(self):
        """Return the presented amount precisions using anonymous labels."""
        return anonymous_uncertainties(self._backend)

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

    def unview(self) -> AnonymousFormula:
        """Return the presented formula as a standalone value.

        :return: The composition-based anonymous formula value.
        """
        backend = self._backend
        if type(backend) is AnonymousFormula:
            return backend
        return AnonymousFormula(str(self))

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)
