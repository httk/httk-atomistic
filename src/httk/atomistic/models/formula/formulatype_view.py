"""Eager site-count anonymous-formula presentation view."""

from typing import TYPE_CHECKING, Any, Self

from httk.core import unwrap

from httk.atomistic.models.formula.anonymous_formula_view import (
    anonymous_amounts,
    anonymous_text,
    anonymous_uncertainties,
)
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.formulatype import Formulatype
from httk.atomistic.models.formula.notation import parse_anonymous_formula
from httk.atomistic.models.formula.view_base import ChemicalFormulaViewBase

if TYPE_CHECKING:
    from httk.atomistic.models.formula.like import ChemicalFormulaLike


class FormulatypeView(ChemicalFormulaViewBase, Formulatype):
    r"""Present how many sites each species class occupies as an eager Formulatype.

    Structures, structuretypes, prototypes and protostructures are counted per species
    class: a structure's classes are its species names, a Wyckoff-family object's are its
    label classes, so the gcd-reduced text equals the object's label prefix, while
    :attr:`amounts` keeps the per-cell class counts. A sibling
    :class:`~httk.atomistic.AnonymousFormulaView` of a structural object is re-counted
    from that object. Disorder and partial
    occupancy do not affect the counts. A bare formula or composition is accepted only
    when every amount is an integer, read as one species per site at full occupancy.

    :param obj: The structural or chemical-formula-like object to present.
    :param \*\*hints: Backend-selection hints.
    :raises ValueError: If a formula or composition has a non-integer amount, or is
        empty or incomplete, or if a ``kind`` hint other than ``"sites"`` is given for a
        structural source.
    """

    _backend: ChemicalFormulaBackend

    def __new__(cls, obj: "ChemicalFormulaLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        from httk.atomistic.models.formula.sites import SiteCountFormula, composition_source

        backend: ChemicalFormulaBackend | None = SiteCountFormula._backend_adopt(obj)
        if backend is None:
            # A composition backend of a structural object (from a sibling view) is
            # re-counted from that object, never read as one species per site.
            composition = cls._prepare_backend(obj, hints)
            source = composition_source(composition)
            backend = composition if source is None else SiteCountFormula(source)
        if isinstance(backend, SiteCountFormula) and hints.get("kind", "sites") != "sites":
            raise ValueError(
                "a Formulatype of a structural source always counts sites; "
                "use AnonymousFormulaView for the composition-based anonymous formula"
            )
        if any(amount.denominator != 1 for _, amount in backend.amounts):
            raise ValueError(
                "a Formulatype needs integer amounts (one species per site, full occupancy); "
                "use AnonymousFormulaView for the composition-based anonymous formula"
            )
        text = anonymous_text(backend)
        instance = str.__new__(cls, text)
        instance._coefficients = parse_anonymous_formula(text)
        instance._backend = backend
        return instance

    def __init__(self, obj: "ChemicalFormulaLike", **hints: Any) -> None:
        pass

    @property
    def amounts(self):
        """Return the per-cell class counts under anonymous labels, not gcd-reduced; the text is reduced."""
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

    def unview(self) -> Formulatype:
        """Return the presented formula as a standalone value.

        :return: The canonical Formulatype value.
        """
        backend = self._backend
        if type(backend) is Formulatype:
            return backend
        return Formulatype(str(self))

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)
