"""Lazy composition presentation view."""

from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, Any, Self

from httk.core import unwrap

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.composition import Composition
from httk.atomistic.models.formula.diagnostics import CompositionDiagnostic
from httk.atomistic.models.formula.view_base import ChemicalFormulaViewBase

if TYPE_CHECKING:
    from httk.atomistic.models.formula.like import ChemicalFormulaLike


class CompositionView(ChemicalFormulaViewBase, Composition):
    r"""Present any chemical-formula backend as a lazy composition.

    :param obj: The chemical-formula-like object to present.
    :param \*\*hints: Backend-selection hints.
    """

    _backend: ChemicalFormulaBackend

    def __new__(cls, obj: "ChemicalFormulaLike", **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        if backend.is_anonymous:
            raise ValueError("a composition view cannot invent elements for anonymous labels")
        instance = super().__new__(cls)
        instance._backend = backend
        return instance

    def __init__(self, obj: "ChemicalFormulaLike", **hints: Any) -> None:
        pass

    def _fill(self) -> None:
        backend = self._backend
        object.__setattr__(self, "amounts", backend.amounts)
        object.__setattr__(self, "uncertainties", backend.uncertainties)
        object.__setattr__(self, "complete", backend.complete)
        object.__setattr__(self, "exact", backend.exact)
        object.__setattr__(self, "normalized", backend.normalized)
        object.__setattr__(self, "normalization_status", backend.normalization_status)
        object.__setattr__(self, "diagnostics", backend.diagnostics)

    @cached_property
    def amounts(self) -> tuple[tuple[str, Fraction], ...]:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Return the lazily materialized elemental amounts."""
        if "amounts" not in self.__dict__:
            self._fill()
        return self.__dict__["amounts"]

    @cached_property
    def uncertainties(self) -> tuple[tuple[str, Fraction | None], ...]:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Return the lazily materialized amount precisions."""
        if "amounts" not in self.__dict__:
            self._fill()
        return self.__dict__["uncertainties"]

    @cached_property
    def complete(self) -> bool:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Return whether the presented composition is complete."""
        if "amounts" not in self.__dict__:
            self._fill()
        return self.__dict__["complete"]

    @cached_property
    def exact(self) -> bool:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Return whether the presented amounts are exact."""
        if "amounts" not in self.__dict__:
            self._fill()
        return self.__dict__["exact"]

    @cached_property
    def normalized(self) -> bool:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Return whether the presented composition is normalized."""
        if "amounts" not in self.__dict__:
            self._fill()
        return self.__dict__["normalized"]

    @cached_property
    def normalization_status(self) -> str:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Return the presented composition's normalization status."""
        if "amounts" not in self.__dict__:
            self._fill()
        return self.__dict__["normalization_status"]

    @cached_property
    def diagnostics(self) -> tuple[CompositionDiagnostic, ...]:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Return diagnostics associated with the presented composition."""
        if "amounts" not in self.__dict__:
            self._fill()
        return self.__dict__["diagnostics"]

    def _ensure_materialized(self) -> None:
        _ = self.amounts

    def __reduce__(self) -> tuple[type[Self], tuple[ChemicalFormulaBackend]]:
        """Rebuild the view from its backend during pickling.

        :return: The view constructor and its backend argument.
        """
        return type(self), (self._backend,)

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> Composition:
        """Return the presented composition as a standalone value.

        :return: The materialized composition value.
        """
        backend = self._backend
        if type(backend) is Composition:
            return backend
        return Composition(
            self.amounts,
            self.uncertainties,
            self.complete,
            self.exact,
            self.normalized,
            self.normalization_status,
            self.diagnostics,
        )
