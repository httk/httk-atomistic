"""
A view presenting any sites backend as a NumericSites object (the plain-numpy presentation).
"""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models._vector_guards import require_numpy
from httk.atomistic.models.cell.numeric import FloatRows, float_precision, float_rows
from httk.atomistic.models.sites.backend import SitesBackend
from httk.atomistic.models.sites.like import SitesLike
from httk.atomistic.models.sites.numeric import NumericSites
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.sites.view import SitesView
from httk.atomistic.models.sites.view_base import SitesViewBase


class SitesNumericView(SitesViewBase, NumericSites):
    r"""
    A view presenting an underlying sites backend as a ``NumericSites`` object.

    This view is a genuine ``NumericSites``, so it can be passed anywhere one is accepted, but it
    is a lossless presentation: the exact sites stay one hop away via :attr:`exact`, and the float
    coordinates and precision are computed lazily from the backend on first access.
    :meth:`unview` returns the standalone ``NumericSites`` value. Like a ``NumericSites`` it
    requires numpy (raising :class:`ImportError` otherwise).

    :param obj: The sites-like object to present.
    :param \**hints: Backend-selection hints.
    """

    _backend: SitesBackend

    def __new__(cls, obj: SitesLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        require_numpy()
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        instance._backend = backend
        return instance

    def __init__(self, obj: SitesLike, **hints: Any) -> None:
        pass

    @cached_property
    def exact(self) -> Sites:
        """The exact sites this view presents.

        :return: The backend itself when it is a ``Sites``, else a lazy ``SitesView`` of it.
        """
        backend = self._backend
        return backend if isinstance(backend, Sites) else SitesView(backend)

    @cached_property
    def _reduced_coords(self) -> FloatRows:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        return float_rows(self.exact.reduced_coords_floats(), "NumericSites reduced_coords")

    @cached_property
    def _precision(self) -> float | None:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        return float_precision(self.exact.precision)

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> NumericSites:
        """Return this presentation as a standalone numeric sites value.

        :return: The float-owning sites.
        """
        return NumericSites(self)
