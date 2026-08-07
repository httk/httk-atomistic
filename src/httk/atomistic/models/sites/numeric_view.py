"""
A view presenting any sites backend as a NumericSites object (the plain-numpy presentation).
"""

from functools import cached_property
from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models._vector_guards import require_numpy
from httk.atomistic.models.sites.backend import SitesBackend
from httk.atomistic.models.sites.like import SitesLike
from httk.atomistic.models.sites.numeric import NumericSites
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.sites.view_base import SitesViewBase


class SitesNumericView(SitesViewBase, NumericSites):
    r"""
    A view presenting an underlying sites backend as a ``NumericSites`` object.

    This view is a genuine ``NumericSites``, so it can be passed anywhere one is accepted. Its exact
    ``Sites`` is built lazily from the backend on first access. Like a ``NumericSites`` it requires
    numpy (raising :class:`ImportError` otherwise).

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

    # Build then assign so a failed exact presentation leaves the shadow unmaterialized.
    def _fill_sites(self) -> None:
        NumericSites.__init__(self, Sites(self._backend.reduced_coords, self._backend.precision))

    @cached_property
    def _sites(self) -> Sites:  # type: ignore[override]  # pyright: ignore[reportIncompatibleVariableOverride]
        self._fill_sites()
        return self.__dict__["_sites"]

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> NumericSites:
        """Return this presentation as standalone numeric sites.

        :return: The plain-numpy presentation.
        """
        # A genuine NumericSites backend is exactly the presented value: reuse it.
        backend = self._backend
        if type(backend) is NumericSites:
            return backend
        return NumericSites(self._sites)
