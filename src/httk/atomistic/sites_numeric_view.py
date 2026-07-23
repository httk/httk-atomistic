"""
A view presenting any sites backend as a NumericSites object (the plain-numpy presentation).
"""

from typing import Any, Self

from httk.core import unwrap

from .numeric_sites import NumericSites
from .sites import Sites
from .sites_backend import SitesBackend
from .sites_like import SitesLike
from .sites_view import SitesView


class SitesNumericView(SitesView, NumericSites):
    """
    A view presenting an underlying sites backend as a ``NumericSites`` object.

    This view is a genuine ``NumericSites``, so it can be passed anywhere one is accepted. Its exact
    ``Sites`` is built eagerly from the backend on construction. Like a ``NumericSites`` it requires
    numpy (raising :class:`ImportError` otherwise).
    """

    _backend: SitesBackend

    def __new__(cls, obj: SitesLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        # NumericSites is mutable, so its state is initialized here in __new__ (keeping __init__ a
        # no-op), so that rewrapping an existing view via cls(view) does not re-initialize it.
        NumericSites.__init__(instance, Sites(backend.reduced_coords))
        instance._backend = backend
        return instance

    def __init__(self, obj: SitesLike, **hints: Any) -> None:
        pass

    def unwrap(self) -> Any:
        return unwrap(self._backend)
