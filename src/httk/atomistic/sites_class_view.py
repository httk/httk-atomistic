"""
A view presenting any sites backend as a Sites object (the class representation).
"""

from typing import Any, Self

from httk.core import unwrap

from .sites import Sites
from .sites_backend import SitesBackend
from .sites_like import SitesLike
from .sites_view import SitesView


class SitesClassView(SitesView, Sites):
    """
    A view presenting an underlying sites backend as a ``Sites`` object.

    This view is a genuine ``Sites``, so it can be passed anywhere a Sites is accepted.
    Its coordinates are built eagerly from the backend on construction.
    """

    _backend: SitesBackend

    def __new__(cls, obj: SitesLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls)
        # Sites is mutable, so its state is initialized here in __new__ (keeping __init__ a no-op),
        # so that rewrapping an existing view via cls(view) does not re-initialize it.
        Sites.__init__(instance, backend.reduced_coords)
        instance._backend = backend
        return instance

    def __init__(self, obj: SitesLike, **hints: Any) -> None:
        pass

    def unwrap(self) -> Any:
        return unwrap(self._backend)
