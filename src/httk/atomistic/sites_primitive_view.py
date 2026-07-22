"""
A view presenting any sites backend as a raw Nx3 tuple of reduced coordinates.
"""

from typing import Any, Self

from httk.core import unwrap

from .sites_backend import SitesBackend
from .sites_like import SitesLike
from .sites_view import SitesView


class SitesPrimitiveView(SitesView, tuple):
    """
    A view presenting an underlying sites backend as a raw Nx3 matrix.

    This view is a genuine tuple of reduced-coordinate rows, built eagerly and immutable.
    """

    _backend: SitesBackend

    def __new__(cls, obj: SitesLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls, backend.reduced_coords)
        instance._backend = backend
        return instance

    def __init__(self, obj: SitesLike, **hints: Any) -> None:
        super().__init__()

    def unwrap(self) -> Any:
        return unwrap(self._backend)
