"""
Backend wrapping a raw Nx3 matrix of reduced coordinates.
"""

from typing import Any

from httk.core import FracVector

from ._vector_guards import is_coords_nx3, to_fracvector
from .sites_backend import SitesBackend


class SitesPrimitive(SitesBackend):
    """
    Backend for sites backed by a raw Nx3 list or tuple of numbers (or any Nx3 vector-like).

    The native representation is preserved verbatim (one site per row); the exact rational
    :class:`~httk.core.FracVector` ``reduced_coords`` are built lazily and cached, and ``unwrap``
    returns the original raw object.
    """

    _raw: Any
    _reduced_coords_cache: FracVector | None

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "primitive") != "primitive":
            return None
        if not is_coords_nx3(obj):
            return None
        return super().__new__(cls)

    def __init__(self, obj: Any, **hints: Any) -> None:
        self._raw = obj
        self._reduced_coords_cache = None

    @property
    def reduced_coords(self) -> FracVector:
        if self._reduced_coords_cache is None:
            self._reduced_coords_cache = to_fracvector(self._raw)
        return self._reduced_coords_cache

    def unwrap(self) -> Any:
        return self._raw
