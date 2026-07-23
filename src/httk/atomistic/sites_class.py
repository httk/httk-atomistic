"""
Backend wrapping a Sites object in the class representation.
"""

from typing import Any

from httk.core import FracVector

from .sites import Sites
from .sites_backend import SitesBackend


class SitesClass(SitesBackend):
    """
    Backend for sites backed by an actual ``Sites`` object.

    Its ``reduced_coords`` accessor delegates to the wrapped Sites, and ``unwrap``
    returns that Sites.
    """

    _sites: Sites

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if not isinstance(obj, Sites):
            return None
        if hints and hints.get("kind", "class") != "class":
            return None
        return super().__new__(cls)

    def __init__(self, obj: Sites, **hints: Any) -> None:
        self._sites = obj

    @property
    def reduced_coords(self) -> FracVector:
        return self._sites.reduced_coords

    def unwrap(self) -> Any:
        return self._sites
