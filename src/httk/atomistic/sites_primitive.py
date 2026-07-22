"""
Backend wrapping a raw Nx3 matrix of reduced coordinates.
"""

from typing import Any

from .sites_backend import SitesBackend


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_nx3(matrix: Any) -> bool:
    if not isinstance(matrix, (list, tuple)):
        return False
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            return False
        if not all(_is_number(x) for x in row):
            return False
    return True


class SitesPrimitive(SitesBackend):
    """
    Backend for sites backed by a raw Nx3 list or tuple of numbers.

    The native representation is an Nx3 nested list or tuple of reduced coordinates (one
    site per row). The ``reduced_coords`` are derived lazily and cached, and ``unwrap``
    returns the original raw object.
    """

    _raw: Any
    _reduced_coords_cache: tuple[tuple[float, ...], ...] | None

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "primitive") != "primitive":
            return None
        if not _is_nx3(obj):
            return None
        return super().__new__(cls)

    def __init__(self, obj: Any, **hints: Any) -> None:
        self._raw = obj
        self._reduced_coords_cache = None

    @property
    def reduced_coords(self) -> tuple[tuple[float, ...], ...]:
        if self._reduced_coords_cache is None:
            self._reduced_coords_cache = tuple(tuple(float(x) for x in row) for row in self._raw)
        return self._reduced_coords_cache

    def unwrap(self) -> Any:
        return self._raw
