"""
Backend wrapping a raw Nx3 matrix of reduced coordinates.
"""

from typing import Any, Self

from httk.core import FracVector

from httk.atomistic.models._vector_guards import is_coords_nx3, to_fracvector
from httk.atomistic.models.sites.backend import SitesBackend


class PlainSites(SitesBackend):
    r"""
    Backend for sites backed by a raw Nx3 list or tuple of numbers (or any Nx3 vector-like).

    The native representation is preserved verbatim (one site per row); the exact rational
    :class:`~httk.core.FracVector` ``reduced_coords`` are built lazily and cached, and ``unwrap``
    returns the original raw object.

    :param obj: The raw reduced-coordinate representation.
    :param \**hints: Backend-selection hints.
    """

    _raw: Any
    _reduced_coords_cache: FracVector | None

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a plain reduced-coordinate matrix.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "plain") != "plain":
            return None
        if not is_coords_nx3(obj):
            return None
        return cls(obj, **hints)

    def __init__(self, obj: Any, **hints: Any) -> None:
        self._raw = obj
        self._reduced_coords_cache = None

    @property
    def reduced_coords(self) -> FracVector:
        """Return the reduced coordinates in the canonical representation.

        :return: The exact reduced coordinates.
        """
        if self._reduced_coords_cache is None:
            self._reduced_coords_cache = to_fracvector(self._raw)
        return self._reduced_coords_cache

    def unwrap(self) -> Any:
        """Return the original coordinate object.

        :return: The raw coordinate representation.
        """
        return self._raw
