"""Cartesian per-site magnetic-moment backend."""

import fractions
from typing import Any

from httk.core import SurdVector, VectorLike

from httk.atomistic.models._vector_guards import to_precision, to_surdvector
from httk.atomistic.models.moments.backend import SiteMomentsBackend


class CartesianSiteMoments(SiteMomentsBackend):
    """Per-site Cartesian magnetic moments, exactly held in Bohr magnetons."""

    kind = "cartesian"
    _cartesian_moments: SurdVector

    def __init__(self, moments: VectorLike, precision: Any = None) -> None:
        value = to_surdvector(moments)
        if len(value.dim) != 2 or value.dim[1] != 3:
            raise ValueError("CartesianSiteMoments moments must be an Nx3 vector-like")
        self._cartesian_moments = value
        self._precision = to_precision(precision)

    @property
    def cartesian_moments(self) -> SurdVector:
        """The exact Nx3 Cartesian moments, one row per site."""
        return self._cartesian_moments

    @property
    def precision(self) -> fractions.Fraction | None:
        return self._precision

    def __len__(self) -> int:
        return self._cartesian_moments.dim[0]

    def __eq__(self, other: object) -> bool:
        """Equality is on components only; stated precision is provenance metadata."""
        if not isinstance(other, CartesianSiteMoments):
            return NotImplemented
        return self._cartesian_moments == other._cartesian_moments

    def __hash__(self) -> int:
        return hash(self._cartesian_moments)

    def __repr__(self) -> str:
        return f"CartesianSiteMoments(moments={self._cartesian_moments!r})"
