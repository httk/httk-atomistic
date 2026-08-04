"""Collinear per-site magnetic-moment backend."""

import fractions
from typing import Any

from httk.core import FracVector, SurdVector, VectorLike

from httk.atomistic.models._vector_guards import to_fracvector, to_precision
from httk.atomistic.models.moments.backend import SiteMomentsBackend


class CollinearSiteMoments(SiteMomentsBackend):
    """Signed per-site scalar moments with no assigned Cartesian axis."""

    kind = "collinear"
    _collinear_moments: FracVector

    def __init__(self, moments: VectorLike, precision: Any = None) -> None:
        value = to_fracvector(moments)
        if len(value.dim) != 1:
            raise ValueError("CollinearSiteMoments moments must be an N-dimensional vector-like")
        self._collinear_moments = value
        self._precision = to_precision(precision)

    @property
    def collinear_moments(self) -> FracVector:
        """The exact signed scalar moment for each site."""
        return self._collinear_moments

    @property
    def cartesian_moments(self) -> SurdVector:
        raise ValueError(
            "a collinear scalar carries no axis and none is fabricated; "
            "construct a CartesianSiteMoments explicitly to assign one"
        )

    @property
    def precision(self) -> fractions.Fraction | None:
        return self._precision

    def __len__(self) -> int:
        return len(self._collinear_moments)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CollinearSiteMoments):
            return NotImplemented
        return self._collinear_moments == other._collinear_moments

    def __hash__(self) -> int:
        return hash(self._collinear_moments)

    def __repr__(self) -> str:
        return f"CollinearSiteMoments(moments={self._collinear_moments!r})"
