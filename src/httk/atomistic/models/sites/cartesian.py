"""
Backend holding Cartesian site coordinates exactly as given.
"""

import fractions
from functools import cached_property
from typing import Any, Self

from httk.core import FracVector, SurdVector, VectorLike, exactmath

from httk.atomistic.models._vector_guards import exact_literal, to_precision, to_surdvector
from httk.atomistic.models.sites.backend import SitesBackend

_FALLBACK_PREC = exactmath.default_accuracy


def _cartesian_precision_factor(basis: SurdVector) -> fractions.Fraction:
    """The factor turning a Cartesian bound in ``basis`` units into a fractional one."""
    # A Cartesian row vector transforms as reduced = Cartesian * basis^-1. If each
    # Cartesian component carries the same absolute bound p, reduced component j is
    # bounded by p * sum_i |basis^-1[i, j]|. The largest column sum is therefore the
    # conservative scalar precision needed by Sites. This is exact for a POSCAR basis,
    # whose decimal tokens are rational, and remains safe for skewed cells where dividing
    # by the shortest lattice-vector length can severely underestimate the uncertainty.
    inverse_surd = basis.inv()
    if inverse_surd.is_rational:
        inverse = inverse_surd.coefficient(1).to_fractions()
    else:
        # ponytail: an irrational (exact-surd) basis bounds through a deterministic rational
        # approximation padded by its accuracy; exact interval arithmetic if that ever matters.
        inverse = [
            [abs(value) + _FALLBACK_PREC for value in row] for row in inverse_surd.to_fractions_approx(_FALLBACK_PREC)
        ]
    return max(sum((abs(inverse[row][column]) for row in range(3)), start=fractions.Fraction()) for column in range(3))


class CartesianSites(SitesBackend):
    """
    Sites backend holding Cartesian coordinates (in ångström) exactly as given.

    The canonical ``reduced_coords`` are derived exactly as ``cartesian_coords * basis^-1``
    (row-vector convention) on first access, so the original Cartesian numbers stay available
    unchanged, e.g. for writing a Cartesian POSCAR back as it was read.

    :param cartesian_coords: The Nx3 Cartesian coordinates, one site per row.
    :param basis: The exact 3x3 lattice vectors (rows) that define the frame.
    :param cartesian_precision: How precisely the coordinates were stated, as a length in
        ångström, or ``None`` when unknown.
    :raises ValueError: If the coordinates are not Nx3 or the basis is not a non-degenerate 3x3.
    """

    _cartesian_coords: SurdVector
    _basis: SurdVector
    _cartesian_precision: fractions.Fraction | None

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt an existing Cartesian sites backend by identity; raw data is never adopted.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: ``obj`` itself, or ``None`` when this backend declines it.
        """
        if hints and hints.get("kind", "cartesian") != "cartesian":
            return None
        return obj if isinstance(obj, cls) else None

    def __init__(self, cartesian_coords: VectorLike, basis: VectorLike, cartesian_precision: Any = None) -> None:
        coords = to_surdvector(cartesian_coords)
        if not (len(coords.dim) == 2 and coords.dim[1] == 3):
            raise ValueError("CartesianSites cartesian_coords must be a non-empty Nx3 vector-like")
        frame = to_surdvector(basis)
        if frame.dim != (3, 3):
            raise ValueError("CartesianSites basis must be a 3x3 vector-like")
        if frame.det().is_zero():
            raise ValueError("CartesianSites basis must be non-degenerate")
        self._cartesian_coords = coords
        self._basis = frame
        self._cartesian_precision = to_precision(cartesian_precision)

    @property
    def cartesian_coords(self) -> SurdVector:
        """The Nx3 Cartesian coordinates exactly as given."""
        return self._cartesian_coords

    @property
    def basis(self) -> SurdVector:
        """The 3x3 lattice vectors defining the Cartesian frame."""
        return self._basis

    @property
    def cartesian_precision(self) -> fractions.Fraction | None:
        """How precisely the coordinates were stated, as a length in ångström, or ``None``."""
        return self._cartesian_precision

    @cached_property
    def reduced_coords(self) -> FracVector:  # pyright: ignore[reportIncompatibleMethodOverride]
        """The exact reduced coordinates ``cartesian_coords * basis^-1``.

        :return: The reduced coordinates as an exact ``FracVector``.
        :raises ValueError: If the reduced coordinates are irrational (no exact ``FracVector``).
        """
        reduced = self._cartesian_coords * self._basis.inv()
        if not reduced.is_rational:
            raise ValueError(
                "CartesianSites reduced coordinates are irrational in this basis; "
                "reduced coordinates must be exact rationals"
            )
        return FracVector(reduced)

    @property
    def precision(self) -> fractions.Fraction | None:
        """The fractional precision bounding ``cartesian_precision`` in this basis, or ``None``."""
        if self._cartesian_precision is None:
            return None
        return self._cartesian_precision * _cartesian_precision_factor(self._basis)

    def __repr__(self) -> str:
        precision = self._cartesian_precision
        return (
            f"CartesianSites(cartesian_coords={exact_literal(self._cartesian_coords)}, "
            f"basis={exact_literal(self._basis)}, "
            f"cartesian_precision={None if precision is None else exact_literal(precision)})"
        )
