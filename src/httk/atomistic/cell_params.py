"""
Backend wrapping cell parameters (a, b, c, alpha, beta, gamma).
"""

import fractions
from typing import Any

from httk.core import SurdScalar, SurdVector
from httk.core.vectors import exactmath

from ._vector_guards import is_params6, to_fracvector
from .cell_backend import CellBackend

# Deterministic precision for the rational fallback when an angle is not a special angle
# (a multiple of 15 or 36 degrees) and so has no exact cosine in the surd field.
_PARAMS_PREC = exactmath.default_accuracy


def _rcos(deg: fractions.Fraction) -> fractions.Fraction:
    """A deterministic rational cosine of ``deg`` degrees at ``_PARAMS_PREC``."""
    return fractions.Fraction(exactmath.cos(deg, degrees=True, prec=_PARAMS_PREC, limit=False))


def _rsin(deg: fractions.Fraction) -> fractions.Fraction:
    """A deterministic rational sine of ``deg`` degrees at ``_PARAMS_PREC``."""
    return fractions.Fraction(exactmath.sin(deg, degrees=True, prec=_PARAMS_PREC, limit=False))


def _params_to_basis(params: tuple[fractions.Fraction, ...]) -> SurdVector:
    """
    Build the standard-orientation cell basis from ``(a, b, c, alpha, beta, gamma)``.

    First cell vector along x, second in the xy-plane. When all of ``cos(alpha)``, ``cos(beta)``,
    ``cos(gamma)``, ``sin(gamma)`` are exact in the surd field (Niven angles) AND the resulting
    ``cz**2`` is rational, the basis is exact (radicals intact). Otherwise it falls back
    **completely** to a deterministic rational matrix at ``_PARAMS_PREC`` — never mixing exact
    and approximate entries.
    """
    a, b, c, alpha, beta, gamma = params
    cos_a = SurdScalar.cos_degrees(alpha)
    cos_b = SurdScalar.cos_degrees(beta)
    cos_g = SurdScalar.cos_degrees(gamma)
    sin_g = SurdScalar.sin_degrees(gamma)
    if cos_a is not None and cos_b is not None and cos_g is not None and sin_g is not None:
        cy = ((cos_a - cos_b * cos_g) / sin_g)._as_scalar()
        cz_sq = (SurdVector.one() - cos_b * cos_b - cy * cy)._as_scalar()
        if cz_sq.is_rational:
            zero = SurdScalar({}, ())
            av = SurdVector.create(a)._as_scalar()
            bv = SurdVector.create(b)._as_scalar()
            cv = SurdVector.create(c)._as_scalar()
            cz = (cv * SurdVector.sqrt_of(cz_sq._rational_fraction()))._as_scalar()
            grid = [
                [av, zero, zero],
                [(bv * cos_g)._as_scalar(), (bv * sin_g)._as_scalar(), zero],
                [(cv * cos_b)._as_scalar(), (cv * cy)._as_scalar(), cz],
            ]
            return SurdVector._from_scalar_grid(grid, (3, 3))
    # Fully deterministic rational fallback (a non-Niven angle, or an irrational cz**2).
    ca, cb, cg = _rcos(alpha), _rcos(beta), _rcos(gamma)
    sg = _rsin(gamma)
    cy_r = (ca - cb * cg) / sg
    cz_sq_r = max(fractions.Fraction(0), fractions.Fraction(1) - cb * cb - cy_r * cy_r)
    cz_r = c * fractions.Fraction(exactmath.sqrt(cz_sq_r, prec=_PARAMS_PREC, limit=False))
    rows = [
        [a, fractions.Fraction(0), fractions.Fraction(0)],
        [b * cg, b * sg, fractions.Fraction(0)],
        [c * cb, c * cy_r, cz_r],
    ]
    return SurdVector.create(rows)


class CellParams(CellBackend):
    """
    Backend for a cell backed by cell parameters ``(a, b, c, alpha, beta, gamma)``.

    The native representation is a flat length-6 vector-like of the cell-vector lengths
    ``a``/``b``/``c`` and the angles ``alpha``/``beta``/``gamma`` in degrees, stored as exact
    :class:`~fractions.Fraction` values (parsed via
    :func:`~httk.core.vectors.exactmath.any_to_fraction`). The exact ``basis`` is derived lazily and
    cached using the standard crystallographic orientation convention (first cell vector along x,
    second in the xy-plane); for the common Niven angles it is exact (radicals intact). Since
    parameters carry no separate length factor, ``scale`` is the exact ``1`` and
    ``unscaled_basis == basis``. Parameters carry no orientation, so a cell → parameters → cell
    round-trip reproduces lengths, angles, and volume, but not the original orientation.
    ``unwrap`` returns the original raw object.
    """

    _raw: Any
    _params: tuple[fractions.Fraction, ...]
    _basis_cache: SurdVector | None

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "params") != "params":
            return None
        if not is_params6(obj):
            return None
        return super().__new__(cls)

    def __init__(self, obj: Any, **hints: Any) -> None:
        params = tuple(to_fracvector(obj).to_fractions())
        a, b, c, alpha, beta, gamma = params
        if a <= 0 or b <= 0 or c <= 0:
            raise ValueError("Cell parameter lengths a, b, c must be positive")
        if not all(0 < angle < 180 for angle in (alpha, beta, gamma)):
            raise ValueError("Cell parameter angles alpha, beta, gamma must be strictly between 0 and 180 degrees")
        if not self._describes_valid_cell(alpha, beta, gamma):
            raise ValueError("Cell parameter angles do not describe a valid (non-degenerate) cell")
        self._raw = obj
        self._params = params
        self._basis_cache = None

    @staticmethod
    def _describes_valid_cell(alpha: fractions.Fraction, beta: fractions.Fraction, gamma: fractions.Fraction) -> bool:
        """Exact positivity of ``1 - cos^2 a - cos^2 b - cos^2 g + 2 cos a cos b cos g``."""
        cos_a = SurdScalar.cos_degrees(alpha)
        cos_b = SurdScalar.cos_degrees(beta)
        cos_g = SurdScalar.cos_degrees(gamma)
        if cos_a is not None and cos_b is not None and cos_g is not None:
            factor = (
                SurdVector.one() - cos_a * cos_a - cos_b * cos_b - cos_g * cos_g + 2 * cos_a * cos_b * cos_g
            )._as_scalar()
            return factor.sign() > 0
        ca, cb, cg = _rcos(alpha), _rcos(beta), _rcos(gamma)
        return (1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg) > 0

    @property
    def basis(self) -> SurdVector:
        if self._basis_cache is None:
            self._basis_cache = _params_to_basis(self._params)
        return self._basis_cache

    @property
    def scale(self) -> SurdScalar:
        return SurdVector.one()

    @property
    def unscaled_basis(self) -> SurdVector:
        return self.basis

    @property
    def params(self) -> tuple[fractions.Fraction, ...]:
        """The stored ``(a, b, c, alpha, beta, gamma)`` as exact Fractions (angles in degrees)."""
        return self._params

    def unwrap(self) -> Any:
        return self._raw
