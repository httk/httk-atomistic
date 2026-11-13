"""
Backend wrapping cell parameters (a, b, c, alpha, beta, gamma).
"""

import fractions
from typing import Any, Self

from httk.core import SurdScalar, SurdVector, exactmath

from httk.atomistic.models._vector_guards import is_params6, to_fracvector, to_precision
from httk.atomistic.models.cell.backend import CellBackend

# Deterministic precision for the rational fallback when an angle is not a special angle
# (a multiple of 15 or 36 degrees) and so has no exact cosine in the surd field.
_PARAMS_PREC = exactmath.default_accuracy


def _rcos(deg: fractions.Fraction) -> fractions.Fraction:
    """A deterministic rational cosine of ``deg`` degrees: exact where rational, else at ``_PARAMS_PREC``."""
    return fractions.Fraction(exactmath.cos(deg, degrees=True, prec=_PARAMS_PREC, limit=False, exact=False))


def _rsin(deg: fractions.Fraction) -> fractions.Fraction:
    """A deterministic rational sine of ``deg`` degrees: exact where rational, else at ``_PARAMS_PREC``."""
    return fractions.Fraction(exactmath.sin(deg, degrees=True, prec=_PARAMS_PREC, limit=False, exact=False))


def _rhombohedral_basis(a: fractions.Fraction, alpha: fractions.Fraction) -> SurdVector:
    """
    The symmetric basis of the rhombohedral-axes cell ``(a, a, a, alpha, alpha, alpha)``.

    The threefold axis is along z: ``v_k = a (r cos phi_k, r sin phi_k, h)`` for ``phi_k = 0, 120,
    240`` degrees, with the rational point ``h = (1 - t**2)/(1 + t**2)``, ``r = 2t/(1 + t**2)`` on
    the unit circle. So ``|v_k|**2 == a**2`` and ``v_i . v_j == a**2 (h**2 - r**2/2)`` exactly, and
    the entries lie in Q(sqrt(3)). The rational ``t`` solves ``h**2 - r**2/2 = cos(alpha)``, i.e.
    ``h = sqrt((1 + 2 cos(alpha))/3)`` and ``t = sqrt((1 - h)/(1 + h))``, with each of the cosine
    and the two roots rounded within ``_PARAMS_PREC / 16``; the realized cosine
    ``h**2 - r**2/2`` is then within ``_PARAMS_PREC`` of ``cos(alpha)``.
    """
    prec = _PARAMS_PREC / 16
    cos_a = fractions.Fraction(exactmath.cos(alpha, degrees=True, prec=prec, limit=False, exact=False))
    h = fractions.Fraction(exactmath.sqrt((1 + 2 * cos_a) / 3, prec=prec, limit=False, exact=False))
    t = fractions.Fraction(exactmath.sqrt((1 - h) / (1 + h), prec=prec, limit=False, exact=False))
    ar, ah = a * 2 * t / (1 + t * t), a * (1 - t * t) / (1 + t * t)

    def scalar(value: fractions.Fraction, radicand: int = 1) -> SurdScalar:
        return SurdVector.from_radicand_map({radicand: value})._as_scalar()

    grid = [
        [scalar(ar), scalar(fractions.Fraction(0)), scalar(ah)],
        [scalar(-ar / 2), scalar(ar / 2, 3), scalar(ah)],
        [scalar(-ar / 2), scalar(-ar / 2, 3), scalar(ah)],
    ]
    return SurdVector._from_scalar_grid(grid, (3, 3))


def _params_to_basis(params: tuple[fractions.Fraction, ...], *, standard: bool = False) -> SurdVector:
    """
    Build the cell basis from ``(a, b, c, alpha, beta, gamma)``.

    Standard orientation: first cell vector along x, second in the xy-plane. When all of
    ``cos(alpha)``, ``cos(beta)``, ``cos(gamma)``, ``sin(gamma)`` are exact in the surd field (Niven
    angles) AND the resulting
    ``cz**2`` is rational, the basis is exact (radicals intact). Otherwise it falls back
    **completely** to a deterministic rational matrix — never mixing surd and approximate entries.
    In that fallback a cosine or sine whose value is rational (0, ±1/2 or ±1, e.g. ``cos 90°``,
    ``cos 120°``, ``sin 30°``) is exact, and only the irrational ones are approximated at
    ``_PARAMS_PREC``, so a monoclinic cell keeps its exact zeros. A rhombohedral-axes cell does not
    keep equal edge lengths there (``sin(gamma)`` and ``cz`` are approximated separately), so unless
    ``standard`` is set such a cell (``a == b == c``, ``alpha == beta == gamma``) is instead given
    the symmetric orientation of :func:`_rhombohedral_basis`, with exactly equal edges and angles.
    ``standard=True`` is for Cartesian data stated in the standard frame (CP2K coordinates, mCIF
    Cartesian moments).
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
            zero = SurdScalar._of({}, ())
            av = SurdVector(a)._as_scalar()
            bv = SurdVector(b)._as_scalar()
            cv = SurdVector(c)._as_scalar()
            cz = (cv * SurdVector.sqrt_of(cz_sq._rational_fraction()))._as_scalar()
            grid = [
                [av, zero, zero],
                [(bv * cos_g)._as_scalar(), (bv * sin_g)._as_scalar(), zero],
                [(cv * cos_b)._as_scalar(), (cv * cy)._as_scalar(), cz],
            ]
            return SurdVector._from_scalar_grid(grid, (3, 3))
    if not standard and a == b == c and alpha == beta == gamma:
        return _rhombohedral_basis(a, alpha)
    # Fully deterministic rational fallback (a non-Niven angle, or an irrational cz**2).
    ca, cb, cg = _rcos(alpha), _rcos(beta), _rcos(gamma)
    sg = _rsin(gamma)
    cy_r = (ca - cb * cg) / sg
    cz_sq_r = max(fractions.Fraction(0), fractions.Fraction(1) - cb * cb - cy_r * cy_r)
    cz_r = c * fractions.Fraction(exactmath.sqrt(cz_sq_r, prec=_PARAMS_PREC, limit=False, exact=False))
    rows = [
        [a, fractions.Fraction(0), fractions.Fraction(0)],
        [b * cg, b * sg, fractions.Fraction(0)],
        [c * cb, c * cy_r, cz_r],
    ]
    return SurdVector(rows)


class CellParams(CellBackend):
    r"""
    Backend for a cell backed by cell parameters ``(a, b, c, alpha, beta, gamma)``.

    The native representation is a flat length-6 vector-like of the cell-vector lengths
    ``a``/``b``/``c`` and the angles ``alpha``/``beta``/``gamma`` in degrees, stored as exact
    :class:`~fractions.Fraction` values (converted through :class:`~httk.core.FracVector`, so a
    string such as ``"3.567"`` is kept exactly as written). The exact ``basis`` is derived lazily and
    cached using the standard crystallographic orientation convention (first cell vector along x,
    second in the xy-plane); for the common Niven angles it is exact (radicals intact), and
    otherwise a deterministic rational approximation in which the cosines and sines that are 0,
    ±1/2 or ±1 stay exact, so a monoclinic cell keeps its right angles. The exception is a
    rhombohedral-axes cell (``a == b == c``, ``alpha == beta == gamma``) that would be
    approximated: it gets the symmetric orientation instead, the threefold axis along z and the
    first vector in the xz-plane, with exactly equal edge lengths and angles (entries in
    Q(sqrt(3)), the common cosine within ``1e-10``), so its metric keeps the threefold axis. Since
    parameters carry no separate length factor, ``scale`` is the exact ``1`` and
    ``unscaled_basis == basis``. Parameters carry neither orientation nor periodicity, so a cell
    → parameters → cell round-trip reproduces lengths and angles, and reproduces volume only for
    a fully periodic source. Reconstruction uses the fully periodic default and therefore discards
    any non-3D periodicity as well as the original orientation.
    ``unwrap`` returns the original raw object.

    :param obj: The six cell parameters in crystallographic order.
    :param \**hints: Backend-selection hints.
    """

    _raw: Any
    _params: tuple[fractions.Fraction, ...]
    _basis_cache: SurdVector | None
    _precision: fractions.Fraction | None

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt six cell parameters.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "params") != "params":
            return None
        if not is_params6(obj):
            return None
        return cls(obj, **hints)

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
        self._precision = to_precision(hints.get("precision"))

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
        """Return the basis derived from the stored parameters.

        :return: The cell vectors, in the standard orientation except for an approximated
            rhombohedral-axes cell (see the class docstring).
        """
        if self._basis_cache is None:
            self._basis_cache = _params_to_basis(self._params)
        return self._basis_cache

    @property
    def scale(self) -> SurdScalar:
        """Return the unit scale factor.

        :return: The factor applied to ``unscaled_basis``.
        """
        return SurdVector.one()

    @property
    def unscaled_basis(self) -> SurdVector:
        """Return the parameter-derived basis before scaling.

        :return: The cell vectors.
        """
        return self.basis

    @property
    def precision(self) -> fractions.Fraction | None:
        """Return the source precision supplied with these cell parameters.

        :return: The absolute length precision, or ``None`` when unstated.
        """
        return self._precision

    @property
    def lengths(self) -> tuple[SurdScalar, ...]:
        """Return the natively stored cell-vector lengths.

        :return: The exact stored ``a``, ``b``, and ``c`` lengths.
        """
        return tuple(SurdVector(value)._as_scalar() for value in self._params[:3])

    @property
    def angles(self) -> tuple[fractions.Fraction, ...]:
        """Return the natively stored cell angles in degrees.

        :return: The exact stored ``alpha``, ``beta``, and ``gamma`` angles.
        """
        return self._params[3:]

    @property
    def params(self) -> tuple[fractions.Fraction, ...]:
        """The stored ``(a, b, c, alpha, beta, gamma)`` in degrees.

        :return: The exact cell parameters.
        """
        return self._params

    def unwrap(self) -> Any:
        """Return the original parameter object.

        :return: The raw parameter representation.
        """
        return self._raw
