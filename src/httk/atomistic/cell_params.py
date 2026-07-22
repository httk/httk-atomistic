"""
Backend wrapping cell parameters (a, b, c, alpha, beta, gamma).
"""

import math
from typing import Any

from .cell_backend import CellBackend


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_params(obj: Any) -> bool:
    return isinstance(obj, (list, tuple)) and len(obj) == 6 and all(_is_number(x) for x in obj)


def _params_to_matrix(params: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    a, b, c, alpha, beta, gamma = params
    cos_alpha = math.cos(math.radians(alpha))
    cos_beta = math.cos(math.radians(beta))
    cos_gamma = math.cos(math.radians(gamma))
    sin_gamma = math.sin(math.radians(gamma))
    cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sq = 1.0 - cos_beta * cos_beta - cy * cy
    return (
        (a, 0.0, 0.0),
        (b * cos_gamma, b * sin_gamma, 0.0),
        (c * cos_beta, c * cy, c * math.sqrt(cz_sq)),
    )


class CellParams(CellBackend):
    """
    Backend for a cell backed by cell parameters ``(a, b, c, alpha, beta, gamma)``.

    The native representation is a flat length-6 list or tuple of the cell-vector
    lengths ``a``/``b``/``c`` and the angles ``alpha``/``beta``/``gamma`` in degrees.
    The ``matrix`` is derived lazily and cached using the standard crystallographic
    orientation convention (the first cell vector along x, the second in the xy-plane);
    since parameters carry no orientation, converting a cell to parameters and back
    reproduces its lengths, angles, and volume, but not its original orientation.
    ``unwrap`` returns the original raw object.
    """

    _raw: Any
    _params: tuple[float, ...]
    _matrix_cache: tuple[tuple[float, ...], ...] | None

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "params") != "params":
            return None
        if not _is_params(obj):
            return None
        return super().__new__(cls)

    def __init__(self, obj: Any, **hints: Any) -> None:
        params = tuple(float(x) for x in obj)
        a, b, c, alpha, beta, gamma = params
        if a <= 0.0 or b <= 0.0 or c <= 0.0:
            raise ValueError("Cell parameter lengths a, b, c must be positive")
        if not all(0.0 < angle < 180.0 for angle in (alpha, beta, gamma)):
            raise ValueError("Cell parameter angles alpha, beta, gamma must be strictly between 0 and 180 degrees")
        cos_alpha = math.cos(math.radians(alpha))
        cos_beta = math.cos(math.radians(beta))
        cos_gamma = math.cos(math.radians(gamma))
        volume_factor = (
            1.0
            - cos_alpha * cos_alpha
            - cos_beta * cos_beta
            - cos_gamma * cos_gamma
            + 2.0 * cos_alpha * cos_beta * cos_gamma
        )
        if volume_factor <= 0.0:
            raise ValueError("Cell parameter angles do not describe a valid (non-degenerate) cell")
        self._raw = obj
        self._params = params
        self._matrix_cache = None

    @property
    def matrix(self) -> tuple[tuple[float, ...], ...]:
        if self._matrix_cache is None:
            self._matrix_cache = _params_to_matrix(self._params)
        return self._matrix_cache

    @property
    def params(self) -> tuple[float, ...]:
        """The stored ``(a, b, c, alpha, beta, gamma)`` as a tuple of floats (angles in degrees)."""
        return self._params

    def unwrap(self) -> Any:
        return self._raw
