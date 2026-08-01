"""Exact value and source-precision helpers used by chemical components."""

import math
import re
from decimal import Decimal
from fractions import Fraction
from typing import Any

from httk.core import decimal_precision

__all__ = ["as_fraction", "as_precision", "normalization"]

_CIF_NUMBER = re.compile(
    r"^(?P<sign>[+-]?)(?P<mant>(?:\d+\.?|\d*\.\d+))(?:\((?P<esd>\d+)\))?(?:[eE](?P<exp>[+-]?\d+))?$"
)


def as_fraction(value: Any, *, field: str = "value") -> tuple[Fraction, Fraction | None]:
    """Return an exact central value and the precision implied by its spelling.

    Fractions and integers are assertions of exactness. Decimal strings, Decimal values,
    and floats retain their decimal-source width. CIF ``central(esd)`` spellings use the
    coarser of their final-digit width and stated ESD.
    """
    if isinstance(value, bool):
        raise TypeError(f"{field} cannot be a bool")
    if isinstance(value, Fraction):
        return value, None
    if isinstance(value, int):
        return Fraction(value), None
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        text = str(value)
        return Fraction(text), decimal_precision(text)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{field} must be finite")
        text = str(value)
        return Fraction(value), decimal_precision(text)
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a rational, integer, decimal string, or float")
    text = value.strip()
    if not text or text in {"?", "."}:
        raise ValueError(f"{field} is missing")
    match = _CIF_NUMBER.fullmatch(text)
    if match is None:
        try:
            return Fraction(text), None
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"{field} is not a rational number: {value!r}") from exc
    central = f"{match.group('sign')}{match.group('mant')}"
    exponent = int(match.group("exp") or 0)
    if exponent:
        central += f"e{exponent}"
    result = Fraction(central)
    width = decimal_precision(central)
    esd = match.group("esd")
    if esd is not None:
        mantissa = match.group("mant")
        places = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
        esd_width = Fraction(int(esd)) * Fraction(10) ** (exponent - places)
        width = esd_width if width is None or esd_width > width else width
    return result, width


def as_precision(value: Any, *, field: str = "precision") -> Fraction | None:
    """Return a positive exact absolute width, with ``None`` denoting exactness."""
    if value is None:
        return None
    precision, _ = as_fraction(value, field=field)
    if precision <= 0:
        raise ValueError(f"{field} must be positive")
    return precision


def normalization(
    values: tuple[Fraction, ...], precisions: tuple[Fraction | None, ...]
) -> tuple[bool, str, Fraction, Fraction | None]:
    """Whether one lies in the propagated absolute-width interval of a sum."""
    total = sum(values, Fraction())
    widths = [width for width in precisions if width is not None]
    width = sum(widths, Fraction()) if widths else None
    normalized = total == 1 if width is None else total - width <= 1 <= total + width
    if not normalized:
        return False, "outside_precision", total, width
    return True, "exact" if width is None else "within_precision", total, width
