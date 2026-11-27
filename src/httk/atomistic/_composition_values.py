"""Exact value and source-precision helpers used by chemical components and the readers."""

import math
import re
from decimal import Decimal
from fractions import Fraction
from typing import Any

from httk.core import decimal_precision

__all__ = ["as_fraction", "as_precision", "bounded_fraction", "check_numeric_text", "normalization"]

# httk.core.exactmath's bound for untrusted numeric text (private there, so restated here).
_MAX_TEXT_DIGITS = 4300

_CIF_NUMBER = re.compile(
    r"^(?P<sign>[+-]?)(?P<mant>(?:\d+\.?|\d*\.\d+))(?:\((?P<esd>\d+)\))?(?:[eE](?P<exp>[+-]?\d+))?$"
)


def check_numeric_text(text: str) -> None:
    """Reject decimal text whose exact value would need more than 4300 digits.

    ``Fraction("1e999999999")`` builds a billion-digit integer; the digit count plus the
    exponent magnitude bounds that work. Rational text (``"a/b"``) carries no exponent and
    its integers are already bounded by Python's integer string-conversion limit.

    :param text: Numeric text as read from a file.
    :raises ValueError: If its digit count plus exponent magnitude exceeds 4300.
    """
    if "/" in text:
        return
    mantissa, _, exponent = text.upper().partition("E")
    try:
        magnitude = abs(int(exponent)) if exponent.strip() else 0
    except ValueError:
        return  # malformed; the caller's parser rejects it
    if sum(c.isdigit() for c in mantissa) + magnitude > _MAX_TEXT_DIGITS:
        raise ValueError(f"numeric text {text[:60]!r} exceeds {_MAX_TEXT_DIGITS} digits of exact value")


def bounded_fraction(text: str) -> Fraction:
    """Parse numeric text exactly, refusing values too large to build (see :func:`check_numeric_text`).

    :param text: Text accepted by :class:`fractions.Fraction`.
    :return: The exact value.
    :raises ValueError: If the text is malformed or its exact value exceeds 4300 digits.
    """
    check_numeric_text(text)
    return Fraction(text)


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
        return bounded_fraction(text), decimal_precision(text)
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a rational, integer, decimal string, or float")
    text = value.strip()
    if not text or text in {"?", "."}:
        raise ValueError(f"{field} is missing")
    match = _CIF_NUMBER.fullmatch(text)
    if match is None:
        try:
            return bounded_fraction(text), None
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"{field} is not a rational number: {value!r}") from exc
    central = f"{match.group('sign')}{match.group('mant')}"
    exponent = int(match.group("exp") or 0)
    if exponent:
        central += f"e{exponent}"
    result = bounded_fraction(central)
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
