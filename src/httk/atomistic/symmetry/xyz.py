"""Exact parsers for crystallographic coordinate-operation notation."""

import re
from collections.abc import Sequence
from decimal import Decimal
from fractions import Fraction

from .affine_operation import AffineOperation

__all__ = ["operation_from_xyz", "operation_from_xyzt", "parse_linear_expression"]

_NUMBER = r"(?:\d+/\d+|\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_SUPERSPACE_VARS = ("x1", "x2", "x3", "x4", "x5", "x6")


def _fraction(text: str) -> Fraction:
    return Fraction(text) if "/" in text else Fraction(Decimal(text))


def _parse_linear_expression(
    expression: str, variables: Sequence[str], *, integer_coefficients: bool = False
) -> tuple[tuple[Fraction, ...], Fraction]:
    text = expression.replace(" ", "")
    if not text:
        raise ValueError("Empty symmetry expression")
    if text[0] not in "+-":
        text = "+" + text

    names = "|".join(re.escape(name) for name in sorted(variables, key=len, reverse=True))
    token = re.compile(rf"([+-])(?:(?:(?P<coefficient>{_NUMBER})\*?)?(?P<variable>{names})|(?P<constant>{_NUMBER}))")
    coefficients = {name: Fraction(0) for name in variables}
    constant = Fraction(0)
    position = 0
    for match in token.finditer(text):
        if match.start() != position:
            raise ValueError(f"Unparsed tail in {expression!r} near {text[position:]!r}")
        position = match.end()
        sign = 1 if match.group(1) == "+" else -1
        variable = match.group("variable")
        if variable is None:
            constant += sign * _fraction(match.group("constant"))
            continue
        coefficient = _fraction(match.group("coefficient")) if match.group("coefficient") else Fraction(1)
        if integer_coefficients and coefficient.denominator != 1:
            raise ValueError(f"Non-integer coefficient on {variable} in {expression!r}")
        coefficients[variable] += sign * coefficient

    if position != len(text):
        raise ValueError(f"Unparsed tail in {expression!r} near {text[position:]!r}")
    return tuple(coefficients[name] for name in variables), constant


def parse_linear_expression(
    expression: str, allowed_vars: Sequence[str] = _SUPERSPACE_VARS
) -> tuple[tuple[int, ...], Fraction]:
    """Parse one superspace linear expression into integer coefficients and a translation.

    The parser accepts the six superspace variables ``x1`` through ``x6`` by default and
    preserves an exact rational translation.

    :param expression: The linear expression to parse.
    :param allowed_vars: The variable names accepted in ``expression``.
    :return: The integer coefficients in ``allowed_vars`` and the exact translation.
    :raises ValueError: If the expression is empty, malformed, or has a non-integer
        variable coefficient.
    """
    coefficients, constant = _parse_linear_expression(expression, allowed_vars, integer_coefficients=True)
    return tuple(coefficient.numerator for coefficient in coefficients), constant


def _operation(parts: Sequence[str]) -> AffineOperation:
    rows = []
    translations = []
    for part in parts:
        row, translation = _parse_linear_expression(part, ("x", "y", "z"))
        rows.append(row)
        translations.append(translation)
    return AffineOperation(rows, translations)


def operation_from_xyz(operation: str) -> AffineOperation:
    """Parse an exact three-coordinate crystallographic operation.

    :param operation: The comma-separated ``x,y,z`` operation string.
    :return: The exact affine operation.
    :raises ValueError: If the operation does not contain three valid coordinate
        expressions.
    """
    parts = tuple(part.strip() for part in operation.split(","))
    if len(parts) != 3:
        raise ValueError(f"Expected three coordinates in symmetry operation: {operation!r}")
    return _operation(parts)


def operation_from_xyzt(operation: str) -> tuple[AffineOperation, int]:
    """Parse an exact magnetic operation and return its time-reversal flag.

    The first three comma-separated fields use crystallographic ``x,y,z`` grammar; the
    fourth field must be ``+1`` or ``-1``.

    :param operation: The comma-separated ``x,y,z,time-reversal`` operation string.
    :return: The affine operation and its time-reversal sign.
    :raises ValueError: If the operation does not contain three coordinate expressions and
        a valid time-reversal flag.
    """
    parts = tuple(part.strip() for part in operation.split(","))
    if len(parts) != 4:
        raise ValueError(f"Expected three coordinates and a time-reversal flag: {operation!r}")
    try:
        time_reversal = int(parts[3])
    except ValueError as error:
        raise ValueError(f"Invalid time-reversal flag in {operation!r}") from error
    if time_reversal not in (-1, 1):
        raise ValueError(f"Time-reversal flag must be +1 or -1 in {operation!r}")
    return _operation(parts[:3]), time_reversal
