"""The one home of httk's reduced/anonymous formula notation."""

import re
from collections.abc import Sequence
from fractions import Fraction
from functools import reduce
from math import gcd

from httk.atomistic.elements import SYMBOLS

_ELEMENT_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*)")
_ANONYMOUS_TOKEN = re.compile(r"([A-Z][a-z]*)([0-9]*)")
_ELEMENTS = frozenset(SYMBOLS)


def anonymous_symbol(index: int) -> str:
    """The unbounded OPTIMADE anonymous symbol for zero-based *index*."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("anonymous symbol index must be a non-negative integer")
    head = chr(ord("A") + index % 26)
    tail_number = index // 26
    tail: list[str] = []
    while tail_number:
        tail_number -= 1
        tail.append(chr(ord("a") + tail_number % 26))
        tail_number //= 26
    return head + "".join(reversed(tail))


def reduced_coefficients(ratios: Sequence[Fraction]) -> tuple[int, ...] | None:
    """Return the least common integer coefficients for exact elemental ratios."""
    if not ratios:
        return None
    denominator = 1
    for ratio in ratios:
        denominator = denominator * ratio.denominator // gcd(denominator, ratio.denominator)
    values = tuple(int(ratio * denominator) for ratio in ratios)
    common = reduce(gcd, values)
    return tuple(value // common for value in values)


def render_reduced(coefficients: Sequence[tuple[str, int]]) -> str:
    """Render element symbols and reduced integer coefficients in the given order."""
    return "".join(element + (str(amount) if amount != 1 else "") for element, amount in coefficients)


def render_anonymous(counts: Sequence[int]) -> str:
    """Render descending-sorted integer counts using OPTIMADE anonymous symbols."""
    return "".join(anonymous_symbol(index) + (str(count) if count != 1 else "") for index, count in enumerate(counts))


def parse_reduced_formula(text: str) -> tuple[tuple[str, int], ...]:
    """Parse a strictly canonical reduced OPTIMADE chemical formula.

    The reduced and anonymous grammars are disjoint by construction: an anonymous
    label may have an arbitrary lowercase tail, while an element symbol has at most one.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("reduced formula must be a non-empty string")
    result: list[tuple[str, int]] = []
    position = 0
    while position < len(text):
        match = _ELEMENT_TOKEN.match(text, position)
        if match is None:
            raise ValueError(f"invalid reduced formula token at position {position}")
        symbol, digits = match.groups()
        if symbol not in _ELEMENTS:
            raise ValueError(f"reduced formula contains unknown element symbol {symbol!r}")
        if any(existing == symbol for existing, _ in result):
            raise ValueError(f"reduced formula repeats element symbol {symbol!r}")
        if result and symbol <= result[-1][0]:
            raise ValueError("reduced formula element symbols must be strictly alphabetical")
        if not digits:
            count = 1
        else:
            count = int(digits)
            if count < 2:
                raise ValueError("reduced formula explicit counts must be at least 2; explicit 1 is invalid")
        result.append((symbol, count))
        position = match.end()
    if gcd(*(count for _, count in result)) != 1:
        raise ValueError("reduced formula coefficients must have greatest common divisor 1")
    return tuple(result)


def parse_anonymous_formula(text: str) -> tuple[tuple[str, int], ...]:
    """Parse a strictly canonical OPTIMADE anonymous chemical formula."""
    if not isinstance(text, str) or not text:
        raise ValueError("anonymous formula must be a non-empty string")
    result: list[tuple[str, int]] = []
    position = 0
    while position < len(text):
        match = _ANONYMOUS_TOKEN.match(text, position)
        if match is None:
            raise ValueError(f"invalid anonymous formula token at position {position}")
        label, digits = match.groups()
        expected = anonymous_symbol(len(result))
        if label != expected:
            raise ValueError(f"anonymous formula labels must be consecutive starting at {expected!r}")
        if not digits:
            count = 1
        else:
            count = int(digits)
            if count < 2:
                raise ValueError("anonymous formula explicit counts must be at least 2; explicit 1 is invalid")
        if result and count > result[-1][1]:
            raise ValueError("anonymous formula coefficients must be in non-increasing order")
        result.append((label, count))
        position = match.end()
    if gcd(*(count for _, count in result)) != 1:
        raise ValueError("anonymous formula coefficients must have greatest common divisor 1")
    return tuple(result)


def try_parse_reduced(text: str) -> tuple[tuple[str, int], ...] | None:
    """Return canonical reduced coefficients, or ``None`` when *text* is not one."""
    try:
        return parse_reduced_formula(text)
    except ValueError:
        return None


def try_parse_anonymous(text: str) -> tuple[tuple[str, int], ...] | None:
    """Return canonical anonymous coefficients, or ``None`` when *text* is not one."""
    try:
        return parse_anonymous_formula(text)
    except ValueError:
        return None
