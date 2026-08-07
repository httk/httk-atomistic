"""The one home of httk's reduced/anonymous formula notation."""

from collections.abc import Sequence
from fractions import Fraction
from functools import reduce
from math import gcd


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
