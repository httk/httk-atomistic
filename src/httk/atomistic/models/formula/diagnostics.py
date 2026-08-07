"""Structured diagnostics produced while working with chemical formulas."""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class CompositionDiagnostic:
    """Record a non-fatal issue encountered while projecting composition.

    :param code: The stable diagnostic code.
    :param message: The human-readable diagnostic message.
    :param subject: The composition subject involved, if any.
    :param total: The calculated total, if applicable.
    :param width: The precision width used for the diagnostic, if applicable.
    """

    code: str
    message: str
    subject: str | None = None
    total: Fraction | None = None
    width: Fraction | None = None
