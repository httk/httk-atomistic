"""Structured diagnostics produced while working with chemical formulas."""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class CompositionDiagnostic:
    """A machine-readable non-fatal issue encountered while projecting composition."""

    code: str
    message: str
    subject: str | None = None
    total: Fraction | None = None
    width: Fraction | None = None
