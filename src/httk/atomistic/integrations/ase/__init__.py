"""Expose optional ASE structure integrations."""

from .models import ASEAtoms, ASEAtomsProtocol

__all__ = ["ASEAtoms", "ASEAtomsProtocol"]

try:
    from .models import ASEAtomsView  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("ASEAtomsView")
