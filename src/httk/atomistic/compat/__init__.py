"""External-library compatibility bridges."""

from .ase import ASEAtoms, ASEAtomsProtocol

__all__ = ["ASEAtoms", "ASEAtomsProtocol"]

try:
    from .ase import ASEAtomsView  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("ASEAtomsView")
