"""External-library compatibility bridges."""

from .ase import ASEAtoms, ASEAtomsProtocol
from .pymatgen import PymatgenStructure, PymatgenStructureProtocol

__all__ = ["ASEAtoms", "ASEAtomsProtocol", "PymatgenStructure", "PymatgenStructureProtocol"]

try:
    from .ase import ASEAtomsView  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("ASEAtomsView")

try:
    from .pymatgen import PymatgenStructureView  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("PymatgenStructureView")
