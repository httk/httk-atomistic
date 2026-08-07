"""Expose external-library integration bridges."""

from .ase import ASEAtoms, ASEAtomsProtocol
from .pymatgen import PymatgenStructure, PymatgenStructureProtocol
from .vasp import VASPStructure, VASPTrajectory

__all__ = [
    "ASEAtoms",
    "ASEAtomsProtocol",
    "PymatgenStructure",
    "PymatgenStructureProtocol",
    "VASPStructure",
    "VASPTrajectory",
]

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
