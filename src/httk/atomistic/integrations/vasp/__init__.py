"""Expose VASP structure and trajectory integrations."""

from .structure import VASPStructure
from .trajectory import VASPTrajectory

__all__ = ["VASPStructure", "VASPTrajectory"]
