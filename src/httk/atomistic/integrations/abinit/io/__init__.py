"""Read ABINIT inputs as structures and main outputs as trajectories, and write geometry-only inputs."""

from .inputs import read_abinit_input
from .outputs import read_abinit_output

__all__ = ["read_abinit_input", "read_abinit_output"]
