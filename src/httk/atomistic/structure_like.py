"""
The accepted-input union for structure functions in httk-atomistic.
"""

from typing import Any

from . import structure, structure_backend, structure_view

type StructureLike = (
    structure_backend.StructureBackend
    | structure_view.StructureView
    | structure.Structure
    | tuple[Any, Any, Any]
    | list[Any]
)
