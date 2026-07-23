"""
The accepted-input union for structure functions in httk-atomistic.
"""

from typing import Any

import httk.core

from . import structure, structure_backend, structure_view

# A structure is any structure backend/view, a Structure, or an spglib-like
# (lattice, positions, numbers) triple whose lattice and positions are vector-like.
type StructureLike = (
    structure_backend.StructureBackend
    | structure_view.StructureView
    | structure.Structure
    | tuple[httk.core.VectorLike, httk.core.VectorLike, Any]
    | list[Any]
)
