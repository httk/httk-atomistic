"""
httk-atomistic: crystal structure representations for httk v2.

Currently provides the Structure domain in two representations, following the
httk-core view/backend pattern: the Simple ``Structure`` and the spglib-like
primitive triple. ASU and exact-vector numerics are planned follow-ups.
"""

from .elements import SYMBOLS, atomic_number, symbol_of
from .species import Species
from .structure import Structure
from .structure_api import StructureAPI
from .structure_backend import StructureBackend
from .structure_like import StructureLike
from .structure_primitive import StructurePrimitive
from .structure_primitive_view import StructurePrimitiveView
from .structure_simple import StructureSimple
from .structure_simple_view import StructureSimpleView
from .structure_view import StructureView

StructureBackend.backend_classes = [StructureSimple, StructurePrimitive]

__all__ = [
    "Structure",
    "Species",
    "StructureLike",
    "StructureAPI",
    "StructureBackend",
    "StructureView",
    "StructureSimple",
    "StructurePrimitive",
    "StructureSimpleView",
    "StructurePrimitiveView",
    "SYMBOLS",
    "atomic_number",
    "symbol_of",
]
