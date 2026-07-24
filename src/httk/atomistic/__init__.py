"""
httk-atomistic: crystal structure representations for httk v2.

Provides the Structure domain and its component families (Cell, Sites, Species),
each following the httk-core view/backend pattern. A Structure holds a ``cell``, a
``sites``, a tuple of ``species``, and a ``species_at_sites``; each component has a
class representation and a primitive representation convertible through views. ASU and
exact-vector numerics are planned follow-ups.
"""

from .cell import Cell
from .cell_api import CellAPI
from .cell_backend import CellBackend
from .cell_class import CellClass
from .cell_class_view import CellClassView
from .cell_like import CellLike
from .cell_numeric_view import CellNumericView
from .cell_params import CellParams
from .cell_params_view import CellParamsView
from .cell_primitive import CellPrimitive
from .cell_primitive_view import CellPrimitiveView
from .cell_view import CellView
from .elements import SYMBOLS, atomic_number, symbol_of
from .numeric_cell import NumericCell
from .numeric_sites import NumericSites
from .numeric_structure import NumericStructure
from .sites import Sites
from .sites_api import SitesAPI
from .sites_backend import SitesBackend
from .sites_class import SitesClass
from .sites_class_view import SitesClassView
from .sites_like import SitesLike
from .sites_numeric_view import SitesNumericView
from .sites_primitive import SitesPrimitive
from .sites_primitive_view import SitesPrimitiveView
from .sites_view import SitesView
from .species import Species
from .species_api import SpeciesAPI
from .species_backend import SpeciesBackend
from .species_class import SpeciesClass
from .species_class_view import SpeciesClassView
from .species_like import SpeciesLike
from .species_primitive import SpeciesPrimitive
from .species_primitive_view import SpeciesPrimitiveView
from .species_view import SpeciesView
from .structure import Structure
from .structure_api import StructureAPI
from .structure_backend import StructureBackend
from .structure_entries import StructureEntryProvider
from .structure_like import StructureLike
from .structure_numeric_view import StructureNumericView
from .structure_primitive import StructurePrimitive
from .structure_primitive_view import StructurePrimitiveView
from .structure_simple import StructureSimple
from .structure_simple_view import StructureSimpleView
from .structure_view import StructureView
from .vasp_structures import load_structure, structure_from_poscar

StructureBackend.backend_classes = [StructureSimple, StructurePrimitive]
CellBackend.backend_classes = [CellClass, CellPrimitive, CellParams]
SitesBackend.backend_classes = [SitesClass, SitesPrimitive]
SpeciesBackend.backend_classes = [SpeciesClass, SpeciesPrimitive]

__all__ = [
    "Structure",
    "StructureLike",
    "StructureAPI",
    "StructureEntryProvider",
    "StructureBackend",
    "StructureView",
    "StructureSimple",
    "StructurePrimitive",
    "StructureSimpleView",
    "StructurePrimitiveView",
    "StructureNumericView",
    "NumericStructure",
    "NumericCell",
    "NumericSites",
    "CellNumericView",
    "SitesNumericView",
    "Cell",
    "CellLike",
    "CellAPI",
    "CellBackend",
    "CellView",
    "CellClass",
    "CellPrimitive",
    "CellParams",
    "CellClassView",
    "CellPrimitiveView",
    "CellParamsView",
    "Sites",
    "SitesLike",
    "SitesAPI",
    "SitesBackend",
    "SitesView",
    "SitesClass",
    "SitesPrimitive",
    "SitesClassView",
    "SitesPrimitiveView",
    "Species",
    "SpeciesLike",
    "SpeciesAPI",
    "SpeciesBackend",
    "SpeciesView",
    "SpeciesClass",
    "SpeciesPrimitive",
    "SpeciesClassView",
    "SpeciesPrimitiveView",
    "SYMBOLS",
    "atomic_number",
    "symbol_of",
    "structure_from_poscar",
    "load_structure",
]
