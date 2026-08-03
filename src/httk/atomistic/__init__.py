"""
httk-atomistic: crystal structure representations for httk v2.

Provides the Structure domain and its component families (Cell, Sites, Species),
each following the httk-core view/backend pattern. A Structure holds a ``cell``, a
``sites``, a tuple of ``species``, and a ``species_at_sites``; each component has a
class representation and a primitive representation convertible through views.

Crystal symmetry is modelled exactly over the rationals: :class:`Spacegroup` carries a
space-group *setting* with its symmetry operations and Wyckoff table, and
:class:`SettingTransform` relates any setting to the International Tables standard one,
so a structure in an arbitrary non-standard setting can be represented without loss.
The underlying tables ship in :mod:`httk.atomistic.data`.
"""

from .affine_operation import AffineOperation
from .asu_recognition import DEFAULT_TOLERANCE, recognize_asu, structure_tolerance
from .asu_structure import ASUSite, ASUStructure, FundamentalDomainStructure
from .asu_structure_view import ASUStructureView
from .cell import Cell
from .cell_backend import CellBackend
from .cell_like import CellLike
from .cell_params import CellParams
from .cell_params_view import CellParamsView
from .cell_primitive import CellPrimitive
from .cell_record_backend import CellRecordBackend
from .cell_view import CellView
from .cif_structures import asu_structure_from_cif, asu_structures_from_cif, cif_setting
from .compat import ASEAtomsBackend
from .composition import (
    Assembly,
)
from .elements import atomic_number, symbol_of
from .numeric_unitcell_structure_backend import NumericUnitcellStructureBackend
from .numeric_unitcell_structure_view import NumericUnitcellStructureView
from .optimade_structure import OptimadeStructure
from .setting_transform import SettingTransform
from .sites import Sites
from .sites_backend import SitesBackend
from .sites_like import SitesLike
from .sites_primitive import SitesPrimitive
from .sites_record_backend import SitesRecordBackend
from .sites_view import SitesView
from .spacegroup import Spacegroup, wyckoff_letter_map
from .species import Species
from .species_backend import SpeciesBackend
from .species_like import SpeciesLike
from .species_primitive import SpeciesPrimitive
from .species_primitive_view import SpeciesPrimitiveView
from .species_record_backend import SpeciesRecordBackend
from .species_view import SpeciesView
from .standardization import ConventionalCellResult, conventional_cell
from .structure import Structure
from .structure_backend import StructureBackend
from .structure_comparison import same_crystal
from .structure_entries import StructureEntry, StructureEntryProvider
from .structure_like import StructureLike
from .structure_primitive import StructurePrimitive
from .structure_primitive_view import StructurePrimitiveView
from .structure_record import (
    ASUStructureRecord,
    FundamentalDomainStructureRecord,
    UnitcellStructureRecord,
    validate_structure_record,
)
from .structure_record_backend import StructureRecordBackend
from .supercell import (
    SupercellResult,
    build_supercell,
    cubic_supercell,
    orthogonal_supercell,
)
from .unitcell_structure_view import UnitcellStructureView
from .vasp_structures import (
    load_asu_structure,
    load_structure,
    structure_from_payload,
    structure_from_poscar,
)
from .wyckoff import WyckoffPosition, wyckoff_positions

StructureBackend.backend_classes = [
    OptimadeStructure,
    StructurePrimitive,
    NumericUnitcellStructureBackend,
    ASEAtomsBackend,
    StructureRecordBackend,
]
CellBackend.backend_classes = [CellPrimitive, CellParams, CellRecordBackend]
SitesBackend.backend_classes = [SitesPrimitive, SitesRecordBackend]
SpeciesBackend.backend_classes = [SpeciesPrimitive, SpeciesRecordBackend]

# Storage opt-in is exact-source-class scoped: core intentionally resolves this
# attribute through vars(type(source)), never by inheritance.
Structure.__httk_storage_record__ = UnitcellStructureRecord
UnitcellStructureView.__httk_storage_record__ = UnitcellStructureRecord
FundamentalDomainStructure.__httk_storage_record__ = FundamentalDomainStructureRecord
ASUStructure.__httk_storage_record__ = ASUStructureRecord
ASUStructureView.__httk_storage_record__ = ASUStructureRecord

__all__ = [
    "DEFAULT_TOLERANCE",
    "ASUSite",
    "ASUStructure",
    "ASUStructureRecord",
    "ASUStructureView",
    "AffineOperation",
    "Assembly",
    "Cell",
    "CellLike",
    "CellParams",
    "CellParamsView",
    "CellView",
    "ConventionalCellResult",
    "FundamentalDomainStructure",
    "FundamentalDomainStructureRecord",
    "NumericUnitcellStructureView",
    "OptimadeStructure",
    "SettingTransform",
    "Sites",
    "SitesLike",
    "SitesView",
    "Spacegroup",
    "Species",
    "SpeciesLike",
    "SpeciesPrimitiveView",
    "SpeciesView",
    "Structure",
    "StructureEntry",
    "StructureEntryProvider",
    "StructureLike",
    "StructurePrimitiveView",
    "SupercellResult",
    "UnitcellStructureRecord",
    "UnitcellStructureView",
    "WyckoffPosition",
    "asu_structure_from_cif",
    "asu_structures_from_cif",
    "atomic_number",
    "build_supercell",
    "cif_setting",
    "conventional_cell",
    "cubic_supercell",
    "load_asu_structure",
    "load_structure",
    "orthogonal_supercell",
    "recognize_asu",
    "same_crystal",
    "structure_from_payload",
    "structure_from_poscar",
    "structure_tolerance",
    "symbol_of",
    "validate_structure_record",
    "wyckoff_letter_map",
    "wyckoff_positions",
]

# ASE is optional. The view module subclasses ase.Atoms at class-definition time, so it
# cannot be imported without ASE; guard it exactly like the optional numpy vector view.
try:
    from .ase_atoms_view import ASEAtomsView  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("ASEAtomsView")
