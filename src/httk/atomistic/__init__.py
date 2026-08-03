"""
httk-atomistic: crystal structure representations for httk v2.

Provides the UnitcellStructure domain and its component families (Cell, Sites, Species),
each following the httk-core view/backend pattern. A UnitcellStructure holds a ``cell``, a
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
from .asu_structure import ASUStructure, FundamentalDomainStructure, WyckoffSite
from .asu_structure_view import ASUStructureView
from .cell import Cell
from .cell_backend import CellBackend
from .cell_like import CellLike
from .cell_params import CellParams
from .cell_params_view import CellParamsView
from .cell_view import CellView
from .cif_structures import asu_structure_from_cif, asu_structures_from_cif, cif_setting
from .compat import ASEAtoms
from .composition import (
    Assembly,
)
from .datastream_structure import DatastreamStructure
from .elements import atomic_number, symbol_of
from .numeric_unitcell_structure import NumericUnitcellStructure
from .numeric_unitcell_structure_view import NumericUnitcellStructureView
from .optimade_structure import OptimadeStructure
from .plain_cell import PlainCell
from .plain_sites import PlainSites
from .plain_species import PlainSpecies
from .plain_species_view import PlainSpeciesView
from .plain_structure import PlainStructure
from .plain_structure_view import PlainStructureView
from .record_cell import RecordCell
from .record_sites import RecordSites
from .record_species import RecordSpecies
from .record_structure import RecordStructure
from .setting_transform import SettingTransform
from .sites import Sites
from .sites_backend import SitesBackend
from .sites_like import SitesLike
from .sites_view import SitesView
from .spacegroup import Spacegroup, wyckoff_letter_map
from .species import Species
from .species_backend import SpeciesBackend
from .species_like import SpeciesLike
from .species_view import SpeciesView
from .standardization import ConventionalCellResult, conventional_cell
from .structure_backend import StructureBackend
from .structure_comparison import same_crystal
from .structure_entries import StructureEntry, StructureEntryProvider
from .structure_like import StructureLike
from .structure_record import (
    ASUStructureRecord,
    FundamentalDomainStructureRecord,
    UnitcellStructureRecord,
    validate_structure_record,
)
from .supercell import (
    SupercellResult,
    build_supercell,
    cubic_supercell,
    orthogonal_supercell,
)
from .unitcell_structure import UnitcellStructure
from .unitcell_structure_view import UnitcellStructureView
from .wyckoff import WyckoffPosition, wyckoff_positions

StructureBackend.backend_classes = [
    OptimadeStructure,
    PlainStructure,
    NumericUnitcellStructure,
    ASEAtoms,
    RecordStructure,
    DatastreamStructure,
]
CellBackend.backend_classes = [PlainCell, CellParams, RecordCell]
SitesBackend.backend_classes = [PlainSites, RecordSites]
SpeciesBackend.backend_classes = [PlainSpecies, RecordSpecies]

# Storage opt-in is exact-source-class scoped: core intentionally resolves this
# attribute through vars(type(source)), never by inheritance.
UnitcellStructure.__httk_storage_record__ = UnitcellStructureRecord
UnitcellStructureView.__httk_storage_record__ = UnitcellStructureRecord
FundamentalDomainStructure.__httk_storage_record__ = FundamentalDomainStructureRecord
ASUStructure.__httk_storage_record__ = ASUStructureRecord
ASUStructureView.__httk_storage_record__ = ASUStructureRecord

__all__ = [
    "DEFAULT_TOLERANCE",
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
    "DatastreamStructure",
    "FundamentalDomainStructure",
    "FundamentalDomainStructureRecord",
    "NumericUnitcellStructureView",
    "OptimadeStructure",
    "PlainSpeciesView",
    "PlainStructureView",
    "SettingTransform",
    "Sites",
    "SitesLike",
    "SitesView",
    "Spacegroup",
    "Species",
    "SpeciesLike",
    "SpeciesView",
    "StructureEntry",
    "StructureEntryProvider",
    "StructureLike",
    "SupercellResult",
    "UnitcellStructure",
    "UnitcellStructureRecord",
    "UnitcellStructureView",
    "WyckoffPosition",
    "WyckoffSite",
    "asu_structure_from_cif",
    "asu_structures_from_cif",
    "atomic_number",
    "build_supercell",
    "cif_setting",
    "conventional_cell",
    "cubic_supercell",
    "orthogonal_supercell",
    "recognize_asu",
    "same_crystal",
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
