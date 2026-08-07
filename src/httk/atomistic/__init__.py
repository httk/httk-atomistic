"""Provide crystal structure representations for httk v2.

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

# ruff: noqa: I001

from typing import Any

from httk.core.views import register_coercer, view_class_coercer

from httk.atomistic.models.cell.backend import CellBackend
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.models.cell.params_view import CellParamsView
from httk.atomistic.models.cell.plain import PlainCell
from httk.atomistic.models.cell.view import CellView
from httk.atomistic.models.sites.backend import SitesBackend
from httk.atomistic.models.sites.like import SitesLike
from httk.atomistic.models.sites.plain import PlainSites
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.sites.view import SitesView
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.cartesian_view import CartesianSiteMomentsView
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.models.moments.crystalaxis import CrystalAxisSiteMoments
from httk.atomistic.models.moments.crystalaxis_view import CrystalAxisSiteMomentsView
from httk.atomistic.models.moments.like import SiteMomentsLike
from httk.atomistic.models.species.backend import SpeciesBackend
from httk.atomistic.models.species.like import SpeciesLike
from httk.atomistic.models.species.plain import PlainSpecies
from httk.atomistic.models.species.plain_view import PlainSpeciesView
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.species.view import SpeciesView
from httk.atomistic.integrations import (
    ASEAtoms,
    ASEAtomsProtocol,
    PymatgenStructure,
    PymatgenStructureProtocol,
    VASPStructure,
    VASPTrajectory,
)
from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure, WyckoffSite
from httk.atomistic.models.structure.asu_view import ASUStructureView
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.comparison import same_crystal
from httk.atomistic.models.structure.datastream import DatastreamStructure
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.modulated import ModulatedStructure
from httk.atomistic.models.structure.numeric import NumericUnitcellStructure
from httk.atomistic.models.structure.numeric_view import NumericUnitcellStructureView
from httk.atomistic.models.structure.optimade import OptimadeStructure
from httk.atomistic.models.structure.plain import PlainStructure
from httk.atomistic.models.structure.plain_view import PlainStructureView
from httk.atomistic.models.structure.semantics import StructureSymmetry
from httk.atomistic.models.structure.symops import SymopsStructure
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.models.trajectory import (
    JsonlTrajectory,
    PlainTrajectory,
    RecordTrajectory,
    Trajectory,
    TrajectoryLike,
    TrajectoryView,
)
from httk.atomistic.wavefunction import PlaneWaveFunctions, save_vesta, wavefunction_overlap
from httk.atomistic.entries.structures import StructureEntry, StructureEntryProvider
from httk.atomistic.entries.trajectories import TrajectoryEntry, TrajectoryEntryProvider
from httk.atomistic.storage.records import (
    ASUStructureRecord,
    FundamentalDomainStructureRecord,
    UnitcellStructureRecord,
    validate_structure_record,
    ObservableSummaryRecord,
    TrajectoryRecord,
)
from httk.atomistic.models.cell.record import RecordCell
from httk.atomistic.models.sites.record import RecordSites
from httk.atomistic.models.species.record import RecordSpecies
from httk.atomistic.models.structure.record import RecordStructure

# Formula imports must follow the structure imports: formula record/like modules pull in
# the storage stack, whose integration bridges depend on a fully initialized structure family.
from httk.atomistic.models.formula.anonymous import AnonymousFormula
from httk.atomistic.models.formula.anonymous_string import AnonymousFormulaString
from httk.atomistic.models.formula.anonymous_view import AnonymousFormulaView
from httk.atomistic.models.formula.backend import ChemicalFormulaBackend
from httk.atomistic.models.formula.composition import Composition
from httk.atomistic.models.formula.composition_view import CompositionView
from httk.atomistic.models.formula.formula import ChemicalFormula
from httk.atomistic.models.formula.formula_string import FormulaString
from httk.atomistic.models.formula.formula_view import ChemicalFormulaView
from httk.atomistic.models.formula.like import ChemicalFormulaLike
from httk.atomistic.models.formula.plain import PlainComposition
from httk.atomistic.models.formula.record import RecordComposition
from httk.atomistic.models.formula.structure import StructureComposition
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.recognition import DEFAULT_TOLERANCE, recognize_asu, structure_tolerance
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup, wyckoff_letter_map
from httk.atomistic.symmetry.primitive import PrimitiveCellResult, primitive_cell
from httk.atomistic.symmetry.standardization import ConventionalCellResult, conventional_cell
from httk.atomistic.symmetry.wyckoff import WyckoffPosition, wyckoff_positions

from .cif_structures import asu_structure_from_cif, asu_structures_from_cif, cif_setting
from .composition import Assembly, ChemicalComposition
from .elements import atomic_number, symbol_of
from .supercell import (
    SupercellResult,
    build_supercell,
    cubic_supercell,
    orthogonal_supercell,
)
from .reduction import (
    NiggliReducedStructureResult,
    NiggliReductionResult,
    is_niggli_reduced,
    niggli_reduce,
    niggli_reduced,
)

# Record backends first: they match their exact record classes by isinstance and
# reject everything else instantly, so record inputs never fall through the
# parse-and-raise probes of the raw-input backends (and raw inputs lose nothing).
ChemicalFormulaBackend.backend_classes = [
    RecordComposition,
    StructureComposition,
    PlainComposition,
    FormulaString,
    AnonymousFormulaString,
]
register_coercer(view_class_coercer([ChemicalFormulaView, AnonymousFormulaView, CompositionView]), Any)
StructureBackend.backend_classes = [
    RecordStructure,
    OptimadeStructure,
    PlainStructure,
    NumericUnitcellStructure,
    ASEAtoms,
    PymatgenStructure,
    DatastreamStructure,
]
CellBackend.backend_classes = [RecordCell, PlainCell, CellParams]
SitesBackend.backend_classes = [RecordSites, PlainSites]
SiteMomentsBackend.backend_classes = []
SpeciesBackend.backend_classes = [RecordSpecies, PlainSpecies]
from httk.atomistic.models.trajectory.backend import TrajectoryBackend

TrajectoryBackend.backend_classes = [RecordTrajectory, PlainTrajectory]

# Storage opt-in is exact-source-class scoped: core intentionally resolves this
# attribute through vars(type(source)), never by inheritance.
UnitcellStructure.__httk_storage_record__ = UnitcellStructureRecord
UnitcellStructureView.__httk_storage_record__ = UnitcellStructureRecord
FundamentalDomainStructure.__httk_storage_record__ = FundamentalDomainStructureRecord
ASUStructure.__httk_storage_record__ = ASUStructureRecord
ASUStructureView.__httk_storage_record__ = ASUStructureRecord
Trajectory.__httk_storage_record__ = TrajectoryRecord
TrajectoryView.__httk_storage_record__ = TrajectoryRecord

__all__ = [
    "DEFAULT_TOLERANCE",
    "ASEAtoms",
    "ASEAtomsProtocol",
    "ASUStructure",
    "ASUStructureRecord",
    "ASUStructureView",
    "AffineOperation",
    "AnonymousFormula",
    "AnonymousFormulaView",
    "Assembly",
    "CartesianSiteMoments",
    "CartesianSiteMomentsView",
    "Cell",
    "CellLike",
    "CellParams",
    "CellParamsView",
    "CellView",
    "ChemicalComposition",
    "ChemicalFormula",
    "ChemicalFormulaLike",
    "ChemicalFormulaView",
    "CollinearSiteMoments",
    "Composition",
    "CompositionView",
    "ConventionalCellResult",
    "CrystalAxisSiteMoments",
    "CrystalAxisSiteMomentsView",
    "DatastreamStructure",
    "FundamentalDomainStructure",
    "FundamentalDomainStructureRecord",
    "JsonlTrajectory",
    "ModulatedStructure",
    "NiggliReducedStructureResult",
    "NiggliReductionResult",
    "NumericUnitcellStructureView",
    "ObservableSummaryRecord",
    "OptimadeStructure",
    "PlainSpeciesView",
    "PlainStructureView",
    "PlainTrajectory",
    "PlaneWaveFunctions",
    "PrimitiveCellResult",
    "PymatgenStructure",
    "PymatgenStructureProtocol",
    "RecordTrajectory",
    "SettingTransform",
    "SiteMomentsLike",
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
    "StructureSymmetry",
    "SupercellResult",
    "SymopsStructure",
    "Trajectory",
    "TrajectoryEntry",
    "TrajectoryEntryProvider",
    "TrajectoryLike",
    "TrajectoryRecord",
    "TrajectoryView",
    "UnitcellStructure",
    "UnitcellStructureRecord",
    "UnitcellStructureView",
    "VASPStructure",
    "VASPTrajectory",
    "WyckoffPosition",
    "WyckoffSite",
    "asu_structure_from_cif",
    "asu_structures_from_cif",
    "atomic_number",
    "build_supercell",
    "cif_setting",
    "conventional_cell",
    "cubic_supercell",
    "is_niggli_reduced",
    "niggli_reduce",
    "niggli_reduced",
    "orthogonal_supercell",
    "primitive_cell",
    "recognize_asu",
    "same_crystal",
    "save_vesta",
    "structure_tolerance",
    "symbol_of",
    "validate_structure_record",
    "wavefunction_overlap",
    "wyckoff_letter_map",
    "wyckoff_positions",
]

# ASE is optional. The view module subclasses ase.Atoms at class-definition time, so it
# cannot be imported without ASE; guard it exactly like the optional numpy vector view.
try:
    from httk.atomistic.integrations import ASEAtomsView  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("ASEAtomsView")

try:
    from httk.atomistic.integrations import PymatgenStructureView  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("PymatgenStructureView")
