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
    ProtostructureRecord,
    PrototypeRecord,
    UnitcellStructureRecord,
    validate_structure_record,
    ObservableSummaryRecord,
    TrajectoryRecord,
    WyckoffOccupationRecord,
)
from httk.atomistic.models.cell.record import RecordCell
from httk.atomistic.models.sites.record import RecordSites
from httk.atomistic.models.species.record import RecordSpecies
from httk.atomistic.models.structure.record import RecordStructure

# Formula imports must follow the structure imports: formula record/like modules pull in
# the storage stack, whose integration bridges depend on a fully initialized structure family.
from httk.atomistic.models.formula.formulapattern import Formulapattern
from httk.atomistic.models.formula.formulapattern_string import FormulapatternString
from httk.atomistic.models.formula.formulapattern_view import FormulapatternView
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

# Alias for discoverability; canonical name is Formulapattern (see docs/prototypes.md).
AnonymousFormula = Formulapattern
AnonymousFormulaView = FormulapatternView

# Crystalpattern imports follow both structure and formula imports: the crystal-pattern-to-formula
# bridge and adapter deliberately depend on the completed lower-level family registrations.
from httk.atomistic.models.crystalpattern.crystalpattern import CrystalPattern
from httk.atomistic.models.crystalpattern.view import CrystalPatternView
from httk.atomistic.models.crystalpattern.anonymized import AnonymizedStructure
from httk.atomistic.models.crystalpattern.backend import CrystalPatternBackend
from httk.atomistic.models.crystalpattern.like import CrystalPatternLike, PrototypeLike
from httk.atomistic.models.crystalpattern.fundamental import ASUPattern, FundamentalDomainPattern
from httk.atomistic.models.crystalpattern.fundamental_view import FundamentalDomainPatternView
from httk.atomistic.models.formula.prototype import PrototypeComposition

# Alias for discoverability; canonical name is CrystalPattern (see docs/prototypes.md).
AnonymousStructure = CrystalPattern
AnonymousStructureView = CrystalPatternView
AnonymousStructureLike = CrystalPatternLike
# Transitional alias, removed in the taxonomy phase 4.
Prototype = FundamentalDomainPattern
PrototypeView = FundamentalDomainPatternView

# Protopattern imports follow the crystalpattern block and precede protostructure: the
# element-free family reuses the formula bridge and crystalpattern recognition above, and
# the protostructure family below extends its label notation.
from httk.atomistic.models.protopattern.backend import ProtopatternBackend
from httk.atomistic.models.protopattern.derived import DerivedProtopattern
from httk.atomistic.models.protopattern.label import ProtopatternLabel
from httk.atomistic.models.protopattern.label_string import ProtopatternLabelString
from httk.atomistic.models.protopattern.like import ProtopatternLike
from httk.atomistic.models.protopattern.occupation import ProtopatternOccupation
from httk.atomistic.models.protopattern.protopattern import Protopattern
from httk.atomistic.models.protopattern.view import ProtopatternView

# Protostructure imports follow the prototype block: the geometry-free family bridges
# through the completed formula and structure registrations above.
from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.label import ProtostructureLabel
from httk.atomistic.models.protostructure.label_string import ProtostructureLabelString
from httk.atomistic.models.protostructure.like import ProtostructureLike
from httk.atomistic.models.protostructure.occupation import WyckoffOccupation
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.protostructure.recognized import RecognizedProtostructure
from httk.atomistic.models.protostructure.view import ProtostructureView
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.recognition import DEFAULT_TOLERANCE, recognize_asu, structure_tolerance
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup, wyckoff_letter_map
from httk.atomistic.symmetry.primitive import PrimitiveCellResult, primitive_cell
from httk.atomistic.symmetry.standardization import ConventionalCellResult, conventional_cell
from httk.atomistic.symmetry.subgroups import (
    maximal_subgroups,
    minimal_supergroups,
    SubgroupRepresentationResult,
    subgroup_closure,
    subgroup_representation,
    supergroup_closure,
)
from httk.atomistic.symmetry.canonical import canonical_asu
from httk.atomistic.symmetry.lift import (
    LiftResult,
    backward_lift,
    canonicalize,
    highest_symmetry,
    lift_candidates,
    rerepresent,
)
from httk.atomistic.symmetry.paths import (
    StructurePath,
    canonicalize_full,
    interpolate_structures,
    list_representations,
    represent_like,
)
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
    PrototypeComposition,
    PlainComposition,
    FormulaString,
    FormulapatternString,
]
CrystalPatternBackend.backend_classes = [AnonymizedStructure]
ProtopatternBackend.backend_classes = [ProtopatternLabelString, DerivedProtopattern]
# The label-string probe is first: it is a cheap exact parse that either matches a
# canonical label or declines, mirroring the record-first rationale, so recognition
# sources never fall through it.
ProtostructureBackend.backend_classes = [ProtostructureLabelString, RecognizedProtostructure]
register_coercer(view_class_coercer([ChemicalFormulaView, FormulapatternView, CompositionView]), Any)
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
Protostructure.__httk_storage_record__ = ProtostructureRecord
FundamentalDomainPattern.__httk_storage_record__ = PrototypeRecord

__all__ = [
    "DEFAULT_TOLERANCE",
    "ASEAtoms",
    "ASEAtomsProtocol",
    "ASUPattern",
    "ASUStructure",
    "ASUStructureRecord",
    "ASUStructureView",
    "AffineOperation",
    "AnonymousFormula",
    "AnonymousFormulaView",
    "AnonymousStructure",
    "AnonymousStructureLike",
    "AnonymousStructureView",
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
    "CrystalPattern",
    "CrystalPatternLike",
    "CrystalPatternView",
    "DatastreamStructure",
    "Formulapattern",
    "FormulapatternView",
    "FundamentalDomainPattern",
    "FundamentalDomainPatternView",
    "FundamentalDomainStructure",
    "FundamentalDomainStructureRecord",
    "JsonlTrajectory",
    "LiftResult",
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
    "Protopattern",
    "ProtopatternLabel",
    "ProtopatternLike",
    "ProtopatternOccupation",
    "ProtopatternView",
    "Protostructure",
    "ProtostructureLabel",
    "ProtostructureLike",
    "ProtostructureRecord",
    "ProtostructureView",
    "Prototype",
    "PrototypeLike",
    "PrototypeRecord",
    "PrototypeView",
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
    "StructurePath",
    "StructureSymmetry",
    "SubgroupRepresentationResult",
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
    "WyckoffOccupation",
    "WyckoffOccupationRecord",
    "WyckoffPosition",
    "WyckoffSite",
    "asu_structure_from_cif",
    "asu_structures_from_cif",
    "atomic_number",
    "backward_lift",
    "build_supercell",
    "canonical_asu",
    "canonicalize",
    "canonicalize_full",
    "cif_setting",
    "conventional_cell",
    "cubic_supercell",
    "highest_symmetry",
    "interpolate_structures",
    "is_niggli_reduced",
    "lift_candidates",
    "list_representations",
    "maximal_subgroups",
    "minimal_supergroups",
    "niggli_reduce",
    "niggli_reduced",
    "orthogonal_supercell",
    "primitive_cell",
    "recognize_asu",
    "represent_like",
    "rerepresent",
    "same_crystal",
    "save_vesta",
    "structure_tolerance",
    "subgroup_closure",
    "subgroup_representation",
    "supergroup_closure",
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
