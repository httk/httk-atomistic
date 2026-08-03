from typing import TYPE_CHECKING

from .api import StructureAPI
from .ase import ASEAtoms, ASEAtomsProtocol
from .asu import ASUStructure, FundamentalDomainStructure, WyckoffSite
from .backend import StructureBackend
from .comparison import same_crystal
from .datastream import DatastreamStructure
from .like import StructureLike
from .numeric import NumericUnitcellStructure
from .numeric_view import NumericUnitcellStructureView
from .optimade import OptimadeStructure
from .plain import PlainStructure
from .plain_view import PlainStructureView
from .record import RecordStructure
from .semantics import (
    OptimizationType,
    StructureSemanticsMixin,
    StructureSymmetry,
    initialize_semantics,
    validate_descriptive_formula,
    validate_hill_formula,
    validate_optimization_type,
)
from .unitcell import UnitcellStructure
from .unitcell_view import UnitcellStructureView
from .view import StructureView

__all__ = [
    "ASEAtoms",
    "ASEAtomsProtocol",
    "ASUStructure",
    "ASUStructureView",
    "DatastreamStructure",
    "FundamentalDomainStructure",
    "NumericUnitcellStructure",
    "NumericUnitcellStructureView",
    "OptimadeStructure",
    "OptimizationType",
    "PlainStructure",
    "PlainStructureView",
    "RecordStructure",
    "StructureAPI",
    "StructureBackend",
    "StructureLike",
    "StructureSemanticsMixin",
    "StructureSymmetry",
    "StructureView",
    "UnitcellStructure",
    "UnitcellStructureView",
    "WyckoffSite",
    "initialize_semantics",
    "same_crystal",
    "validate_descriptive_formula",
    "validate_hill_formula",
    "validate_optimization_type",
]

if TYPE_CHECKING:
    from .asu_view import ASUStructureView

try:
    from .ase_view import ASEAtomsView  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("ASEAtomsView")


def __getattr__(name: str) -> object:
    if name == "ASUStructureView":
        from .asu_view import ASUStructureView

        globals()[name] = ASUStructureView
        return ASUStructureView
    raise AttributeError(name)
