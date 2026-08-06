from typing import TYPE_CHECKING

from .definitions import load_httk_definitions
from .precision import PRECISION_PROPERTY_KEYS, precision_definitions, precision_properties

__all__ = [
    "PRECISION_PROPERTY_KEYS",
    "SETTING_PROPERTY_KEYS",
    "SYMMETRY_PROPERTY_KEYS",
    "StructureEntry",
    "StructureEntryProvider",
    "TrajectoryEntry",
    "TrajectoryEntryProvider",
    "load_httk_definitions",
    "precision_definitions",
    "precision_properties",
    "setting_definitions",
    "symmetry_properties",
]

if TYPE_CHECKING:
    from .structures import StructureEntry, StructureEntryProvider
    from .symmetry import SETTING_PROPERTY_KEYS, SYMMETRY_PROPERTY_KEYS, setting_definitions, symmetry_properties
    from .trajectories import TrajectoryEntry, TrajectoryEntryProvider


def __getattr__(name: str) -> object:
    if name in {"StructureEntry", "StructureEntryProvider"}:
        from .structures import StructureEntry, StructureEntryProvider

        globals().update(StructureEntry=StructureEntry, StructureEntryProvider=StructureEntryProvider)
        return globals()[name]
    if name in {"TrajectoryEntry", "TrajectoryEntryProvider"}:
        from .trajectories import TrajectoryEntry, TrajectoryEntryProvider

        globals().update(TrajectoryEntry=TrajectoryEntry, TrajectoryEntryProvider=TrajectoryEntryProvider)
        return globals()[name]
    if name in {"SETTING_PROPERTY_KEYS", "SYMMETRY_PROPERTY_KEYS", "setting_definitions", "symmetry_properties"}:
        from .symmetry import SETTING_PROPERTY_KEYS, SYMMETRY_PROPERTY_KEYS, setting_definitions, symmetry_properties

        globals().update(
            SETTING_PROPERTY_KEYS=SETTING_PROPERTY_KEYS,
            SYMMETRY_PROPERTY_KEYS=SYMMETRY_PROPERTY_KEYS,
            setting_definitions=setting_definitions,
            symmetry_properties=symmetry_properties,
        )
        return globals()[name]
    raise AttributeError(name)
