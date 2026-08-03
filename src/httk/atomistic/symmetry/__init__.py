from typing import TYPE_CHECKING

from .affine_operation import AffineOperation
from .setting_transform import SettingTransform
from .spacegroup import Spacegroup, wyckoff_letter_map
from .wyckoff import WyckoffBranch, WyckoffPosition, wyckoff_positions
from .xyz import operation_from_xyz, operation_from_xyzt, parse_linear_expression

__all__ = [
    "DEFAULT_TOLERANCE",
    "AffineOperation",
    "ConventionalCellResult",
    "SettingTransform",
    "Spacegroup",
    "WyckoffBranch",
    "WyckoffPosition",
    "conventional_cell",
    "operation_from_xyz",
    "operation_from_xyzt",
    "parse_linear_expression",
    "recognize_asu",
    "structure_tolerance",
    "wyckoff_letter_map",
    "wyckoff_positions",
]

if TYPE_CHECKING:
    from .recognition import DEFAULT_TOLERANCE, recognize_asu, structure_tolerance
    from .standardization import ConventionalCellResult, conventional_cell


def __getattr__(name: str) -> object:
    if name in {"DEFAULT_TOLERANCE", "recognize_asu", "structure_tolerance"}:
        from .recognition import DEFAULT_TOLERANCE, recognize_asu, structure_tolerance

        globals().update(
            DEFAULT_TOLERANCE=DEFAULT_TOLERANCE,
            recognize_asu=recognize_asu,
            structure_tolerance=structure_tolerance,
        )
        return globals()[name]
    if name in {"ConventionalCellResult", "conventional_cell"}:
        from .standardization import ConventionalCellResult, conventional_cell

        globals().update(ConventionalCellResult=ConventionalCellResult, conventional_cell=conventional_cell)
        return globals()[name]
    raise AttributeError(name)
