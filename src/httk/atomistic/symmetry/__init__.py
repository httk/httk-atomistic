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
    "PrimitiveCellResult",
    "SettingTransform",
    "Spacegroup",
    "SubgroupTransform",
    "WyckoffBranch",
    "WyckoffPosition",
    "WyckoffSplitPiece",
    "conventional_cell",
    "maximal_subgroups",
    "minimal_supergroups",
    "operation_from_xyz",
    "operation_from_xyzt",
    "parse_linear_expression",
    "primitive_cell",
    "recognize_asu",
    "structure_tolerance",
    "subgroup_closure",
    "subgroup_transforms",
    "supergroup_closure",
    "wyckoff_letter_map",
    "wyckoff_positions",
]

if TYPE_CHECKING:
    from .primitive import PrimitiveCellResult, primitive_cell
    from .recognition import DEFAULT_TOLERANCE, recognize_asu, structure_tolerance
    from .standardization import ConventionalCellResult, conventional_cell
    from .subgroups import (
        SubgroupTransform,
        WyckoffSplitPiece,
        maximal_subgroups,
        minimal_supergroups,
        subgroup_closure,
        subgroup_transforms,
        supergroup_closure,
    )


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
    if name in {"PrimitiveCellResult", "primitive_cell"}:
        from .primitive import PrimitiveCellResult, primitive_cell

        globals().update(PrimitiveCellResult=PrimitiveCellResult, primitive_cell=primitive_cell)
        return globals()[name]
    if name in {
        "SubgroupTransform",
        "WyckoffSplitPiece",
        "maximal_subgroups",
        "minimal_supergroups",
        "subgroup_closure",
        "subgroup_transforms",
        "supergroup_closure",
    }:
        from .subgroups import (
            SubgroupTransform,
            WyckoffSplitPiece,
            maximal_subgroups,
            minimal_supergroups,
            subgroup_closure,
            subgroup_transforms,
            supergroup_closure,
        )

        globals().update(
            SubgroupTransform=SubgroupTransform,
            WyckoffSplitPiece=WyckoffSplitPiece,
            maximal_subgroups=maximal_subgroups,
            minimal_supergroups=minimal_supergroups,
            subgroup_closure=subgroup_closure,
            subgroup_transforms=subgroup_transforms,
            supergroup_closure=supergroup_closure,
        )
        return globals()[name]
    raise AttributeError(name)
