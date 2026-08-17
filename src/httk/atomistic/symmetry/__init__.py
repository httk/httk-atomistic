from typing import TYPE_CHECKING

from .affine_operation import AffineOperation
from .setting_transform import SettingTransform
from .spacegroup import Spacegroup, wyckoff_letter_map
from .wyckoff import WyckoffBranch, WyckoffPosition, wyckoff_positions
from .xyz import operation_from_xyz, operation_from_xyzt, parse_linear_expression

__all__ = [
    "DEFAULT_TOLERANCE",
    "AffineOperation",
    "CommonSubgroupResult",
    "ConventionalCellResult",
    "LiftResult",
    "PrimitiveCellResult",
    "SettingTransform",
    "Spacegroup",
    "StructurePath",
    "SubgroupRepresentationResult",
    "SubgroupTransform",
    "WyckoffBranch",
    "WyckoffPosition",
    "WyckoffSplitPiece",
    "backward_lift",
    "canonical_asu",
    "canonicalize",
    "common_subgroup_representation",
    "conventional_cell",
    "highest_symmetry",
    "interpolate_structures",
    "lift_candidates",
    "maximal_subgroups",
    "minimal_supergroups",
    "operation_from_xyz",
    "operation_from_xyzt",
    "parse_linear_expression",
    "primitive_cell",
    "recognize_asu",
    "represent_like",
    "rerepresent",
    "structure_tolerance",
    "subgroup_closure",
    "subgroup_transforms",
    "supergroup_closure",
    "wyckoff_letter_map",
    "wyckoff_positions",
]

if TYPE_CHECKING:
    from .canonical import canonical_asu
    from .lift import LiftResult, backward_lift, canonicalize, highest_symmetry, lift_candidates, rerepresent
    from .paths import (
        CommonSubgroupResult,
        StructurePath,
        common_subgroup_representation,
        interpolate_structures,
        represent_like,
    )
    from .primitive import PrimitiveCellResult, primitive_cell
    from .recognition import DEFAULT_TOLERANCE, recognize_asu, structure_tolerance
    from .standardization import ConventionalCellResult, conventional_cell
    from .subgroups import (
        SubgroupRepresentationResult,
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
    if name in {
        "CommonSubgroupResult",
        "StructurePath",
        "common_subgroup_representation",
        "interpolate_structures",
        "represent_like",
    }:
        from .paths import (
            CommonSubgroupResult,
            StructurePath,
            common_subgroup_representation,
            interpolate_structures,
            represent_like,
        )

        globals().update(
            CommonSubgroupResult=CommonSubgroupResult,
            StructurePath=StructurePath,
            common_subgroup_representation=common_subgroup_representation,
            interpolate_structures=interpolate_structures,
            represent_like=represent_like,
        )
        return globals()[name]
    if name in {"PrimitiveCellResult", "primitive_cell"}:
        from .primitive import PrimitiveCellResult, primitive_cell

        globals().update(PrimitiveCellResult=PrimitiveCellResult, primitive_cell=primitive_cell)
        return globals()[name]
    if name in {
        "SubgroupRepresentationResult",
        "SubgroupTransform",
        "WyckoffSplitPiece",
        "maximal_subgroups",
        "minimal_supergroups",
        "subgroup_closure",
        "subgroup_transforms",
        "supergroup_closure",
    }:
        from .subgroups import (
            SubgroupRepresentationResult,
            SubgroupTransform,
            WyckoffSplitPiece,
            maximal_subgroups,
            minimal_supergroups,
            subgroup_closure,
            subgroup_transforms,
            supergroup_closure,
        )

        globals().update(
            SubgroupRepresentationResult=SubgroupRepresentationResult,
            SubgroupTransform=SubgroupTransform,
            WyckoffSplitPiece=WyckoffSplitPiece,
            maximal_subgroups=maximal_subgroups,
            minimal_supergroups=minimal_supergroups,
            subgroup_closure=subgroup_closure,
            subgroup_transforms=subgroup_transforms,
            supergroup_closure=supergroup_closure,
        )
        return globals()[name]
    if name in {"LiftResult", "backward_lift", "canonicalize", "highest_symmetry", "lift_candidates", "rerepresent"}:
        from .lift import LiftResult, backward_lift, canonicalize, highest_symmetry, lift_candidates, rerepresent

        globals().update(
            LiftResult=LiftResult,
            backward_lift=backward_lift,
            canonicalize=canonicalize,
            highest_symmetry=highest_symmetry,
            lift_candidates=lift_candidates,
            rerepresent=rerepresent,
        )
        return globals()[name]
    if name == "canonical_asu":
        from .canonical import canonical_asu

        globals().update(canonical_asu=canonical_asu)
        return globals()[name]
    raise AttributeError(name)
