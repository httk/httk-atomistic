"""Graph queries and exact transformations for Bärnighausen subgroup trees.

The vendored tables contain maximal subgroup relations and their standard-setting
transformations for all 230 space-group types. Only those one-hop relations are
tabulated; closures and the inverted supergroup graph are derived here.

The split-affine convention pinned by ``tests/test_subgroups.py`` is that a
:class:`WyckoffSplitPiece` operation maps a parent standard-setting coordinate directly
to a child standard-setting coordinate. A :class:`SubgroupTransform` operation is the
tabulated affine basis change between the parent and child standard settings.
"""

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from types import MappingProxyType
from typing import Any

from httk.atomistic import data
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.spacegroup import Spacegroup

__all__ = [
    "SubgroupTransform",
    "WyckoffSplitPiece",
    "maximal_subgroups",
    "minimal_supergroups",
    "subgroup_closure",
    "subgroup_transforms",
    "supergroup_closure",
]


@dataclass(frozen=True, slots=True)
class WyckoffSplitPiece:
    """One child Wyckoff piece in a maximal subgroup transformation.

    :param letter: The child-setting Wyckoff letter as tabulated.
    :param xyz: The tabulated coordinate expression, for display and provenance.
    :param operation: The exact affine map from a parent standard-setting coordinate
        directly to a child standard-setting coordinate.
    """

    letter: str
    xyz: str
    operation: AffineOperation


@dataclass(frozen=True, slots=True)
class SubgroupTransform:
    """One exact standard-setting transformation in a maximal subgroup table entry.

    :param parent: The parent space group in its IT standard setting.
    :param subgroup: The subgroup in its IT standard setting.
    :param index: The tabulated subgroup index ``[G:H]``.
    :param subgroup_type: The tabulated relation type, ``"t"`` or ``"k"``.
    :param k_subtype: The tabulated ``k`` subtype, or ``None`` for a ``t`` relation.
    :param operation: The exact tabulated affine basis change mapping parent
        standard-setting coordinates into subgroup standard-setting coordinates.
    :param splittings: An immutable mapping from parent Wyckoff letters to child pieces.
    """

    parent: Spacegroup
    subgroup: Spacegroup
    index: int
    subgroup_type: str
    k_subtype: str | None
    operation: AffineOperation
    splittings: Mapping[str, tuple[WyckoffSplitPiece, ...]]

    def __post_init__(self) -> None:
        """Freeze the supplied splitting mapping after dataclass construction."""
        object.__setattr__(self, "splittings", MappingProxyType(dict(self.splittings)))

    def __hash__(self) -> int:
        """Hash the immutable transform, including its table-ordered splittings."""
        return hash(
            (
                self.parent,
                self.subgroup,
                self.index,
                self.subgroup_type,
                self.k_subtype,
                self.operation,
                tuple(self.splittings.items()),
            )
        )


def _it_number(spacegroup: Spacegroup | int) -> int:
    """Normalize a space-group object or number and validate its IT record."""
    it_number = spacegroup.it_number if isinstance(spacegroup, Spacegroup) else spacegroup
    if not isinstance(it_number, int):
        raise TypeError(f"expected Spacegroup or int, got {type(spacegroup).__name__}")
    data.spacegroup_subgroup_record(it_number)
    return it_number


def _affine_from_3x4(record: list[list[str]]) -> AffineOperation:
    """Build an exact affine operation from three matrix-plus-vector rows."""
    matrix = [[Fraction(value) for value in row[:3]] for row in record]
    vector = [Fraction(row[3]) for row in record]
    return AffineOperation(matrix, vector)


def _affine_from_record(record: dict[str, Any]) -> AffineOperation:
    """Build an exact affine operation from a vendored matrix/vector record."""
    affine = record["affine_transformation"]
    matrix = [[Fraction(value) for value in row] for row in affine["matrix"]]
    vector = [Fraction(value) for value in affine["vector"]]
    return AffineOperation(matrix, vector)


def maximal_subgroups(spacegroup: Spacegroup | int) -> tuple[int, ...]:
    """Return the distinct tabulated maximal-subgroup IT numbers.

    Self-referencing isomorphic entries are excluded from graph navigation.

    :param spacegroup: A space group or IT number identifying the parent.
    :return: Sorted unique target IT numbers, excluding ``spacegroup`` itself.
    :raises KeyError: If the IT number has no vendored subgroup record.
    """
    it_number = _it_number(spacegroup)
    return tuple(
        sorted(
            {
                entry["target_it_number"]
                for entry in data.spacegroup_subgroup_record(it_number)["baernighausen"]
                if entry["target_it_number"] != it_number
            }
        )
    )


@cache
def _supergroup_graph() -> dict[int, tuple[int, ...]]:
    """Build the inverted non-self maximal-subgroup graph once."""
    parents: dict[int, set[int]] = {it_number: set() for it_number in range(1, 231)}
    for parent in range(1, 231):
        for entry in data.spacegroup_subgroup_record(parent)["baernighausen"]:
            child = entry["target_it_number"]
            if child != parent:
                parents[child].add(parent)
    return {child: tuple(sorted(group)) for child, group in parents.items()}


def minimal_supergroups(spacegroup: Spacegroup | int) -> tuple[int, ...]:
    """Return the distinct tabulated minimal-supergroup IT numbers.

    Self-referencing isomorphic entries are excluded from graph navigation.

    :param spacegroup: A space group or IT number identifying the subgroup.
    :return: Sorted unique parent IT numbers, excluding ``spacegroup`` itself.
    :raises KeyError: If the IT number has no vendored subgroup record.
    """
    it_number = _it_number(spacegroup)
    return _supergroup_graph()[it_number]


def _closure(
    spacegroup: Spacegroup | int,
    neighbours: Callable[[int], tuple[int, ...]],
    include_self: bool,
) -> tuple[int, ...]:
    """Return a sorted breadth-first transitive closure."""
    root = _it_number(spacegroup)
    visited = {root}
    pending = deque([root])
    while pending:
        current = pending.popleft()
        for neighbour in neighbours(current):
            if neighbour not in visited:
                visited.add(neighbour)
                pending.append(neighbour)
    if not include_self:
        visited.remove(root)
    return tuple(sorted(visited))


def subgroup_closure(spacegroup: Spacegroup | int, *, include_self: bool = False) -> tuple[int, ...]:
    """Return the graph-derived transitive subgroup closure.

    :param spacegroup: A space group or IT number identifying the parent.
    :param include_self: Include the root IT number in the result.
    :return: Sorted reachable subgroup IT numbers.
    :raises KeyError: If the IT number has no vendored subgroup record.
    """
    return _closure(spacegroup, maximal_subgroups, include_self)


def supergroup_closure(spacegroup: Spacegroup | int, *, include_self: bool = False) -> tuple[int, ...]:
    """Return the graph-derived transitive supergroup closure.

    :param spacegroup: A space group or IT number identifying the subgroup.
    :param include_self: Include the root IT number in the result.
    :return: Sorted reachable supergroup IT numbers.
    :raises KeyError: If the IT number has no vendored subgroup record.
    """
    return _closure(spacegroup, minimal_supergroups, include_self)


def subgroup_transforms(parent: Spacegroup | int, subgroup: Spacegroup | int) -> tuple[SubgroupTransform, ...]:
    """Return every tabulated transform for one parent/target pair.

    Results retain table order, including self-targeted isomorphic entries. The returned
    tuple is empty when the pair has no tabulated relation.

    :param parent: The parent space group or IT number.
    :param subgroup: The subgroup space group or IT number.
    :return: All exact standard-setting transformations for the pair.
    :raises KeyError: If either IT number has no vendored subgroup record.
    """
    parent_number = _it_number(parent)
    subgroup_number = _it_number(subgroup)
    parent_standard = Spacegroup.standard(parent_number)
    subgroup_standard = Spacegroup.standard(subgroup_number)
    result: list[SubgroupTransform] = []
    for entry in data.spacegroup_subgroup_record(parent_number)["baernighausen"]:
        if entry["target_it_number"] != subgroup_number:
            continue
        for transform in entry["transforms"]:
            splittings: dict[str, tuple[WyckoffSplitPiece, ...]] = {}
            for splitting in transform["wyckoff_splitting"]:
                splittings[splitting["parent"]] = tuple(
                    WyckoffSplitPiece(
                        letter=piece["letter"],
                        xyz=piece["xyz"],
                        operation=_affine_from_3x4(piece["affine"]),
                    )
                    for piece in splitting["splits"]
                )
            result.append(
                SubgroupTransform(
                    parent=parent_standard,
                    subgroup=subgroup_standard,
                    index=transform["index"],
                    subgroup_type=transform["subgroup_type"],
                    k_subtype=transform["k_subtype"],
                    operation=_affine_from_record(transform),
                    splittings=splittings,
                )
            )
    return tuple(result)
