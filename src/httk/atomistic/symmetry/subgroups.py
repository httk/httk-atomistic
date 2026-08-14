"""Graph queries and exact transformations for Bärnighausen subgroup trees.

The vendored tables contain maximal subgroup relations and their standard-setting
transformations for all 230 space-group types. Only those one-hop relations are
tabulated; closures and the inverted supergroup graph are derived here.

The split-affine convention pinned by ``tests/test_subgroups.py`` is that a
:class:`WyckoffSplitPiece` operation maps a parent standard-setting coordinate directly
to a child standard-setting coordinate. The tabulated
:class:`SubgroupTransform` operation has matrix ``M`` with the child basis on the
left: ``B_child = M.T() * B_parent``. At the affine-coordinate level it maps child
coordinates into parent coordinates, ``f_parent = f_child * M.T() + v``; its inverse
matrix is the coordinate basis change used by the split operations (up to their listed
origin translations).
"""

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from types import MappingProxyType
from typing import Any

from httk.core import SurdVector

from httk.atomistic import data
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.structure.asu import ASUStructure, WyckoffSite
from httk.atomistic.symmetry._periodicity_guard import require_full_periodicity
from httk.atomistic.symmetry._standardization_common import (
    _matrix_column_sum_factor,
    _matrix_row_sum_factor,
    _scaled_precision,
)
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup

__all__ = [
    "SubgroupRepresentationResult",
    "SubgroupTransform",
    "WyckoffSplitPiece",
    "maximal_subgroups",
    "minimal_supergroups",
    "subgroup_closure",
    "subgroup_transforms",
    "supergroup_closure",
]


@dataclass(frozen=True, slots=True)
class SubgroupRepresentationResult:
    """Store an exact asymmetric-unit representation in a subgroup.

    :param asu: The subgroup-standard-setting asymmetric unit with identity transform.
    :param spacegroup: The subgroup space group in its standard setting.
    :param path: The selected maximal-subgroup transforms, in parent-first order.
    :param multiplier: The exact child-to-parent cell-content ratio.
    """

    asu: ASUStructure
    spacegroup: Spacegroup
    path: tuple["SubgroupTransform", ...]
    multiplier: Fraction


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
    :param operation: The exact tabulated affine basis change from child to parent
        coordinates. If its matrix is ``M`` and vector is ``v``, it evaluates as
        ``f_parent = f_child * M.T() + v``; the child basis is ``M.T() * parent_basis``.
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


def _wrapped_coordinate_key(coordinate: Any) -> tuple[Fraction, ...]:
    """Return one exact wrapped coordinate as a hashable tuple."""
    return tuple(coordinate.normalize().to_fractions())


def _orbit_key(position: Any, parameters: Any) -> frozenset[tuple[Fraction, ...]]:
    """Return the exact wrapped coordinate set naming one child orbit."""
    return frozenset(_wrapped_coordinate_key(point) for point in position.coordinates(parameters))


def _standard_input(structure: ASUStructure) -> ASUStructure:
    """Normalize an ASU cell to its standard frame while retaining exact metadata."""
    # Already in the IT standard setting with an identity transform: the rebuild below is a pure
    # copy (identity basis, precision factor 1, own spacegroup/sites), so return the input unchanged.
    if structure.spacegroup.is_standard_setting and structure.transform_from_standard.is_identity():
        return structure
    transform = structure.transform_from_standard
    standard, sites = structure._standard_wyckoff_sites()
    basis_matrix = transform.matrix.T()
    new_cell = Cell(
        transform.basis_to_standard(structure.cell.basis),
        precision=_scaled_precision(structure.cell.precision, _matrix_row_sum_factor(basis_matrix)),
        periodicity=structure.cell.periodicity,
    )
    return ASUStructure(
        new_cell,
        standard,
        tuple(WyckoffSite(site.wyckoff, site.free_params, site.species) for site in sites),
        structure.species,
        transform=SettingTransform.identity(),
        coordinate_precision=_scaled_precision(
            structure.coordinate_precision,
            _matrix_column_sum_factor(basis_matrix.inv()),
        ),
        charge=structure.charge,
    )


def _child_sites(parent: ASUStructure, transform: SubgroupTransform) -> tuple[WyckoffSite, ...]:
    """Map every parent ASU site through one tabulated split entry."""
    sites: list[WyckoffSite] = []
    seen: set[frozenset[tuple[Fraction, ...]]] = set()
    for site in parent.wyckoff_sites:
        pieces = transform.splittings.get(site.wyckoff)
        if pieces is None:
            raise ValueError(f"no splitting for parent Wyckoff site {site.wyckoff!r}")
        parent_point = parent.spacegroup.wyckoff_position(site.wyckoff).representative.coordinate(site.free_params)
        for piece in pieces:
            child_point = piece.operation.apply_wrapped(parent_point)
            identified = transform.subgroup.identify_wyckoff(child_point)
            if identified is None:
                raise ValueError(f"parent site {site!r} maps to an unidentified child point")
            position, parameters = identified
            orbit = _orbit_key(position, parameters)
            if orbit in seen:
                continue
            seen.add(orbit)
            sites.append(WyckoffSite(position.letter, parameters, site.species))
    return tuple(sites)


def _hop(
    parent: ASUStructure,
    subgroup: Spacegroup,
) -> tuple[ASUStructure, SubgroupTransform]:
    """Apply the first table entry that exactly recognizes all parent sites."""
    last_error: ValueError | None = None
    for transform in subgroup_transforms(parent.spacegroup, subgroup):
        try:
            sites = _child_sites(parent, transform)
        except ValueError as error:
            last_error = error
            continue
        matrix = transform.operation.matrix.T()
        child_cell = Cell(
            SurdVector(matrix) * SurdVector(parent.cell.basis),
            precision=_scaled_precision(parent.cell.precision, _matrix_row_sum_factor(matrix)),
            periodicity=parent.cell.periodicity,
        )
        child = ASUStructure(
            child_cell,
            transform.subgroup,
            sites,
            parent.species,
            transform=SettingTransform.identity(),
            coordinate_precision=_scaled_precision(
                parent.coordinate_precision,
                _matrix_column_sum_factor(matrix.inv()),
            ),
            charge=parent.charge,
        )
        return child, transform
    detail = "" if last_error is None else f": {last_error}"
    raise ValueError(
        f"no subgroup transform from {parent.spacegroup.setting} to {subgroup.setting} maps every site{detail}"
    )


def _shortest_subgroup_path(source: int, target: int) -> tuple[int, ...]:
    """Return the deterministic shortest IT-number path, excluding the source."""
    pending: deque[tuple[int, tuple[int, ...]]] = deque()
    pending.append((source, ()))
    visited = {source}
    while pending:
        current, path = pending.popleft()
        for child in maximal_subgroups(current):
            if child in visited:
                continue
            next_path = path + (child,)
            if child == target:
                return next_path
            visited.add(child)
            pending.append((child, next_path))
    raise ValueError(f"space group {target} is not in the subgroup closure of space group {source}")


def subgroup_representation(
    structure: ASUStructure,
    subgroup: Spacegroup | int,
) -> SubgroupRepresentationResult:
    """Express an exact ASU in a subgroup's IT standard setting.

    :param structure: The fully periodic parent asymmetric-unit structure.
    :param subgroup: The target subgroup space group or IT number.
    :return: The child ASU, selected maximal-subgroup path, and exact multiplier.
    :raises TypeError: If ``structure`` is not an :class:`ASUStructure`.
    :raises ValueError: If the structure is not fully periodic, carries site moments,
        assemblies, or molecular semantics, or if the target is not reachable.
    """
    if not isinstance(structure, ASUStructure):
        raise TypeError(f"expected ASUStructure, got {type(structure).__name__}")
    require_full_periodicity(structure.cell, "subgroup_representation")
    if any(site.moment is not None for site in structure.wyckoff_sites):
        raise ValueError("subgroup_representation does not support structures with site moments")
    if structure.assemblies is not None:
        raise ValueError("subgroup_representation does not support structures with assemblies")
    if structure.molecular:
        raise ValueError("subgroup_representation does not support molecular structures")

    target_number = _it_number(subgroup)
    current = _standard_input(structure)
    if current.spacegroup.it_number == target_number:
        return SubgroupRepresentationResult(current, current.spacegroup, (), Fraction(1))

    path_numbers = _shortest_subgroup_path(current.spacegroup.it_number, target_number)
    path: list[SubgroupTransform] = []
    multiplier = Fraction(1)
    for child_number in path_numbers:
        child_group = Spacegroup.standard(child_number)
        parent_count = len(current.expand_sites())
        child, transform = _hop(current, child_group)
        child_count = len(child.expand_sites())
        if parent_count == 0:
            raise ValueError("subgroup_representation cannot determine a multiplier for an empty ASU")
        hop_multiplier = Fraction(child_count, parent_count)
        if child.cell.periodic_measure != current.cell.periodic_measure * hop_multiplier:
            raise ValueError(
                f"subgroup table multiplier mismatch for {current.spacegroup.setting} -> {child_group.setting}"
            )
        multiplier *= hop_multiplier
        child_charge = None if current.charge is None else current.charge * hop_multiplier
        if child_charge is not None:
            child = ASUStructure(
                child.cell,
                child.spacegroup,
                child.wyckoff_sites,
                child.species,
                transform=SettingTransform.identity(),
                coordinate_precision=child.coordinate_precision,
                charge=child_charge,
            )
        current = child
        path.append(transform)
    return SubgroupRepresentationResult(current, current.spacegroup, tuple(path), multiplier)
