"""Declared space-group operations: basis-independent compatibility and cell metric."""

from collections import defaultdict
from fractions import Fraction as F

import pytest
from httk.core import FracVector

from httk.atomistic import UnitcellStructure, data
from httk.atomistic.models.structure.semantics import (
    StructureSymmetry,
    _operation_group_invariant,
    _plain_operation,
    _standard_group_invariant,
    _tabulated_group,
)
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.spacegroup import Spacegroup

_HALF = F(1, 2)


def _rebased(it_number, basis):
    """Standard-setting operations of ``it_number`` re-expressed in a new cell ``basis``.

    ``basis`` holds the new cell vectors as rows in the standard cell's fractional
    coordinates; operations whose rotation is not integral in the new basis are dropped.
    """
    change = AffineOperation(FracVector(basis).T().inv())
    operations = set()
    for operation in Spacegroup.standard(it_number).symmetry_operations:
        rebased = operation.conjugated_by(change).wrapped()
        if all(value.denominator == 1 for row in rebased.matrix.to_fractions() for value in row):
            operations.add(rebased)
    return operations


def _xyz(operations):
    return tuple(sorted(operation.to_xyz() for operation in operations))


def test_every_tabulated_setting_of_a_number_has_the_same_invariant():
    invariants = defaultdict(set)
    for record in data.spacegroup_settings():
        invariants[record["it_number"]].add(_operation_group_invariant(_tabulated_group(record["hall_entry"])))
    assert sorted(invariants) == list(range(1, 231))
    assert all(len(values) == 1 for values in invariants.values())
    # The exact collision partition documented on _operation_group_invariant: eleven
    # enantiomorphic pairs and six further pairs; every other type is separated.
    numbers = defaultdict(list)
    for number, values in invariants.items():
        numbers[next(iter(values))].append(number)
    assert sorted(group for group in numbers.values() if len(group) > 1) == [
        [23, 24],
        [32, 34],
        [76, 78],
        [91, 95],
        [92, 96],
        [117, 118],
        [144, 145],
        [151, 153],
        [152, 154],
        [169, 170],
        [171, 172],
        [178, 179],
        [180, 181],
        [185, 186],
        [193, 194],
        [197, 199],
        [212, 213],
    ]


def test_glide_arrangement_and_centring_are_told_apart():
    pbam = _xyz(value.wrapped() for value in Spacegroup.standard(55).symmetry_operations)
    origin_shift = AffineOperation(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (F(1, 7), F(2, 11), F(3, 13)))
    shifted = _xyz(value.conjugated_by(origin_shift).wrapped() for value in Spacegroup.standard(55).symmetry_operations)
    assert StructureSymmetry(space_group_it_number=55, space_group_symmetry_operations_xyz=shifted)
    for number in (57, 58):
        with pytest.raises(ValueError, match="different type"):
            StructureSymmetry(space_group_it_number=number, space_group_symmetry_operations_xyz=pbam)
    fm3m = _xyz(value.wrapped() for value in Spacegroup.standard(225).symmetry_operations)
    with pytest.raises(ValueError, match="different type"):
        StructureSymmetry(space_group_it_number=229, space_group_symmetry_operations_xyz=fm3m)


def test_matched_settings_are_read_only_copies():
    symmetry = StructureSymmetry(space_group_it_number=225)
    with pytest.raises(TypeError):
        symmetry.matched_settings[0]["hall_entry"] = "x"  # type: ignore[index]
    with pytest.raises(AttributeError):
        symmetry.matched_settings[0]["symops"].append(None)
    assert data.standard_spacegroup_setting(225)["hall_entry"] == symmetry.matched_settings[0]["hall_entry"]


def test_primitive_fcc_operations_are_accepted_for_225():
    primitive = _rebased(225, ((0, _HALF, _HALF), (_HALF, 0, _HALF), (_HALF, _HALF, 0)))
    assert len(primitive) == 48
    assert _operation_group_invariant(_plain_operation(value) for value in primitive) == _standard_group_invariant(225)
    operations = _xyz(primitive)
    symmetry = StructureSymmetry(space_group_it_number=225, space_group_symmetry_operations_xyz=operations)
    assert symmetry.space_group_symmetry_operations_xyz == operations
    assert {record["it_number"] for record in symmetry.matched_settings} == {225}
    a = F(18, 10)
    structure = UnitcellStructure(
        [[0, a, a], [a, 0, a], [a, a, 0]], [[0, 0, 0]], species_at_sites=["Cu"], symmetry=symmetry
    )
    assert structure.space_group_symmetry_operations_xyz == operations
    with pytest.raises(ValueError, match="different type"):
        StructureSymmetry(space_group_it_number=229, space_group_symmetry_operations_xyz=operations)


def test_supercell_operations_are_rejected_for_the_parent_type():
    supercell = _rebased(221, ((2, 0, 0), (0, 1, 0), (0, 0, 1)))
    shift = AffineOperation(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (_HALF, 0, 0))
    operations = _xyz(supercell | {(shift * operation).wrapped() for operation in supercell})
    assert len(operations) == 32
    with pytest.raises(ValueError, match="not compatible with space group 221: they form a group of a different type"):
        StructureSymmetry(space_group_it_number=221, space_group_symmetry_operations_xyz=operations)
    # The same operations alone are a valid group of unidentified type.
    assert StructureSymmetry(space_group_symmetry_operations_xyz=operations).matched_settings == ()


def test_symbol_only_operations_compare_with_the_symbol_settings():
    primitive = _xyz(_rebased(225, ((0, _HALF, _HALF), (_HALF, 0, _HALF), (_HALF, _HALF, 0))))
    assert StructureSymmetry(
        space_group_symbol_hermann_mauguin="F m -3 m", space_group_symmetry_operations_xyz=primitive
    )
    with pytest.raises(ValueError, match="disagree with its number or symbols"):
        StructureSymmetry(space_group_symbol_hermann_mauguin="I m -3 m", space_group_symmetry_operations_xyz=primitive)


def test_declared_operations_must_preserve_the_cell_metric():
    conventional = _xyz(value.wrapped() for value in Spacegroup.standard(225).symmetry_operations)
    symmetry = StructureSymmetry(space_group_it_number=225, space_group_symmetry_operations_xyz=conventional)
    UnitcellStructure([[4, 0, 0], [0, 4, 0], [0, 0, 4]], [[0, 0, 0]], species_at_sites=["Cu"], symmetry=symmetry)
    for cell in ([[0, 2, 2], [2, 0, 2], [2, 2, 0]], [[3, 0, 0], [F(1, 3), 4, 0], [F(1, 7), F(2, 5), 5]]):
        with pytest.raises(ValueError, match="do not preserve the cell metric"):
            UnitcellStructure(cell, [[0, 0, 0]], species_at_sites=["Cu"], symmetry=symmetry)
    # A number alone states no operations, so any cell can carry it.
    number_only = UnitcellStructure(
        [[0, 2, 2], [2, 0, 2], [2, 2, 0]],
        [[0, 0, 0]],
        species_at_sites=["Cu"],
        symmetry=StructureSymmetry(space_group_it_number=225),
    )
    assert number_only.space_group_symmetry_operations_xyz is None


def test_float_hexagonal_cells_pass_the_metric_check():
    operations = _xyz(value.wrapped() for value in Spacegroup.standard(191).symmetry_operations)
    symmetry = StructureSymmetry(space_group_it_number=191, space_group_symmetry_operations_xyz=operations)
    for height in (2.598076211353316, 2.5981):
        structure = UnitcellStructure(
            [[3, 0, 0], [-1.5, height, 0], [0, 0, 5]], [[0, 0, 0]], species_at_sites=["Mg"], symmetry=symmetry
        )
        assert structure.space_group_symmetry_operations_xyz == operations


def test_metric_tolerance_is_relative_to_each_entry():
    operations = _xyz(value.wrapped() for value in Spacegroup.standard(123).symmetry_operations)
    symmetry = StructureSymmetry(space_group_it_number=123, space_group_symmetry_operations_xyz=operations)
    UnitcellStructure([[3, 0, 0], [0, 3, 0], [0, 0, 100]], [[0, 0, 0]], species_at_sites=["Cu"], symmetry=symmetry)
    for a, b in ((3, F(31, 10)), (1, F(6, 5))):
        with pytest.raises(ValueError, match="do not preserve the cell metric"):
            UnitcellStructure(
                [[a, 0, 0], [0, b, 0], [0, 0, 100]], [[0, 0, 0]], species_at_sites=["Cu"], symmetry=symmetry
            )
