"""Tests for exact symmetry alignment and interpolation paths."""

import math
import pickle
from fractions import Fraction
from typing import Any

import pytest
from httk.core import FracVector, SurdVector

from httk.atomistic import (
    ASUStructure,
    Cell,
    FundamentalDomainStructure,
    Species,
    StructurePath,
    WyckoffSite,
    data,
    interpolate_structures,
    represent_like,
    same_crystal,
    subgroup_representation,
)
from httk.atomistic.composition import Assembly
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.symmetry import common_subgroup_representation, structure_delta, subgroup_closure
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.lift import _apply_normalizer
from httk.atomistic.symmetry.paths import _pair_travel_score
from httk.atomistic.symmetry.spacegroup import Spacegroup

F = Fraction
NO_PARAMETERS = FracVector(())
ONE_THIRD = F(1, 3)


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _parent(number: int, sites: list[WyckoffSite], basis: object = ((5, 0, 0), (0, 5, 0), (0, 0, 5))) -> ASUStructure:
    return ASUStructure(Cell(basis), number, sites, _species(*(sorted({site.species for site in sites}))))


def _fifteen_parent(value: F = ONE_THIRD, *, charge: F | None = None) -> ASUStructure:
    return ASUStructure(
        Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        15,
        [WyckoffSite("e", FracVector([value]), "Si")],
        _species("Si"),
        charge=charge,
    )


def _child_two(value: F, *, charge: F | None = None) -> ASUStructure:
    parent = _fifteen_parent(charge=charge)
    child = subgroup_representation(parent, 2).asu
    return ASUStructure(
        child.cell,
        child.spacegroup,
        [WyckoffSite("i", FracVector([F(2, 3), F(1, 3), value]), "Si")],
        child.species,
        charge=charge,
    )


def _same_named_species_structure(symbol: str, value: F) -> ASUStructure:
    return ASUStructure(
        Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        2,
        [WyckoffSite("i", FracVector([F(1, 5), F(1, 6), value]), "site")],
        [Species(name="site", chemical_symbols=(symbol,), concentration=(1,))],
    )


def _skew_p1(
    value: F,
    basis: object = ((1, 0, 0), (10, F(1, 100), 0), (0, 0, 1)),
) -> ASUStructure:
    species = _species("C", "O")
    return ASUStructure(
        Cell(basis),
        1,
        [
            WyckoffSite("a", FracVector((0, 0, 0)), "C"),
            WyckoffSite("a", FracVector((value, value, F(1, 10))), "O"),
        ],
        species,
    )


def test_represent_like_recovers_a_normalizer_coset_image() -> None:
    parent = _parent(5, [WyckoffSite("a", FracVector([F(2, 17)]), "Si")], ((5, 0, 0), (0, 6, 0), (0, 0, 7)))
    original = subgroup_representation(parent, 3).asu
    coset = data.affine_normalizer_coset_record(original.spacegroup.hall_entry)["affine_normalizer_cosets"][0]
    cosetted = _apply_normalizer(original, coset)
    assert cosetted is not None

    result = represent_like(cosetted, original)

    assert result.spacegroup == original.spacegroup
    assert result.transform == original.transform
    assert tuple(site.free_params.to_fractions() for site in result.wyckoff_sites) == tuple(
        site.free_params.to_fractions() for site in original.wyckoff_sites
    )


def test_non_involutory_normalizer_transforms_the_cell_once() -> None:
    species = Species(name="site", chemical_symbols=("C",), concentration=(1,))
    source = ASUStructure(
        Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        143,
        [WyckoffSite("d", FracVector([F(1, 7), F(2, 11), F(3, 13)]), "site")],
        [species],
    )

    def non_involutory(candidate: dict[str, Any]) -> bool:
        candidate_operation = AffineOperation.from_record(candidate)
        return candidate_operation.matrix * candidate_operation.matrix != FracVector.eye((3, 3))

    coset = next(
        candidate
        for candidate in data.affine_normalizer_coset_record(source.spacegroup.hall_entry)["affine_normalizer_cosets"]
        if non_involutory(candidate)
    )
    reference = _apply_normalizer(source, coset)
    assert reference is not None

    result = represent_like(source, reference)

    assert result.cell.basis == reference.cell.basis


def test_represent_like_descends_and_rejects_unrelated_or_mismatched_inputs() -> None:
    parent = _fifteen_parent()
    child = subgroup_representation(parent, 2).asu
    result = represent_like(parent, child)
    assert result.spacegroup.it_number == 2
    assert result.wyckoff_sites == child.wyckoff_sites

    with pytest.raises(ValueError):
        represent_like(_parent(5, [WyckoffSite("a", FracVector([F(1, 3)]), "Si")]), parent)
    mismatch = ASUStructure(child.cell, 2, [WyckoffSite("a", NO_PARAMETERS, "Si")], child.species)
    with pytest.raises(ValueError, match="signatures.*Si.*i.*Si.*a"):
        represent_like(child, mismatch)


def test_common_subgroup_221_123_166_selects_15_and_preserves_descents() -> None:
    parent = _parent(221, [WyckoffSite("a", NO_PARAMETERS, "C")])
    first_input = subgroup_representation(parent, 123).asu
    second_input = subgroup_representation(parent, 166).asu
    result = common_subgroup_representation(first_input, second_input)

    common = set(subgroup_closure(123, include_self=True)) & set(subgroup_closure(166, include_self=True))
    ordered = sorted(common, key=lambda number: (-len(Spacegroup.standard(number).symmetry_operations), -number))
    assert result.spacegroup.it_number == 15
    assert result.spacegroup.it_number == ordered[0]
    assert result.first.spacegroup == result.second.spacegroup == result.spacegroup
    assert same_crystal(result.first, represent_like(first_input, result.first))
    assert same_crystal(result.second, represent_like(second_input, result.second))
    assert result.first.expand_sites().reduced_coords.to_fractions()
    assert result.second.expand_sites().reduced_coords.to_fractions()


def test_structure_delta_measures_physical_orbit_travel_and_lattice_change() -> None:
    start = _child_two(F(1, 4))
    end = _child_two(F(1, 5))

    assert structure_delta(start, start) == 0.0
    # The P-1 general position has two sites.  Each moves 7 * (1/4 - 1/5) = 7/20 A.
    assert structure_delta(start, end) == pytest.approx(7 / 10)

    wrapped_start = _child_two(F(1, 10))
    wrapped_end = _child_two(F(9, 10))
    # The shortest periodic-image pairing of the two P-1 orbit members totals 2 A.
    assert structure_delta(wrapped_start, wrapped_end) == pytest.approx(2)

    stretched = ASUStructure(
        Cell(((F(5, 2), -3, 0), (F(5, 2), 3, 0), (0, 0, 8))),
        end.spacegroup,
        end.wyckoff_sites,
        end.species,
    )
    # The two z = +/- 1/5 sites each move 1/5 A as c changes from 7 to 8 A.
    assert structure_delta(end, stretched) == pytest.approx(2 / 5)


def test_structure_delta_terminates_for_near_identical_rocksalt() -> None:
    # Regression: two rocksalts differing only by a small isotropic lattice change did not
    # finish within 90 s, because the search re-represented and doubly aligned every one of
    # ~200 common subgroups. The bounded first-finite-subgroup rule returns promptly.
    import time

    def _rocksalt(edge: Fraction | int) -> ASUStructure:
        cell = ((edge, 0, 0), (0, edge, 0), (0, 0, edge))
        return ASUStructure(
            Cell(cell),
            225,
            (WyckoffSite("a", NO_PARAMETERS, "Na"), WyckoffSite("b", NO_PARAMETERS, "Cl")),
            _species("Na", "Cl"),
        )

    first = _rocksalt(5)
    second = _rocksalt(F(51, 10))
    started = time.monotonic()
    delta = structure_delta(first, second)
    elapsed = time.monotonic() - started
    # A ~2% isotropic lattice change is a fraction of an angstrom of total physical travel,
    # aligned in the inputs' own space group 225 rather than a same-order sibling.
    assert math.isfinite(delta) and 0.0 < delta < 1.0
    assert structure_delta(second, first) == delta
    assert elapsed <= 10.0, f"structure_delta took {elapsed:.1f} s"


def test_structure_delta_is_normalizer_invariant_and_uses_common_group() -> None:
    parent = _parent(5, [WyckoffSite("a", FracVector([F(2, 17)]), "Si")], ((5, 0, 0), (0, 6, 0), (0, 0, 7)))
    original = subgroup_representation(parent, 3).asu
    coset = data.affine_normalizer_coset_record(original.spacegroup.hall_entry)["affine_normalizer_cosets"][0]
    cosetted = _apply_normalizer(original, coset)
    assert cosetted is not None
    assert structure_delta(original, cosetted) == 0.0

    cubic_parent = _parent(221, [WyckoffSite("a", NO_PARAMETERS, "C")])
    first = subgroup_representation(cubic_parent, 123).asu
    second = subgroup_representation(cubic_parent, 166).asu
    assert structure_delta(first, second) == 0.0


def test_structure_delta_rejects_incompatible_signatures_and_is_deterministic() -> None:
    carbon = _same_named_species_structure("C", F(1, 10))
    oxygen = _same_named_species_structure("O", F(1, 8))
    with pytest.raises(ValueError, match="common subgroup"):
        structure_delta(carbon, oxygen)

    start = _child_two(F(1, 4))
    end = _child_two(F(1, 5))
    assert structure_delta(start, end) == structure_delta(start, end)


def test_structure_delta_accepts_an_exact_fundamental_domain() -> None:
    source = _child_two(F(1, 5))
    fundamental = FundamentalDomainStructure(
        source.cell,
        source.spacegroup,
        source.wyckoff_sites,
        source.species,
        transform=source.transform,
        coordinate_precision=source.coordinate_precision,
        charge=source.charge,
    )

    assert structure_delta(fundamental, fundamental) == 0.0


def test_structure_delta_uses_a_symmetric_skew_cell_minimum_image() -> None:
    start = _skew_p1(F(1, 100))
    end = _skew_p1(F(49, 100))

    # The shortest image needs a translation of five first lattice vectors and one
    # second lattice vector; the fixed [-1, 1]^3 image box would miss it.  The
    # mean-metric convention is also invariant under reversing the path.
    expected = math.hypot(7 / 25, 3 / 625)
    assert structure_delta(start, end) == pytest.approx(expected)
    assert structure_delta(end, start) == pytest.approx(expected)


def test_structure_delta_is_rotation_invariant_for_unequal_endpoint_cells() -> None:
    start = _skew_p1(F(1, 100))
    end = _skew_p1(F(49, 100), ((F(6, 5), 0, 0), (8, F(1, 80), 0), (0, 0, F(11, 10))))
    rotation = FracVector(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
    rotated_start = _skew_p1(F(1, 100), SurdVector(start.cell.basis) * rotation)
    rotated_end = _skew_p1(F(49, 100), SurdVector(end.cell.basis) * rotation)

    assert structure_delta(rotated_start, rotated_end) == pytest.approx(structure_delta(start, end))


def test_pair_travel_score_assigns_large_repeated_classes_without_permutations() -> None:
    species = _species("Si")
    reference = ASUStructure(
        Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        1,
        [WyckoffSite("a", FracVector((F(index, 10), F(1, 7), F(1, 9))), "Si") for index in range(9)],
        species,
    )
    candidate = ASUStructure(
        reference.cell,
        reference.spacegroup,
        list(reversed(reference.wyckoff_sites)),
        reference.species,
    )

    score, pairs = _pair_travel_score(candidate, reference)

    assert score == pytest.approx(0.0)
    assert pairs == tuple((index, 8 - index) for index in range(9))


def test_interpolation_is_exact_and_keeps_the_shared_setting() -> None:
    start = _child_two(F(1, 4))
    end = _child_two(F(1, 5))
    path = interpolate_structures(start, end, steps=5)

    assert isinstance(path, StructurePath)
    assert len(path.frames) == 5
    assert path.frames[0] == path.start
    assert path.frames[-1] == path.end
    assert all(frame.spacegroup == path.spacegroup for frame in path.frames)
    assert path.frames[2].wyckoff_sites[0].free_params.to_fractions()[-1] == F(9, 40)
    assert path.frames[2].cell.basis == (SurdVector(start.cell.basis) + SurdVector(end.cell.basis)) * F(1, 2)
    assert all(
        all(isinstance(value, F) for value in site.free_params.to_fractions())
        for frame in path.frames
        for site in frame.wyckoff_sites
    )


def test_interpolation_wraps_the_short_way() -> None:
    path = interpolate_structures(_child_two(F(1, 10)), _child_two(F(9, 10)), steps=3)
    assert path.frames[1].wyckoff_sites[0].free_params.to_fractions()[-1] == F(0)


def test_same_name_different_species_definitions_are_rejected() -> None:
    carbon = _same_named_species_structure("C", F(1, 10))
    oxygen = _same_named_species_structure("O", F(1, 8))

    with pytest.raises(ValueError, match="signatures"):
        represent_like(oxygen, carbon)
    with pytest.raises(ValueError, match="signatures"):
        interpolate_structures(carbon, oxygen, steps=3)
    assert len(interpolate_structures(carbon, _same_named_species_structure("C", F(1, 9)), steps=3).frames) == 3


def test_interpolation_rejects_steps_charges_and_orbit_collision() -> None:
    with pytest.raises(ValueError, match="steps >= 2"):
        interpolate_structures(_child_two(F(1, 4)), _child_two(F(1, 5)), steps=1)
    with pytest.raises(ValueError, match="equal charges"):
        interpolate_structures(_child_two(F(1, 4), charge=F(1)), _child_two(F(1, 5), charge=F(2)), steps=3)

    species = _species("C", "O")
    start = ASUStructure(
        Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        2,
        [
            WyckoffSite("a", NO_PARAMETERS, "C"),
            WyckoffSite("i", FracVector([F(1, 4), F(1, 5), F(1, 5)]), "O"),
        ],
        species,
    )
    end = ASUStructure(
        start.cell,
        2,
        [
            WyckoffSite("a", NO_PARAMETERS, "C"),
            WyckoffSite("i", FracVector([F(3, 4), F(4, 5), F(4, 5)]), "O"),
        ],
        species,
    )
    with pytest.raises(ValueError, match="interpolation step 1"):
        interpolate_structures(start, end, steps=3)


def test_rejection_guards_and_determinism() -> None:
    structure = _fifteen_parent()
    with pytest.raises(ValueError, match="fully 3D-periodic"):
        ASUStructure(
            Cell(((5, 0, 0), (0, 6, 0), (0, 0, 1)), periodicity=(True, True, False)),
            15,
            structure.wyckoff_sites,
            structure.species,
        )
    moment = ASUStructure(
        structure.cell,
        structure.spacegroup,
        [WyckoffSite("e", FracVector([F(1, 3)]), "Si", moment=CollinearSiteMoments([1]))],
        structure.species,
    )
    assembly = ASUStructure(
        structure.cell,
        structure.spacegroup,
        structure.wyckoff_sites,
        structure.species,
        assemblies=(Assembly(((0,),), (1,)),),
    )
    molecular = ASUStructure(
        structure.cell, structure.spacegroup, structure.wyckoff_sites, structure.species, molecular=True
    )
    for rejected in (moment, assembly, molecular):
        with pytest.raises(ValueError):
            represent_like(rejected, structure)

    first = interpolate_structures(_child_two(F(1, 4)), _child_two(F(1, 5)), steps=5)
    second = interpolate_structures(_child_two(F(1, 4)), _child_two(F(1, 5)), steps=5)
    assert pickle.dumps(first) == pickle.dumps(second)
    assert pickle.dumps(represent_like(structure, structure)) == pickle.dumps(represent_like(structure, structure))
