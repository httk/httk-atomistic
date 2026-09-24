"""Exact symbolic Wyckoff actions used by protostructure-first canonicalization."""

import datetime
from fractions import Fraction as F

import pytest
from httk.core import FracVector

from httk.atomistic import (
    ASUStructure,
    Cell,
    ChemicalComposition,
    Species,
    WyckoffSite,
    canonical_asu_protostructure,
    data,
)
from httk.atomistic.symmetry import canonical_protostructure as canonical_module
from httk.atomistic.symmetry._wyckoff_actions import UnsupportedWyckoffAction, compile_wyckoff_action
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.spacegroup import Spacegroup


def _translation(values: tuple[F, F, F]) -> AffineOperation:
    return AffineOperation(FracVector.eye((3, 3)), values)


def _assert_complete_orbit(
    spacegroup: Spacegroup,
    operation: AffineOperation,
    letter: str,
    parameters: tuple[F, ...],
) -> None:
    action = compile_wyckoff_action(spacegroup, operation, letter)
    source = spacegroup.wyckoff_position(letter)
    target = spacegroup.wyckoff_position(action.target_letter)
    actual = {
        tuple(operation.apply_wrapped(point).to_fractions()) for point in source.coordinates(FracVector(parameters))
    }
    expected = {
        tuple(FracVector(point).normalize().to_fractions())
        for point in target.coordinates(action.apply(FracVector(parameters)))
    }
    assert actual == expected
    assert len(actual) == source.multiplicity == target.multiplicity


@pytest.mark.parametrize(
    ("number", "letter", "parameters", "translation"),
    (
        (2, "a", (), (F(0), F(0), F(1, 2))),
        (47, "i", (F(2, 7),), (F(0), F(0), F(1, 2))),
        (47, "u", (F(2, 7), F(3, 11)), (F(0), F(0), F(1, 2))),
        (47, "α", (F(2, 7), F(3, 11), F(5, 13)), (F(0), F(0), F(1, 2))),
        (216, "a", (), (F(1, 4), F(1, 4), F(1, 4))),
        (216, "h", (F(2, 7), F(3, 11)), (F(1, 4), F(1, 4), F(1, 4))),
    ),
)
def test_compiled_action_matches_the_complete_transformed_orbit(
    number: int,
    letter: str,
    parameters: tuple[F, ...],
    translation: tuple[F, F, F],
) -> None:
    spacegroup = Spacegroup.standard(number)
    _assert_complete_orbit(spacegroup, _translation(translation), letter, parameters)


@pytest.mark.parametrize(
    ("number", "coset_index", "letter", "parameters", "target_letter"),
    (
        (47, 4, "u", (F(2, 7), F(3, 11)), "v"),
        (15, 0, "e", (F(2, 7),), "e"),
        (152, 0, "c", (F(2, 7), F(3, 11), F(5, 13)), "c"),
    ),
)
def test_tabulated_matrix_normalizer_compiles_the_complete_orbit(
    number: int,
    coset_index: int,
    letter: str,
    parameters: tuple[F, ...],
    target_letter: str,
) -> None:
    spacegroup = Spacegroup.standard(number)
    record = data.affine_normalizer_coset_record(spacegroup.hall_entry)
    operation = AffineOperation.from_record(record["affine_normalizer_cosets"][coset_index])
    assert operation.matrix != FracVector.eye((3, 3))
    assert compile_wyckoff_action(spacegroup, operation, letter).target_letter == target_letter
    _assert_complete_orbit(spacegroup, operation, letter, parameters)


def test_compiler_rejects_an_operation_that_is_not_a_group_normalizer() -> None:
    spacegroup = Spacegroup.standard(221)
    shear = AffineOperation(((1, 1, 0), (0, 1, 0), (0, 0, 1)), (0, 0, 0))
    with pytest.raises(UnsupportedWyckoffAction, match="221"):
        compile_wyckoff_action(spacegroup, shear, spacegroup.wyckoff[-1].letter)


def test_discrete_partition_prunes_before_parameter_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    species = [
        Species(name="Zn", chemical_symbols=("Zn",), concentration=(1,)),
        Species(name="S", chemical_symbols=("S",), concentration=(1,)),
    ]
    structure = ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        216,
        [WyckoffSite("a", FracVector(()), "Zn"), WyckoffSite("c", FracVector(()), "S")],
        species,
    )
    raw_count = (
        len(canonical_module._representatives(structure))
        * len(canonical_module._discrete_normalizer_translations(structure.spacegroup))
        * len(canonical_module._point_operations(structure))
    )
    monkeypatch.setattr(
        canonical_module,
        "_apply_action",
        lambda *_args: pytest.fail("discrete classification evaluated Wyckoff parameters"),
    )

    surviving = canonical_module._discrete_candidates(structure)

    assert surviving
    assert len(surviving) < raw_count


def _metadata_source() -> ASUStructure:
    return ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        221,
        [WyckoffSite("a", FracVector(()), "Na")],
        [Species(name="Na", chemical_symbols=("Na",), concentration=(1,))],
        chemical_composition=ChemicalComposition({"Na": 1}, mode="full"),
        chemical_formula_descriptive="Na",
        chemical_formula_hill="Na",
        optimization_type="experimental",
        immutable_id="source-1",
        last_modified=datetime.datetime(2026, 9, 24, tzinfo=datetime.UTC),
    )


def _metadata(structure: ASUStructure) -> tuple[object, ...]:
    return (
        structure.chemical_composition,
        structure.chemical_formula_descriptive,
        structure.chemical_formula_hill,
        structure.optimization_type,
        structure.immutable_id,
        structure.last_modified,
    )


def test_exact_seam_preserves_source_metadata_and_is_idempotent() -> None:
    source = _metadata_source()
    first = canonical_module._canonical_protostructure_asu(source)
    second = canonical_module._canonical_protostructure_asu(first)

    assert _metadata(first) == _metadata(second) == _metadata(source)


def test_public_wrapper_restores_metadata_lost_by_recognition() -> None:
    source = _metadata_source()
    result = canonical_asu_protostructure(source, tolerance=1e-3)

    assert _metadata(result) == _metadata(source)


def test_exact_seam_scales_supplied_composition_when_reducing_a_partial_occupancy_supercell() -> None:
    species = [Species(name="Na_half", chemical_symbols=("Na",), concentration=(F(1, 2),))]
    primitive = ASUStructure(
        Cell(((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        1,
        [WyckoffSite("a", FracVector((0, 0, 0)), "Na_half")],
        species,
        chemical_composition=ChemicalComposition({"H": 1}, mode="implicit"),
    )
    supercell = ASUStructure(
        Cell(((2, 0, 0), (0, 1, 0), (0, 0, 1))),
        1,
        [
            WyckoffSite("a", FracVector((0, 0, 0)), "Na_half"),
            WyckoffSite("a", FracVector((F(1, 2), 0, 0)), "Na_half"),
        ],
        species,
        chemical_composition=ChemicalComposition({"H": 2}, mode="implicit"),
    )

    primitive_result = canonical_module._canonical_protostructure_asu(primitive)
    supercell_result = canonical_module._canonical_protostructure_asu(supercell)

    assert primitive_result.chemical_composition is not None
    assert supercell_result.chemical_composition is not None
    assert primitive_result.chemical_composition.amount_mapping["H"] == 1
    assert supercell_result.chemical_composition == primitive_result.chemical_composition
