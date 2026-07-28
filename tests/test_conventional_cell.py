"""Tests for exact conversion to IT standard-setting conventional cells."""

import fractions
import importlib
from typing import Any

import pytest
from httk.core import FracVector, SurdVector

from httk.atomistic import (
    ASUSite,
    ASUStructure,
    ASUStructureView,
    Cell,
    SettingTransform,
    Spacegroup,
    Species,
    Structure,
    StructureASU,
    StructureLike,
    UnitcellStructureView,
    conventional_cell,
    recognize_asu,
    same_crystal,
)

F = fractions.Fraction
NO_PARAMETERS = FracVector.create(())
CUBIC = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
ORTHO = [[5, 0, 0], [0, 6, 0], [0, 0, 7]]


def _species(*names: str) -> list[Species]:
    return [
        Species(name=name, chemical_symbols=(name,), concentration=(1.0,))
        for name in names
    ]


def _rocksalt() -> ASUStructure:
    return ASUStructure(
        CUBIC,
        225,
        [ASUSite("a", NO_PARAMETERS, "Na"), ASUSite("b", NO_PARAMETERS, "Cl")],
        _species("Na", "Cl"),
    )


def _monoclinic() -> tuple[ASUStructure, SettingTransform]:
    transform = Spacegroup.for_setting("15:c1").transform_from_standard
    return (
        ASUStructure(
            ORTHO,
            15,
            [ASUSite("e", FracVector.create(["1/3"]), "Si")],
            _species("Si"),
            transform=transform,
        ),
        transform,
    )


def _hexagonal_basis_pair() -> tuple[SurdVector, SurdVector]:
    """Literal own/standard bases for the SG 166 rhombohedral setting."""
    zero = SurdVector.create(0)._as_scalar()
    two = SurdVector.create(2)._as_scalar()
    four = SurdVector.create(4)._as_scalar()
    minus_two = SurdVector.create(-2)._as_scalar()
    twelve = SurdVector.create(12)._as_scalar()
    root_three = SurdVector.sqrt_of(3)
    standard = SurdVector._from_scalar_grid(
        [
            [four, zero, zero],
            [minus_two, root_three * two, zero],
            [zero, zero, twelve],
        ],
        (3, 3),
    )
    own = SurdVector._from_scalar_grid(
        [
            [zero, -(root_three * F(4, 3)), four],
            [two, root_three * F(2, 3), four],
            [minus_two, root_three * F(2, 3), four],
        ],
        (3, 3),
    )
    return own, standard


def test_a_standard_setting_is_unchanged() -> None:
    asu = _rocksalt()

    result = conventional_cell(asu)

    assert result.structure.cell.basis == asu.cell.basis
    assert result.asu.transform.is_identity()
    assert result.multiplier == F(1)
    assert same_crystal(result.structure, UnitcellStructureView(asu))


def test_repeating_a_result_uses_its_unwrapped_standard_asu_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("httk.atomistic.standardization")
    first = conventional_cell(_monoclinic()[0])

    def fail_recognition(*args: object, **kwargs: object) -> None:
        raise AssertionError("exact ASU dispatch should not call recognize_asu")

    monkeypatch.setattr(module, "recognize_asu", fail_recognition)
    repeated = conventional_cell(first.structure)

    assert repeated.multiplier == F(1)
    assert repeated.structure.cell.basis == first.structure.cell.basis
    assert repeated.structure == first.structure


def test_asu_views_and_backends_are_used_without_recognition() -> None:
    asu = _rocksalt()

    assert conventional_cell(ASUStructureView(asu)).asu == conventional_cell(asu).asu
    assert conventional_cell(StructureASU(asu)).asu == conventional_cell(asu).asu


def test_a_nonstandard_setting_is_mapped_back_to_the_standard_cell() -> None:
    asu, transform = _monoclinic()
    original = UnitcellStructureView(asu)

    result = conventional_cell(asu)
    mapped = Structure(
        result.structure.cell,
        [
            transform.to_standard(row).normalize()
            for row in original.sites.reduced_coords
        ],
        original.species,
        original.species_at_sites,
    )

    assert result.spacegroup == Spacegroup.standard(15)
    assert result.asu.transform.is_identity()
    assert result.structure.cell.basis == transform.basis_to_standard(asu.cell.basis)
    assert len(result.structure.sites) == len(original.sites)
    assert sorted(result.structure.species_at_sites) == sorted(
        original.species_at_sites
    )
    assert same_crystal(mapped, result.structure)
    recognized = recognize_asu(result.structure, setting=result.spacegroup)
    assert recognized.spacegroup.it_number == 15
    assert recognized.is_standard_setting


def test_rhombohedral_setting_expands_to_three_standard_cell_sites() -> None:
    transform = Spacegroup.for_setting("166:R").transform_from_standard
    rhombohedral_basis, expected_basis = _hexagonal_basis_pair()
    asu = ASUStructure(
        rhombohedral_basis,
        166,
        [ASUSite("a", NO_PARAMETERS, "Bi")],
        _species("Bi"),
        transform=transform,
    )

    result = conventional_cell(asu)

    assert len(UnitcellStructureView(asu).sites) == 1
    assert len(result.structure.sites) == 3
    assert result.structure.cell.basis == expected_basis
    assert {
        tuple(row) for row in result.structure.sites.reduced_coords.to_fractions()
    } == {
        (F(0), F(0), F(0)),
        (F(1, 3), F(2, 3), F(2, 3)),
        (F(2, 3), F(1, 3), F(1, 3)),
    }
    assert result.multiplier == F(3)


def test_precision_is_scaled_by_the_exact_induced_matrix_norms() -> None:
    transform = SettingTransform([[1, 2, 0], [0, 1, 1], [0, 0, 1]])
    asu = ASUStructure(
        Cell(
            [[5, 0, 0], [-10, 5, 0], [10, -5, 5]],
            precision=F(1, 50),
        ),
        221,
        [ASUSite("a", NO_PARAMETERS, "C")],
        _species("C"),
        transform=transform,
        coordinate_precision=F(1, 1000),
    )

    result = conventional_cell(asu)

    assert result.structure.cell.basis == SurdVector.create(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
    )
    assert result.structure.cell.precision == F(3, 50)
    assert result.asu.coordinate_precision == F(1, 200)


def test_an_untabulated_half_determinant_transform_can_have_a_subunit_multiplier() -> (
    None
):
    transform = SettingTransform([[F(1, 2), 0, 0], [0, 1, 0], [0, 0, 1]])
    asu = ASUStructure(
        [[10, 0, 0], [0, 5, 0], [0, 0, 5]],
        221,
        [ASUSite("a", NO_PARAMETERS, "C")],
        _species("C"),
        transform=transform,
    )

    result = conventional_cell(asu)

    assert result.multiplier == F(1, 2)


def test_plain_structure_path_matches_recognized_asu_path_and_forwards_tolerance() -> (
    None
):
    expanded = UnitcellStructureView(_rocksalt())
    plain = Structure(
        expanded.cell,
        expanded.sites,
        expanded.species,
        expanded.species_at_sites,
    )
    direct = conventional_cell(plain)
    expected = conventional_cell(recognize_asu(plain))
    assert direct.structure == expected.structure

    one_site = UnitcellStructureView(
        ASUStructure(CUBIC, 221, [ASUSite("a", NO_PARAMETERS, "C")], _species("C"))
    )
    noisy = Structure(
        one_site.cell,
        [[F(1, 100000), F(0), F(0)]],
        one_site.species,
        one_site.species_at_sites,
    )
    assert conventional_cell(noisy, tolerance=1e-3).spacegroup.it_number == 221
    with pytest.raises(ValueError, match="Wyckoff position|not symmetric"):
        conventional_cell(noisy, tolerance=1e-8)


def test_plain_structure_path_forwards_limit_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("httk.atomistic.standardization")
    original_recognize = module.recognize_asu
    captured: dict[str, Any] = {}
    expanded = UnitcellStructureView(_rocksalt())
    plain = Structure(
        expanded.cell,
        expanded.sites,
        expanded.species,
        expanded.species_at_sites,
    )

    def capture_recognition(structure: StructureLike, **kwargs: Any) -> ASUStructure:
        captured.update(kwargs)
        return original_recognize(structure, **kwargs)

    monkeypatch.setattr(module, "recognize_asu", capture_recognition)
    conventional_cell(plain, tolerance=1e-3, limit_denominator=12)

    assert captured["tolerance"] == 1e-3
    assert captured["limit_denominator"] == 12


def test_asu_input_rejects_recognition_arguments() -> None:
    with pytest.raises(ValueError, match="existing ASU"):
        conventional_cell(_rocksalt(), tolerance=1e-3)
    with pytest.raises(ValueError, match="existing ASU"):
        conventional_cell(_rocksalt(), limit_denominator=12)


def test_non_three_dimensional_plain_input_is_refused_by_recognition() -> None:
    structure = Structure(
        Cell(CUBIC, periodicity=(True, True, False)),
        [[0, 0, 0]],
        _species("C"),
        ("C",),
    )

    with pytest.raises(
        ValueError, match="recognize_asu requires a fully 3D-periodic structure"
    ):
        conventional_cell(structure)
