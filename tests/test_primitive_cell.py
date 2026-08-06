"""Tests for the fixed spglib-convention primitive-cell operation."""

import fractions

import pytest
from httk.core import FracVector, SurdVector

from httk.atomistic import (
    Assembly,
    ASUStructure,
    CartesianSiteMoments,
    Cell,
    ChemicalComposition,
    Spacegroup,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    build_supercell,
    conventional_cell,
    primitive_cell,
    recognize_asu,
    same_crystal,
)

F = fractions.Fraction
NO_PARAMETERS = FracVector.create(())


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _hexagonal_basis() -> SurdVector:
    zero = SurdVector.create(0)._as_scalar()
    two = SurdVector.create(2)._as_scalar()
    four = SurdVector.create(4)._as_scalar()
    twelve = SurdVector.create(12)._as_scalar()
    root_three = SurdVector.sqrt_of(3)
    return SurdVector._from_scalar_grid(
        [
            [four, zero, zero],
            [-two, root_three * two, zero],
            [zero, zero, twelve],
        ],
        (3, 3),
    )


def _fixture(it_number: int, *, two_species: bool = False) -> ASUStructure:
    basis = (
        _hexagonal_basis()
        if it_number == 166
        else ([[5, 0, 0], [0, 6, 0], [2, 0, 7]] if it_number in (12, 38) else [[5, 0, 0], [0, 5, 0], [0, 0, 5]])
    )
    free_params = FracVector.create(["1/4"]) if it_number == 38 else NO_PARAMETERS
    sites = [WyckoffSite("a", free_params, "Na" if two_species else "C")]
    species = _species("Na", "Cl") if two_species else _species("C")
    if two_species:
        sites.append(WyckoffSite("b", NO_PARAMETERS, "Cl"))
    return ASUStructure(basis, it_number, sites, species)


def _site_multiset(structure: UnitcellStructure) -> set[tuple[str, tuple[fractions.Fraction, ...]]]:
    return {
        (species, tuple(coordinate.normalize().to_fractions()))
        for species, coordinate in zip(
            structure.species_at_sites,
            structure.sites.reduced_coords,
            strict=True,
        )
    }


def _determinant(matrix: list[list[float]]) -> float:
    return (
        matrix[0][0] * matrix[1][1] * matrix[2][2]
        + matrix[0][1] * matrix[1][2] * matrix[2][0]
        + matrix[0][2] * matrix[1][0] * matrix[2][1]
        - matrix[0][2] * matrix[1][1] * matrix[2][0]
        - matrix[0][1] * matrix[1][0] * matrix[2][2]
        - matrix[0][0] * matrix[1][2] * matrix[2][1]
    )


EXPECTED_ROW_TRANSFORMS = {
    "P": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "A": [[1, 0, 0], [0, F(1, 2), F(1, 2)], [0, F(-1, 2), F(1, 2)]],
    "C": [[F(1, 2), F(-1, 2), 0], [F(1, 2), F(1, 2), 0], [0, 0, 1]],
    "R": [[F(2, 3), F(1, 3), F(1, 3)], [F(-1, 3), F(1, 3), F(1, 3)], [F(-1, 3), F(-2, 3), F(1, 3)]],
    "I": [[F(-1, 2), F(1, 2), F(1, 2)], [F(1, 2), F(-1, 2), F(1, 2)], [F(1, 2), F(1, 2), F(-1, 2)]],
    "F": [[0, F(1, 2), F(1, 2)], [F(1, 2), 0, F(1, 2)], [F(1, 2), F(1, 2), 0]],
}


@pytest.mark.parametrize("it_number", [221, 229, 225, 12, 38, 166])
def test_vendored_transform_volume_and_exact_centering_collapse(it_number: int) -> None:
    result = primitive_cell(_fixture(it_number))
    centring = result.spacegroup.centring_type
    expected = FracVector.create(EXPECTED_ROW_TRANSFORMS[centring])
    n = len(result.spacegroup.centering_translations)

    assert result.transform == expected
    assert result.transform.det().to_fraction() == F(1, n)
    assert result.structure.cell.volume * n == result.conventional.structure.cell.volume
    assert len(result.conventional.structure.sites) == n * len(result.structure.sites)
    for translation in result.spacegroup.centering_translations:
        mapped = translation * result.transform.inv()
        assert all(value.denominator == 1 for value in mapped.to_fractions())


@pytest.mark.parametrize("it_number", [221, 229, 225, 12, 38, 166])
def test_inverse_transform_round_trips_to_the_conventional_cell(it_number: int) -> None:
    result = primitive_cell(_fixture(it_number))
    inverse = result.transform.inv().simplify()

    assert inverse.denom == 1
    assert inverse.det().to_fraction() == len(result.spacegroup.centering_translations)
    restored = build_supercell(result.structure, inverse)
    assert same_crystal(restored.structure, result.conventional.structure)


def test_determinism_and_site_order_independence() -> None:
    first = primitive_cell(_fixture(225, two_species=True))
    second = primitive_cell(_fixture(225, two_species=True))
    shuffled = ASUStructure(
        _fixture(225, two_species=True).cell,
        225,
        [WyckoffSite("b", NO_PARAMETERS, "Cl"), WyckoffSite("a", NO_PARAMETERS, "Na")],
        _species("Na", "Cl"),
    )
    third = primitive_cell(shuffled)

    assert first.structure.cell.basis == second.structure.cell.basis == third.structure.cell.basis
    assert _site_multiset(first.structure) == _site_multiset(second.structure) == _site_multiset(third.structure)


def test_existing_asu_rejects_recognition_arguments() -> None:
    asu = _fixture(221)
    with pytest.raises(ValueError, match=r"primitive_cell\(\).*existing ASU"):
        primitive_cell(asu, tolerance=1e-3)
    with pytest.raises(ValueError, match=r"primitive_cell\(\).*existing ASU"):
        primitive_cell(asu, limit_denominator=12)


def test_assemblies_and_site_moments_are_refused() -> None:
    asu = ASUStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        221,
        [WyckoffSite("a", NO_PARAMETERS, "C")],
        _species("C"),
        assemblies=(Assembly(((0,),), (1,)),),
    )
    with pytest.raises(ValueError, match="correlated site groups.*primitive cell"):
        primitive_cell(asu)

    moment_asu = ASUStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        221,
        [WyckoffSite("a", NO_PARAMETERS, "C", moment=CartesianSiteMoments([[1, 0, 0]]))],
        _species("C"),
    )
    with pytest.raises(ValueError, match="site moments; keep the original setting"):
        primitive_cell(moment_asu)

    full = UnitcellStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        [[0, 0, 0]],
        _species("C"),
        ["C"],
        assemblies=(Assembly(((0,),), (1,)),),
    )
    with pytest.raises(ValueError, match="correlated site groups.*primitive cell"):
        primitive_cell(UnitcellStructureView(full))


def test_charge_composition_and_precision_scale() -> None:
    source = ASUStructure(
        Cell([[5, 0, 0], [0, 5, 0], [0, 0, 5]], precision=F(1, 100)),
        229,
        [WyckoffSite("a", NO_PARAMETERS, "C")],
        _species("C"),
        coordinate_precision=F(1, 1000),
        chemical_composition=ChemicalComposition({"C": 2}, mode="implicit"),
        charge=4,
    )
    result = primitive_cell(source)

    assert result.multiplier == F(1, 2)
    assert result.structure.charge == 2
    assert result.structure.chemical_composition is not None
    assert result.structure.chemical_composition.amount_mapping["C"] == 1
    assert result.structure.cell.precision is not None
    assert result.structure.coordinate_precision is not None


def test_plain_unitcell_recognition_path_handles_fcc() -> None:
    expanded = UnitcellStructureView(_fixture(225))
    plain = UnitcellStructure(expanded.cell, expanded.sites, expanded.species, expanded.species_at_sites)

    result = primitive_cell(plain)

    assert result.spacegroup.it_number == 225
    assert result.spacegroup.centring_type == "F"
    assert len(result.structure.sites) == 1


def test_spglib_recognizes_plain_rhombohedral_primitive_input() -> None:
    pytest.importorskip("spglib")
    rhombohedral = Spacegroup.for_setting("166:R")
    transform = rhombohedral.transform_from_standard
    original = ASUStructure(
        transform.basis_to_setting(_hexagonal_basis()),
        166,
        [WyckoffSite("a", NO_PARAMETERS, "C")],
        _species("C"),
        transform=transform,
    )
    expanded = UnitcellStructureView(original)
    plain = UnitcellStructure(expanded.cell, expanded.sites, expanded.species, expanded.species_at_sites)

    recognized = recognize_asu(plain, tolerance=1e-5)
    result = primitive_cell(plain, tolerance=1e-5)

    assert abs(recognized.transform.determinant()) == 3
    assert same_crystal(plain, UnitcellStructureView(recognized))
    assert len(result.structure.sites) == 1
    assert result.multiplier == F(1)


def test_plain_supercell_recognition_preserves_the_lattice_index() -> None:
    primitive = primitive_cell(_fixture(225)).structure
    built = build_supercell(primitive, [[2, 0, 0], [0, 2, 0], [0, 0, 2]]).structure
    supercell = UnitcellStructure(
        built.cell,
        built.sites,
        built.species,
        built.species_at_sites,
        chemical_composition=ChemicalComposition({"C": 8}, mode="implicit"),
        charge=8,
    )

    result = primitive_cell(supercell)
    conventional = conventional_cell(supercell)

    assert len(supercell.sites) == 8
    assert len(result.structure.sites) == 1
    assert result.multiplier == F(1, 8)
    assert result.structure.charge == 1
    assert result.structure.chemical_composition is not None
    assert result.structure.chemical_composition.amount_mapping["C"] == 1
    assert len(conventional.structure.sites) == 4


def test_spglib_agrees_on_fixture_volume_and_site_count() -> None:
    spglib = pytest.importorskip("spglib")
    for it_number in [221, 229, 225, 12, 38, 166]:
        result = primitive_cell(_fixture(it_number))
        view = UnitcellStructureView(result.conventional.structure)
        number_by_species = {name: index + 1 for index, name in enumerate(sorted(set(view.species_at_sites)))}
        numbers = [number_by_species[name] for name in view.species_at_sites]
        primitive = spglib.find_primitive(
            (view.cell.basis.to_floats(), view.sites.reduced_coords.to_floats(), numbers),
            symprec=1e-5,
        )
        assert primitive is not None
        assert len(primitive[1]) == len(result.structure.sites)
        primitive_volume = abs(_determinant(primitive[0]))
        expected_volume = float(result.structure.cell.volume)
        assert abs(primitive_volume - expected_volume) < 1e-9 * expected_volume
