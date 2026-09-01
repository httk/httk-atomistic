"""Tests for exact conversion to IT standard-setting conventional cells."""

import fractions
import importlib
from typing import Any

import pytest
from httk.core import FracVector, SurdVector

from httk.atomistic import (
    Assembly,
    ASUStructure,
    ASUStructureView,
    CartesianSiteMoments,
    Cell,
    SettingTransform,
    Spacegroup,
    Species,
    StructureLike,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    conventional_cell,
    recognize_asu,
    same_crystal,
)
from httk.atomistic.composition import ChemicalComposition
from httk.atomistic.models.sites.sites import Sites

F = fractions.Fraction
NO_PARAMETERS = FracVector(())
CUBIC = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
ORTHO = [[5, 0, 0], [0, 6, 0], [0, 0, 7]]


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _rocksalt() -> ASUStructure:
    return ASUStructure(
        CUBIC,
        225,
        [WyckoffSite("a", NO_PARAMETERS, "Na"), WyckoffSite("b", NO_PARAMETERS, "Cl")],
        _species("Na", "Cl"),
    )


def _monoclinic() -> tuple[ASUStructure, SettingTransform]:
    transform = Spacegroup.from_setting("15:c1").transform_from_standard
    return (
        ASUStructure(
            ORTHO,
            15,
            [WyckoffSite("e", FracVector(["1/3"]), "Si")],
            _species("Si"),
            transform=transform,
        ),
        transform,
    )


def _hexagonal_basis_pair() -> tuple[SurdVector, SurdVector]:
    """Literal own/standard bases for the SG 166 rhombohedral setting."""
    zero = SurdVector(0)._as_scalar()
    two = SurdVector(2)._as_scalar()
    four = SurdVector(4)._as_scalar()
    minus_two = SurdVector(-2)._as_scalar()
    twelve = SurdVector(12)._as_scalar()
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


def test_standardization_preserves_chemical_annotations() -> None:
    asu = ASUStructure(
        CUBIC,
        225,
        [WyckoffSite("a", NO_PARAMETERS, "Na"), WyckoffSite("b", NO_PARAMETERS, "Cl")],
        _species("Na", "Cl"),
        chemical_composition=ChemicalComposition({"H": 2}, mode="implicit"),
        chemical_formula_descriptive="Cl2HNa2",
        chemical_formula_hill="Cl2HNa2",
        optimization_type="experimental",
    )

    transformed = conventional_cell(asu)
    result = transformed.structure

    assert result.chemical_formula_reduced == asu.chemical_formula_reduced == "Cl2HNa2"
    assert result.chemical_formula_anonymous == asu.chemical_formula_anonymous
    assert result.chemical_formula_descriptive == "Cl2HNa2"
    assert result.chemical_formula_hill == "Cl2HNa2"
    assert result.optimization_type == "experimental"
    assert transformed.asu.chemical_formula_reduced == "Cl2HNa2"
    assert transformed.asu.chemical_formula_hill == "Cl2HNa2"


def test_repeating_a_result_uses_its_unwrapped_standard_asu_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("httk.atomistic.symmetry.standardization")
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
    assert conventional_cell(asu).asu == conventional_cell(asu).asu


def test_a_nonstandard_setting_is_mapped_back_to_the_standard_cell() -> None:
    asu, transform = _monoclinic()
    original = UnitcellStructureView(asu)

    result = conventional_cell(asu)
    mapped = UnitcellStructure(
        result.structure.cell,
        [transform.to_standard(row).normalize() for row in original.sites.reduced_coords],
        original.species,
        original.species_at_sites,
    )

    assert result.spacegroup == Spacegroup.standard(15)
    assert result.asu.transform.is_identity()
    assert result.structure.cell.basis == transform.basis_to_standard(asu.cell.basis)
    assert len(result.structure.sites) == len(original.sites)
    assert sorted(result.structure.species_at_sites) == sorted(original.species_at_sites)
    assert same_crystal(mapped, result.structure)
    recognized = recognize_asu(result.structure, setting=result.spacegroup)
    assert recognized.spacegroup.it_number == 15
    assert recognized.is_standard_setting


def test_rhombohedral_setting_expands_to_three_standard_cell_sites() -> None:
    transform = Spacegroup.from_setting("166:R").transform_from_standard
    rhombohedral_basis, expected_basis = _hexagonal_basis_pair()
    asu = ASUStructure(
        rhombohedral_basis,
        166,
        [WyckoffSite("a", NO_PARAMETERS, "Bi")],
        _species("Bi"),
        transform=transform,
        chemical_composition=ChemicalComposition({"H": 1}, mode="implicit"),
        chemical_formula_hill="BiH",
    )

    result = conventional_cell(asu)

    assert len(UnitcellStructureView(asu).sites) == 1
    assert len(result.structure.sites) == 3
    assert result.structure.cell.basis == expected_basis
    assert {tuple(row) for row in result.structure.sites.reduced_coords.to_fractions()} == {
        (F(0), F(0), F(0)),
        (F(1, 3), F(2, 3), F(2, 3)),
        (F(2, 3), F(1, 3), F(1, 3)),
    }
    assert result.multiplier == F(3)
    assert result.structure.chemical_composition is not None
    assert result.structure.chemical_composition.amount_mapping["H"] == 3
    assert result.structure.chemical_formula_reduced == "BiH"
    assert result.structure.chemical_formula_hill == "BiH"


def test_standardization_remaps_assemblies_only_for_an_exact_bijection() -> None:
    carbon = _species("C")
    exact = UnitcellStructure(
        CUBIC,
        [[0, 0, 0]],
        carbon,
        ["C"],
        assemblies=(Assembly(((0,),), (1,)),),
    )
    result = conventional_cell(exact).structure
    assert result.assemblies is not None
    assert result.assemblies[0].sites_in_groups == ((0,),)

    noisy = UnitcellStructure(
        CUBIC,
        [[F(1, 100_000), 0, 0]],
        carbon,
        ["C"],
        assemblies=(Assembly(((0,),), (1,)),),
    )
    with pytest.raises(ValueError, match="exact site bijection"):
        conventional_cell(noisy, tolerance=1e-3)


def test_precision_is_scaled_by_the_exact_induced_matrix_norms() -> None:
    transform = SettingTransform([[1, 2, 0], [0, 1, 1], [0, 0, 1]])
    asu = ASUStructure(
        Cell(
            [[5, 0, 0], [-10, 5, 0], [10, -5, 5]],
            precision=F(1, 50),
        ),
        221,
        [WyckoffSite("a", NO_PARAMETERS, "C")],
        _species("C"),
        transform=transform,
        coordinate_precision=F(1, 1000),
    )

    result = conventional_cell(asu)

    assert result.structure.cell.basis == SurdVector([[5, 0, 0], [0, 5, 0], [0, 0, 5]])
    assert result.structure.cell.precision == F(3, 50)
    assert result.asu.coordinate_precision == F(1, 200)


def test_an_untabulated_half_determinant_transform_can_have_a_subunit_multiplier() -> None:
    transform = SettingTransform([[F(1, 2), 0, 0], [0, 1, 0], [0, 0, 1]])
    asu = ASUStructure(
        [[10, 0, 0], [0, 5, 0], [0, 0, 5]],
        221,
        [WyckoffSite("a", NO_PARAMETERS, "C")],
        _species("C"),
        transform=transform,
    )

    result = conventional_cell(asu)

    assert result.multiplier == F(1, 2)


def test_plain_structure_path_matches_recognized_asu_path_and_forwards_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expanded = UnitcellStructureView(_rocksalt())
    plain = UnitcellStructure(
        expanded.cell,
        expanded.sites,
        expanded.species,
        expanded.species_at_sites,
    )
    direct = conventional_cell(plain)
    expected = conventional_cell(recognize_asu(plain))
    assert direct.structure == expected.structure

    one_site = UnitcellStructureView(ASUStructure(CUBIC, 221, [WyckoffSite("a", NO_PARAMETERS, "C")], _species("C")))
    noisy = UnitcellStructure(
        one_site.cell,
        [[F(1, 100000), F(0), F(0)]],
        one_site.species,
        one_site.species_at_sites,
    )
    assert conventional_cell(noisy, tolerance=1e-3).spacegroup.it_number == 221

    # A common translation of a one-site crystal is an origin choice, not structural noise. Verify
    # the tight tolerance is forwarded explicitly instead of relying on the old, incorrect behavior
    # that rounded spglib's data-derived origin onto a small-fraction grid and rejected this cell.
    module = importlib.import_module("httk.atomistic.symmetry.standardization")
    original_recognize = module.recognize_asu
    captured: dict[str, Any] = {}

    def capture_recognition(structure: StructureLike, **kwargs: Any) -> ASUStructure:
        captured.update(kwargs)
        return original_recognize(structure, **kwargs)

    monkeypatch.setattr(module, "recognize_asu", capture_recognition)
    assert conventional_cell(noisy, tolerance=1e-8).spacegroup.it_number == 221
    assert captured["tolerance"] == 1e-8


def test_plain_structure_path_forwards_limit_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("httk.atomistic.symmetry.standardization")
    original_recognize = module.recognize_asu
    captured: dict[str, Any] = {}
    expanded = UnitcellStructureView(_rocksalt())
    plain = UnitcellStructure(
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
    structure = UnitcellStructure(
        Cell(CUBIC, periodicity=(True, True, False)),
        [[0, 0, 0]],
        _species("C"),
        ("C",),
    )

    with pytest.raises(ValueError, match="recognize_asu requires a fully 3D-periodic structure"):
        conventional_cell(structure)


def _rutile_altermagnet(perturbation: F | None = None) -> UnitcellStructure:
    # Rutile-type P4_2/mnm (SG 136): two metals on the single 2a orbit carry opposite z moments;
    # the four O sit on 4f. ``perturbation`` displaces every coordinate to model noisy CONTCAR floats.
    u = F(61, 200)
    coordinates = [
        [F(0), F(0), F(0)],
        [F(1, 2), F(1, 2), F(1, 2)],
        [u, u, F(0)],
        [-u, -u, F(0)],
        [u + F(1, 2), F(1, 2) - u, F(1, 2)],
        [F(1, 2) - u, u + F(1, 2), F(1, 2)],
    ]
    if perturbation is not None:
        coordinates = [[value + perturbation for value in row] for row in coordinates]
    moments = CartesianSiteMoments([[0, 0, 3], [0, 0, -3], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    return UnitcellStructure(
        Cell([[F(23, 5), 0, 0], [0, F(23, 5), 0], [0, 0, F(74, 25)]], precision=F(1, 10000)),
        Sites(coordinates, precision=F(1, 10000)),
        None,
        ["Ru", "Ru", "O", "O", "O", "O"],
        site_moments=moments,
    )


def _metal_z_moments(structure: UnitcellStructure) -> list[float]:
    moments = structure.site_moments
    assert isinstance(moments, CartesianSiteMoments)
    grid = moments.cartesian_moments
    return sorted(
        float(grid._element((index, 2)).to_float())
        for index, name in enumerate(structure.species_at_sites)
        if name == "Ru"
    )


def test_altermagnet_moments_are_carried_to_the_conventional_cell() -> None:
    result = conventional_cell(_rutile_altermagnet())
    assert result.spacegroup.it_number == 136
    # The nuclear ASU stays moment-free; the opposite moments ride on the expanded structure only.
    assert all(site.moment is None for site in result.asu.wyckoff_sites)
    assert _metal_z_moments(result.structure) == [-3.0, 3.0]


def test_noisy_float_positions_still_match_every_site() -> None:
    result = conventional_cell(_rutile_altermagnet(perturbation=F(1, 10000)))
    assert result.spacegroup.it_number == 136
    assert _metal_z_moments(result.structure) == [-3.0, 3.0]


def test_cartesian_moments_are_unchanged_by_a_non_standard_setting_change() -> None:
    # Frame-invariance pin (retired ponytail refusal): a setting change recombines the basis and
    # shifts the origin without rotating the Cartesian frame, so a Cartesian moment is unchanged.
    # This input is in setting 15:c1 and recognition maps it back through a non-identity transform.
    transform = Spacegroup.from_setting("15:c1").transform_from_standard
    asu = ASUStructure(
        ORTHO,
        15,
        [WyckoffSite("e", FracVector(["1/3"]), "Si", moment=CartesianSiteMoments([[0, 0, "5/2"]]))],
        _species("Si"),
        transform=transform,
    )
    view = UnitcellStructureView(asu)
    plain = UnitcellStructure(
        view.cell,
        view.sites,
        view.species,
        view.species_at_sites,
        site_moments=CartesianSiteMoments([[0, 0, "5/2"]] * len(view.sites)),
    )
    result = conventional_cell(plain)

    assert not result.transform.is_identity()
    moments = result.structure.site_moments
    assert isinstance(moments, CartesianSiteMoments)
    rows = {
        tuple(float(moments.cartesian_moments._element((index, column)).to_float()) for column in range(3))
        for index in range(len(result.structure.sites))
    }
    assert rows == {(0.0, 0.0, 2.5)}


def _cubic_c_doubling(first_z: int, second_z: int) -> UnitcellStructure:
    # A 1x1x2 doubling of a one-atom simple-cubic cell: two P-cubic sites along c carrying the
    # given z moments. Nuclear recognition finds the one-atom cell (multiplier 1/2).
    return UnitcellStructure(
        Cell([[F(3), 0, 0], [0, F(3), 0], [0, 0, F(6)]], precision=F(1, 10000)),
        Sites([[F(0), F(0), F(0)], [F(0), F(0), F(1, 2)]], precision=F(1, 10000)),
        None,
        ["Fe", "Fe"],
        site_moments=CartesianSiteMoments([[0, 0, first_z], [0, 0, second_z]]),
    )


def test_antiferromagnetic_supercell_is_refused_not_silently_collapsed() -> None:
    # The magnetic cell is larger than the nuclear cell: collapsing would drop the opposite moment
    # and return a ferromagnet. Coverage of every input site must catch this and refuse.
    with pytest.raises(ValueError, match="magnetic order incompatible with the primitive cell"):
        conventional_cell(_cubic_c_doubling(2, -2))


def test_ferromagnetic_supercell_folds_and_keeps_the_moment() -> None:
    result = conventional_cell(_cubic_c_doubling(2, 2))
    assert result.multiplier == F(1, 2)
    assert len(result.structure.sites) == 1
    moments = result.structure.site_moments
    assert isinstance(moments, CartesianSiteMoments)
    assert float(moments.cartesian_moments._element((0, 2)).to_float()) == 2.0
