"""Exact supercell construction and deterministic shape selection."""

import fractions

import pytest
from httk.core import FracVector, SurdScalar

from httk.atomistic import (
    Assembly,
    ASUSite,
    ASUStructure,
    Cell,
    CellParams,
    Sites,
    Species,
    Structure,
    SupercellResult,
    build_supercell,
    cubic_supercell,
    orthogonal_supercell,
)
from httk.atomistic.composition import ChemicalComposition

F = fractions.Fraction


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _binary_structure() -> Structure:
    return Structure(
        cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        sites=[[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]],
        species=_species("Na", "Cl"),
        species_at_sites=["Na", "Cl"],
    )


def _single_site(cell: object) -> Structure:
    return Structure(cell, [[0, 0, 0]], _species("Na"), ["Na"])


def test_a_diagonal_supercell_is_built_exactly_in_cell_major_order() -> None:
    result = build_supercell(_binary_structure(), [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

    assert isinstance(result, SupercellResult)
    assert result.multiplier == 6
    assert result.transformation == FracVector.create([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    assert result.structure.cell.basis == FracVector.create([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    assert len(result.structure.sites) == 12
    assert result.structure.species_at_sites == ("Na", "Cl") * 6
    assert result.structure.sites.reduced_coords == FracVector.create(
        [
            [0, 0, 0],
            [F(1, 2), F(1, 4), F(1, 6)],
            [0, 0, F(1, 3)],
            [F(1, 2), F(1, 4), F(1, 2)],
            [0, 0, F(2, 3)],
            [F(1, 2), F(1, 4), F(5, 6)],
            [0, F(1, 2), 0],
            [F(1, 2), F(3, 4), F(1, 6)],
            [0, F(1, 2), F(1, 3)],
            [F(1, 2), F(3, 4), F(1, 2)],
            [0, F(1, 2), F(2, 3)],
            [F(1, 2), F(3, 4), F(5, 6)],
        ]
    )
    assert result.orthogonality_score == 0


def test_shears_and_negative_determinants_are_supported() -> None:
    sheared = _binary_structure().supercell([[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    reflected = _binary_structure().supercell([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    assert sheared.multiplier == 2
    assert len(sheared.structure.sites) == 4
    assert sheared.structure.cell.basis == FracVector.create([[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    assert reflected.multiplier == 1
    assert reflected.structure.cell.basis == FracVector.create([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert reflected.structure.sites.reduced_coords == _binary_structure().sites.reduced_coords


@pytest.mark.parametrize(
    "transformation, message",
    [
        ([[1, 0], [0, 1]], "3x3"),
        ([[F(1, 2), 0, 0], [0, 1, 0], [0, 0, 1]], "integers"),
        ([[1, 0, 0], [1, 0, 0], [0, 0, 1]], "nonsingular"),
    ],
)
def test_invalid_transformations_are_rejected(transformation: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_supercell(_binary_structure(), transformation)


def test_a_singular_source_cell_is_rejected_clearly() -> None:
    """Lazy views accept bad geometry until an operation needs the basis.

    The geometry is accepted when written down, then rejected at the first geometric access.
    ``build_supercell`` keeps its own nonsingular check as defence in depth, although this
    deferred cell validation runs first for this source.
    """
    structure = _single_site([[1, 0, 0], [0, 1, 0], [1, 1, 0]])

    with pytest.raises(ValueError, match="non-degenerate"):
        _ = structure.cell.basis
    with pytest.raises(ValueError, match="non-degenerate"):
        build_supercell(structure, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_site_limit_is_checked_before_materialization() -> None:
    with pytest.raises(ValueError, match=r"12 sites .* max_sites=10"):
        build_supercell(
            _binary_structure(),
            [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
            max_sites=10,
        )

    assert (
        len(
            build_supercell(
                _binary_structure(),
                [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                max_sites=None,
            ).structure.sites
        )
        == 12
    )


def test_precision_bounds_are_transformed_conservatively() -> None:
    structure = Structure(
        cell=Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]], precision=F(1, 1000)),
        sites=Sites([[F(1, 3), F(1, 5), F(1, 7)]], precision=F(1, 10_000)),
        species=_species("Na"),
        species_at_sites=["Na"],
    )

    result = build_supercell(structure, [[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    assert result.structure.basis_precision == F(3, 1000)
    assert result.structure.coordinate_precision == F(3, 20_000)


def test_supercell_preserves_formula_annotations_and_remaps_assemblies() -> None:
    source = Structure(
        [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        [[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]],
        _species("Na", "Cl"),
        ["Na", "Cl"],
        molecular=True,
        assemblies=(Assembly(((0,), (1,)), (F(1, 2), F(1, 2))),),
        chemical_composition=ChemicalComposition({"H": 2}, mode="implicit"),
        chemical_formula_descriptive="ClH4Na",
        chemical_formula_hill="ClH4Na",
        optimization_type="local",
    )

    result = build_supercell(source, [[2, 0, 0], [0, 1, 0], [0, 0, 1]]).structure

    assert result.molecular
    assert result.assemblies is not None
    assert tuple(assembly.sites_in_groups for assembly in result.assemblies) == (
        ((0,), (1,)),
        ((2,), (3,)),
    )
    assert result.chemical_composition is not None
    assert result.chemical_composition.amount_mapping["H"] == 4
    assert result.chemical_formula_reduced == source.chemical_formula_reduced == "ClH4Na"
    assert result.chemical_formula_anonymous == source.chemical_formula_anonymous
    assert result.chemical_formula_descriptive == "ClH4Na"
    assert result.chemical_formula_hill == "ClH4Na"
    assert result.optimization_type == "local"


def test_non_simple_inputs_expand_to_a_plain_structure() -> None:
    no_parameters = FracVector.create(())
    asu = ASUStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        221,
        [ASUSite("a", no_parameters, "Na")],
        _species("Na"),
    )
    primitive = (
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        [[0, 0, 0]],
        [11],
    )

    from_asu = build_supercell(asu, [[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    from_primitive = build_supercell(primitive, [[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert type(from_asu.structure) is Structure
    assert type(from_primitive.structure) is Structure
    assert len(from_asu.structure.sites) == 2
    assert len(from_primitive.structure.sites) == 2


def test_cubic_search_finds_the_exact_two_by_two_by_two_cell() -> None:
    structure = _single_site([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = cubic_supercell(structure, 8)

    assert result.transformation == FracVector.create([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    assert result.multiplier == 8
    assert result.orthogonality_score == 0
    assert result.cubicity_score == 0
    assert isinstance(result.cubicity_score, SurdScalar)


@pytest.mark.extended
def test_shape_search_uses_the_metric_not_the_cartesian_orientation() -> None:
    identity = _single_site([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rotated = _single_site([[F(3, 5), F(-4, 5), 0], [F(4, 5), F(3, 5), 0], [0, 0, 1]])

    assert cubic_supercell(identity, 8).transformation == cubic_supercell(rotated, 8).transformation


@pytest.mark.extended
def test_orthogonal_search_recovers_the_tutorial_cells_exact_rectangle() -> None:
    structure = _single_site(
        [
            [F(1622, 400), 0, 0],
            [0, F(1622, 400), 0],
            [F(811, 400), F(811, 400), F(2524, 400)],
        ]
    )

    result = orthogonal_supercell(structure, 2)
    assert result.transformation == FracVector.create([[1, 0, 0], [0, 1, 0], [-1, -1, 2]])
    assert result.orthogonality_score == 0


@pytest.mark.extended
def test_orthogonal_tolerance_finds_the_smallest_exact_hexagonal_supercell() -> None:
    structure = _single_site(CellParams((1, 1, 3, 90, 90, 120)).basis)

    result = orthogonal_supercell(structure, tolerance=0)

    # The in-plane (1, 0), (1, 2) transform makes the 120-degree pair orthogonal.
    assert result.multiplier == 2
    assert result.transformation == FracVector.create([[1, 0, 0], [1, 2, 0], [0, 0, 1]])
    assert result.orthogonality_score.is_zero()


@pytest.mark.extended
def test_orthogonal_tolerance_stops_at_an_already_orthogonal_cell() -> None:
    structure = _single_site(CellParams((2, 2, 3, 90, 90, 90)).basis)

    result = structure.orthogonal_supercell(tolerance=0.01)

    assert result.multiplier == 1
    assert result.transformation == FracVector.create([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert result.orthogonality_score.is_zero()


@pytest.mark.extended
def test_cubic_tolerance_finds_the_smallest_exact_tetragonal_supercell() -> None:
    structure = _single_site(CellParams((2, 2, 1, 90, 90, 90)).basis)

    result = cubic_supercell(structure, tolerance=0, max_multiplier=2)

    # Two cells stacked along c turn c=1 into the same length as a=b=2.
    assert result.multiplier == 2
    assert result.transformation == FracVector.create([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
    assert result.cubicity_score.is_zero()


@pytest.mark.extended
def test_tolerance_failure_reports_the_best_exact_score() -> None:
    structure = _single_site(CellParams((2, 2, 1, 90, 90, 90)).basis)

    with pytest.raises(ValueError, match=r"best was \(multiplier=1, cubicity_score="):
        cubic_supercell(structure, tolerance=0, max_multiplier=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"multiplier": 1, "tolerance": 0},
    ],
)
def test_tolerance_and_multiplier_are_mutually_exclusive(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="exactly one of multiplier or tolerance"):
        orthogonal_supercell(_binary_structure(), **kwargs)


def test_negative_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        orthogonal_supercell(_binary_structure(), tolerance=-1)


@pytest.mark.extended
def test_tolerance_float_uses_its_decimal_spelling() -> None:
    structure = _single_site(CellParams((2, 2, 3, 90, 90, 90)).basis)

    result = orthogonal_supercell(structure, tolerance=0.01, max_multiplier=1)

    assert result.multiplier == 1


@pytest.mark.extended
def test_tolerance_mode_respects_max_sites() -> None:
    structure = _single_site(CellParams((2, 2, 1, 90, 90, 90)).basis)

    with pytest.raises(ValueError, match="max_sites=1"):
        cubic_supercell(structure, tolerance=0, max_multiplier=2, max_sites=1)


def test_radius_zero_still_has_a_determinant_matching_fallback() -> None:
    result = cubic_supercell(
        _single_site([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        6,
        search_radius=0,
    )
    assert result.multiplier == 6


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_search_multiplier_must_be_a_positive_integer(bad: object) -> None:
    with pytest.raises(ValueError, match="multiplier"):
        orthogonal_supercell(_binary_structure(), bad)


@pytest.mark.parametrize("bad", [-1, 3, True, 1.5])
def test_search_radius_is_bounded(bad: object) -> None:
    with pytest.raises(ValueError, match="search_radius"):
        cubic_supercell(_binary_structure(), 1, search_radius=bad)
