"""Exact supercell construction and deterministic shape selection."""

import fractions

import pytest
from httk.core import FracVector, SurdScalar

from httk.atomistic import (
    ASUSite,
    ASUStructure,
    Cell,
    Sites,
    Species,
    Structure,
    SupercellResult,
    build_supercell,
    cubic_supercell,
    orthogonal_supercell,
)

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
    """Rejected at construction, so it can never reach a supercell in the first place.

    `Cell` requires a non-degenerate basis, and `Structure` funnels every cell through
    `Cell`, so the rejection happens at the point the bad geometry is written down rather
    than at the operation that later trips over it. `build_supercell` keeps its own
    nonsingular check as defence in depth for any future backend that does not go through
    `Cell`.
    """
    with pytest.raises(ValueError, match="non-degenerate"):
        _single_site([[1, 0, 0], [0, 1, 0], [1, 1, 0]])


def test_site_limit_is_checked_before_materialization() -> None:
    with pytest.raises(ValueError, match=r"12 sites .* max_sites=10"):
        build_supercell(
            _binary_structure(),
            [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
            max_sites=10,
        )

    assert len(
        build_supercell(
            _binary_structure(),
            [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
            max_sites=None,
        ).structure.sites
    ) == 12


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


def test_shape_search_uses_the_metric_not_the_cartesian_orientation() -> None:
    identity = _single_site([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rotated = _single_site([[F(3, 5), F(-4, 5), 0], [F(4, 5), F(3, 5), 0], [0, 0, 1]])

    assert cubic_supercell(identity, 8).transformation == cubic_supercell(rotated, 8).transformation


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
