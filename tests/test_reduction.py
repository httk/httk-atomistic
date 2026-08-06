"""Tests for exact Niggli cell and structure reduction."""

import random
from fractions import Fraction as F

import pytest
from httk.core import FracVector, SurdVector

from httk.atomistic import (
    Assembly,
    CartesianSiteMoments,
    Cell,
    CellParams,
    ChemicalComposition,
    CrystalAxisSiteMoments,
    Sites,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    build_supercell,
    is_niggli_reduced,
    niggli_reduce,
    niggli_reduced,
    same_crystal,
)
from httk.atomistic.reduction import _is_niggli_parameters


def _metric_fractions(cell: Cell) -> FracVector:
    metric = cell.metric()
    assert metric.is_rational
    return metric.coefficient(1)


def _identity() -> FracVector:
    return FracVector.create([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def _fixture_cells() -> dict[str, Cell]:
    return {
        "cubic": Cell([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        "tetragonal": Cell([[2, 0, 0], [0, 2, 0], [0, 0, 3]]),
        "orthorhombic": Cell([[2, 0, 0], [0, 3, 0], [0, 0, 4]]),
        "hexagonal": Cell(CellParams((2, 2, 3, 90, 90, 120)).basis),
        "rhombohedral": Cell(CellParams((2, 2, 2, 60, 60, 60)).basis),
        "monoclinic": Cell([[2, 0, 0], [1, 3, 0], [0, 0, 4]]),
        "triclinic": Cell([[2, 0, 0], [1, 3, 0], [1, 1, 4]]),
        "equal-edge-tie": Cell([[1, 0, 0], [0, 1, 0], [F(-1, 2), F(-1, 2), 1]]),
        "xi-boundary": Cell([[1, 0, 0], [0, 2, 0], [0, -1, 2]]),
    }


@pytest.mark.parametrize("name", tuple(_fixture_cells()))
def test_reduction_is_exact_and_has_an_integral_orientation_preserving_transform(name: str) -> None:
    source = _fixture_cells()[name]
    result = niggli_reduce(source)

    assert result.transform.denom == 1
    assert result.transform.det().to_fraction() == F(1)
    expected_metric = result.transform * _metric_fractions(source) * result.transform.T()
    assert _metric_fractions(result.cell) == expected_metric
    assert tuple(result.parameters) == (
        expected_metric.to_fractions()[0][0],
        expected_metric.to_fractions()[1][1],
        expected_metric.to_fractions()[2][2],
        2 * expected_metric.to_fractions()[1][2],
        2 * expected_metric.to_fractions()[0][2],
        2 * expected_metric.to_fractions()[0][1],
    )
    assert is_niggli_reduced(result.cell)


@pytest.mark.parametrize("name", tuple(_fixture_cells()))
def test_reduction_is_idempotent(name: str) -> None:
    first = niggli_reduce(_fixture_cells()[name])
    second = niggli_reduce(first.cell)

    assert second.transform == _identity()
    assert second.cell == first.cell
    assert second.parameters == first.parameters


def test_niggli_predicate_checks_boundary_conditions_exactly() -> None:
    assert is_niggli_reduced(_fixture_cells()["xi-boundary"])
    assert is_niggli_reduced(_fixture_cells()["equal-edge-tie"])
    assert not is_niggli_reduced(Cell([[2, 0, 0], [0, 1, 0], [0, 0, 3]]))

    # Here A=1, B=5, xi=B, eta=0, and zeta=2, so the xi=B special condition fails.
    assert not is_niggli_reduced(Cell([[1, 0, 0], [1, 2, 0], [0, F(5, 4), 3]]))


def test_niggli_parameter_predicate_rejects_algorithmic_tie_and_sign_counterexamples() -> None:
    assert not _is_niggli_parameters((F(1), F(1), F(2), F(-1), F(0), F(0)))
    assert _is_niggli_parameters((F(1), F(1), F(2), F(-1), F(-1), F(0)))
    assert not _is_niggli_parameters((F(1), F(2), F(3), F(1), F(0), F(1)))
    assert _is_niggli_parameters((F(1), F(2), F(3), F(-1), F(0), F(-1)))


def test_cell_level_tie_counterexample_is_reduced_before_predicate_acceptance() -> None:
    source = Cell([[1, 0, 0], [0, 1, 0], [0, F(-1, 2), 1]])
    assert is_niggli_reduced(source) is False

    result = niggli_reduce(source)
    assert is_niggli_reduced(result.cell)


def test_cell_parameter_backend_is_accepted_as_a_cell_like_input() -> None:
    result = niggli_reduce(CellParams((2, 2, 3, 90, 90, 120)))
    assert is_niggli_reduced(result.cell)


@pytest.mark.parametrize(
    "transform",
    [
        [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 1], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [1, -1, 1]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    ],
)
def test_unimodular_premixes_reduce_to_the_same_metric(transform: list[list[int]]) -> None:
    source = _fixture_cells()["triclinic"]
    premix = FracVector.create(transform)
    assert premix.det().to_fraction() == F(1)

    expected = _metric_fractions(niggli_reduce(source).cell)
    actual = _metric_fractions(niggli_reduce(Cell(SurdVector.create(premix) * source.basis)).cell)
    assert actual == expected


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _hand_structure(*, precision: bool = False) -> UnitcellStructure:
    return UnitcellStructure(
        Cell(
            [[1, 0, 0], [0, 1, 0], [1, 0, 1]],
            precision=F(1, 1000) if precision else None,
        ),
        Sites(
            [[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]],
            precision=F(1, 10_000) if precision else None,
        ),
        _species("Na", "Cl"),
        ["Na", "Cl"],
    )


def _round_trip(source: UnitcellStructure, result: object) -> None:
    reduced = result.structure  # type: ignore[attr-defined]
    transform = result.transform  # type: ignore[attr-defined]
    inverse = transform.inv().simplify()
    assert inverse.denom == 1
    assert inverse.det().to_fraction() == F(1)
    restored = build_supercell(reduced, inverse)
    assert same_crystal(UnitcellStructureView(source), UnitcellStructureView(restored.structure))


def test_hand_fixture_remaps_coordinates_using_the_actual_final_transform() -> None:
    source = _hand_structure()
    result = niggli_reduced(source)
    inverse = result.transform.inv().simplify()

    expected = (source.sites[1] * inverse).normalize()
    assert result.structure.sites[1] == expected
    assert result.structure.sites[1] == FracVector.create([0, F(1, 2), F(1, 2)])
    assert _metric_fractions(result.cell) == FracVector.create([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    _round_trip(source, result)


def test_structure_reduction_preserves_annotations_moments_charge_and_precision() -> None:
    source = UnitcellStructure(
        Cell([[1, 0, 0], [0, 1, 0], [1, 0, 1]], precision=F(1, 1000)),
        Sites([[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]], precision=F(1, 10_000)),
        _species("Na", "Cl"),
        ["Na", "Cl"],
        site_moments=CartesianSiteMoments([[1, 2, 3], [F(1, 2), 0, -1]]),
        molecular=True,
        assemblies=(Assembly(((0,), (1,)), (F(1, 2), F(1, 2))),),
        chemical_composition=ChemicalComposition({"H": 2}),
        chemical_formula_descriptive="ClH4Na",
        chemical_formula_hill="ClH4Na",
        optimization_type="local",
        charge=F(2),
    )

    result = niggli_reduced(source)
    assert result.cell is result.structure.cell
    assert result.structure.species == source.species
    assert result.structure.species_at_sites == source.species_at_sites
    assert result.structure.charge == source.charge
    assert result.structure.chemical_composition == source.chemical_composition
    assert result.structure.assemblies == source.assemblies
    assert result.structure.site_moments == source.site_moments
    assert result.structure.molecular == source.molecular
    assert result.structure.basis_precision == F(1, 500)
    assert result.structure.coordinate_precision == F(1, 5000)
    assert result.cell.volume == source.cell.volume
    assert len(result.structure.sites) == len(source.sites)
    _round_trip(source, result)


def test_niggli_reduced_widens_crystal_axis_moment_precision() -> None:
    cell = Cell(CellParams((2, 2, 3, 90, 90, 120)).basis)
    moment_precision = F(1, 1000)
    moments = CrystalAxisSiteMoments([[1, 0, 0]], cell, precision=moment_precision)
    source = UnitcellStructure(
        cell,
        [[0, 0, 0]],
        _species("C"),
        ["C"],
        site_moments=moments,
    )

    result = niggli_reduced(source)

    assert result.structure.site_moments is not None
    assert result.structure.site_moments.kind == "cartesian"
    assert result.structure.site_moments.precision == 3 * moment_precision
    assert result.structure.site_moments.cartesian_moments == moments.cartesian_moments


def test_niggli_reduce_preserves_cell_scale_factoring() -> None:
    source = Cell([[1, 0, 0], [0, 1, 0], [1, 0, 1]], scale=F(7, 3))

    result = niggli_reduce(source)

    assert result.cell.scale == source.scale
    assert result.cell.basis == SurdVector.create(result.transform) * source.basis
    assert result.cell.unscaled_basis == (SurdVector.create(result.transform) * source.unscaled_basis)


def test_reduction_refuses_non_three_dimensional_periodicity() -> None:
    cell = Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]], periodicity=(True, True, False))
    with pytest.raises(ValueError, match="fully 3D-periodic"):
        niggli_reduce(cell)
    with pytest.raises(ValueError, match="fully 3D-periodic"):
        niggli_reduced(UnitcellStructure(cell, [[0, 0, 0]], _species("C"), ["C"]))


def _float_gram(basis: list[list[float]]) -> list[list[float]]:
    return [[sum(row[index] * other[index] for index in range(3)) for other in basis] for row in basis]


def test_spglib_niggli_cross_check_full_gram_on_fixtures_and_seeded_random_bases() -> None:
    spglib = pytest.importorskip("spglib")

    sources = list(_fixture_cells().values())
    rng = random.Random(20260806)
    while len(sources) < len(_fixture_cells()) + 100:
        candidate = [[rng.randint(-5, 5) for _ in range(3)] for _ in range(3)]
        if abs(FracVector.create(candidate).det().to_fraction()) <= F(1, 2):
            continue
        sources.append(Cell(candidate))

    for source in sources:
        result = niggli_reduce(source)
        reduced = spglib.niggli_reduce(source.basis.to_floats())
        assert reduced is not None
        ours_gram = _float_gram(result.cell.basis.to_floats())
        spglib_gram = _float_gram(reduced.tolist())
        for ours_row, spglib_row in zip(ours_gram, spglib_gram):
            assert ours_row == pytest.approx(spglib_row, abs=1e-9)
