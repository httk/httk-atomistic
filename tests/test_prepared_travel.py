"""Regression tests for the prepared geometry used by structure travel."""

import itertools
import math
from fractions import Fraction

import pytest
from httk.core import FracVector, SurdVector

from httk.atomistic import ASUStructure, Cell, CellParams, Species, WyckoffSite
from httk.atomistic.symmetry import paths
from httk.atomistic.symmetry._nearest_image import _NearestImageMetric
from httk.atomistic.symmetry.paths import (
    _cartesian_orbits,
    _pair_travel_score,
    _prepared_orbit_travel,
    _TravelMetric,
)

F = Fraction


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _basis_rows(cell: Cell) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in SurdVector(cell.basis).to_floats())


def _mean_gram(
    first: tuple[tuple[float, float, float], ...], second: tuple[tuple[float, float, float], ...]
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple((_dot(first[row], first[column]) + _dot(second[row], second[column])) / 2.0 for column in range(3))
        for row in range(3)
    )


def _cholesky(gram: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    lower00 = math.sqrt(gram[0][0])
    lower10 = gram[1][0] / lower00
    lower20 = gram[2][0] / lower00
    lower11 = math.sqrt(gram[1][1] - lower10**2)
    lower21 = (gram[2][1] - lower20 * lower10) / lower11
    lower22 = math.sqrt(gram[2][2] - lower20**2 - lower21**2)
    return ((lower00, 0.0, 0.0), (lower10, lower11, 0.0), (lower20, lower21, lower22))


def _row_matrix_product(
    row: tuple[float, float, float], matrix: tuple[tuple[float, float, float], ...]
) -> tuple[float, float, float]:
    return (
        math.fsum(row[index] * matrix[index][0] for index in range(3)),
        math.fsum(row[index] * matrix[index][1] for index in range(3)),
        math.fsum(row[index] * matrix[index][2] for index in range(3)),
    )


def _solve_cholesky(
    lower: tuple[tuple[float, float, float], ...], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    first = right[0] / lower[0][0]
    second = (right[1] - lower[1][0] * first) / lower[1][1]
    third = (right[2] - lower[2][0] * first - lower[2][1] * second) / lower[2][2]
    result_third = third / lower[2][2]
    result_second = (second - lower[2][1] * result_third) / lower[1][1]
    return (
        (first - lower[1][0] * result_second - lower[2][0] * result_third) / lower[0][0],
        result_second,
        result_third,
    )


def _original_point_oracle(
    first_cartesian: SurdVector,
    first_cell: Cell,
    second_cartesian: SurdVector,
    second_cell: Cell,
) -> float:
    """Frozen copy of the original point-travel arithmetic, after exact conversion."""
    first_rows = _basis_rows(first_cell)
    second_rows = _basis_rows(second_cell)
    displacement_values = tuple(float(value) for value in (first_cartesian - second_cartesian).to_floats())
    displacement = displacement_values[0], displacement_values[1], displacement_values[2]
    gram = _mean_gram(first_rows, second_rows)
    lower = _cholesky(gram)
    mean_rows = tuple(
        tuple((first_rows[index][column] + second_rows[index][column]) / 2.0 for column in range(3))
        for index in range(3)
    )
    linear = tuple(_dot(displacement, mean_rows[index]) for index in range(3))
    center = _solve_cholesky(lower, (-linear[0], -linear[1], -linear[2]))
    nearest_squared = _NearestImageMetric(lower, (True, True, True)).distance(_row_matrix_product(center, lower)) ** 2
    baseline = _dot(displacement, displacement) - _dot(center, _row_matrix_product(center, gram))
    squared = baseline + nearest_squared
    if squared < 0.0:
        roundoff = 1e-12 * max(1.0, abs(baseline), nearest_squared)
        if squared >= -roundoff:
            squared = 0.0
        else:
            raise ValueError("oracle produced a negative travel squared")
    return math.sqrt(squared)


def _cartesian(frac: tuple[Fraction, Fraction, Fraction], cell: Cell) -> SurdVector:
    return SurdVector(FracVector(frac)) * cell.basis


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _travel_structures() -> tuple[Cell, Cell]:
    # The first basis contains an exact sqrt(3), and both bases are skew and unequal.
    return (
        Cell(CellParams((3, 3, 5, 90, 90, 120)).basis),
        Cell(((4, 1, 0), (F(1, 2), 5, 1), (1, 0, 6))),
    )


def test_prepared_metric_matches_original_formula_for_irrational_unequal_cells() -> None:
    first_cell, second_cell = _travel_structures()
    metric = _TravelMetric.from_cells(first_cell, second_cell)
    points = (
        ((F(1, 7), F(2, 9), F(1, 5)), (F(4, 11), F(3, 8), F(7, 13))),
        ((F(13, 17), F(1, 19), F(11, 23)), (F(2, 5), F(9, 14), F(5, 12))),
    )

    for first_frac, second_frac in points:
        first = _cartesian(first_frac, first_cell)
        second = _cartesian(second_frac, second_cell)
        assert metric.distance(first, second) == _original_point_oracle(first, first_cell, second, second_cell)


def test_prepared_metric_subtracts_large_exact_cartesian_coordinates_before_float_conversion() -> None:
    first_cell = Cell(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    second_cell = Cell(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    huge = F(10**16)
    first = _cartesian((huge + F(1, 3), huge + F(1, 5), huge + F(1, 7)), first_cell)
    second = _cartesian((huge + F(1, 3) - F(1, 11), huge + F(1, 5) + F(1, 13), huge + F(1, 7)), second_cell)

    expected = _original_point_oracle(first, first_cell, second, second_cell)
    assert expected > 0.0
    assert _TravelMetric.from_cells(first_cell, second_cell).distance(first, second) == expected


def test_cartesian_orbits_keep_exact_coordinates() -> None:
    cell = Cell(CellParams((3, 3, 5, 90, 90, 120)).basis)
    structure = ASUStructure(
        cell,
        2,
        (
            WyckoffSite("i", FracVector([F(1, 7), F(2, 9), F(1, 5)]), "Si"),
            WyckoffSite("i", FracVector([F(3, 8), F(4, 11), F(5, 13)]), "Si"),
        ),
        _species("Si"),
    )

    orbits = _cartesian_orbits(structure)
    assert tuple(len(orbit) for orbit in orbits) == (2, 2)
    for site, orbit in zip(structure.wyckoff_sites, orbits, strict=True):
        fractional = structure.spacegroup.wyckoff_position(site.wyckoff).coordinates(site.free_params)
        expected = tuple(SurdVector(FracVector(coordinate)) * cell.basis for coordinate in fractional)
        assert orbit == expected
        assert all(isinstance(coordinate, SurdVector) for coordinate in orbit)


def test_prepared_orbit_travel_uses_minimum_assignment_and_matches_brute_force() -> None:
    first_cell, second_cell = _travel_structures()
    metric = _TravelMetric.from_cells(first_cell, second_cell)
    first = tuple(
        _cartesian(coords, first_cell)
        for coords in (
            (F(1, 9), F(2, 7), F(1, 5)),
            (F(4, 9), F(5, 7), F(3, 8)),
            (F(7, 9), F(1, 7), F(5, 8)),
        )
    )
    second = tuple(
        _cartesian(coords, second_cell)
        for coords in (
            (F(3, 11), F(2, 5), F(7, 13)),
            (F(8, 11), F(4, 5), F(1, 13)),
            (F(5, 11), F(1, 5), F(10, 13)),
        )
    )
    costs = tuple(
        tuple(_original_point_oracle(left, first_cell, right, second_cell) for right in second) for left in first
    )
    expected = min(
        math.fsum(costs[left][right] for left, right in enumerate(permutation))
        for permutation in itertools.permutations(range(3))
    )

    assert _prepared_orbit_travel(first, second, metric) == expected


def test_pair_travel_score_prepares_geometry_once_for_all_local_orbit_comparisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_cell, second_cell = _travel_structures()
    first = ASUStructure(
        first_cell,
        2,
        (
            WyckoffSite("i", FracVector([F(1, 7), F(2, 9), F(1, 5)]), "Si"),
            WyckoffSite("i", FracVector([F(3, 8), F(4, 11), F(5, 13)]), "Si"),
        ),
        _species("Si"),
    )
    second = ASUStructure(
        second_cell,
        2,
        (
            WyckoffSite("i", FracVector([F(5, 13), F(1, 6), F(3, 10)]), "Si"),
            WyckoffSite("i", FracVector([F(7, 13), F(2, 5), F(1, 8)]), "Si"),
        ),
        _species("Si"),
    )
    original = _TravelMetric.from_cells
    calls = 0

    def counted(cls: type[_TravelMetric], *cells: Cell) -> _TravelMetric:
        nonlocal calls
        calls += 1
        return original.__func__(cls, *cells)

    prepared_structures: list[ASUStructure] = []

    def counted_orbits(structure: ASUStructure) -> tuple[tuple[SurdVector, ...], ...]:
        prepared_structures.append(structure)
        return _cartesian_orbits(structure)

    monkeypatch.setattr(_TravelMetric, "from_cells", classmethod(counted))
    monkeypatch.setattr(paths, "_cartesian_orbits", counted_orbits)
    score, pairs = _pair_travel_score(first, second)

    assert calls == 1
    assert len(prepared_structures) == 2
    assert prepared_structures[0] is second
    assert prepared_structures[1] is first
    assert len(pairs) == 2
    assert math.isfinite(score) and score >= 0.0


def test_incompatible_orbit_counts_are_rejected_before_geometry_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    cell = Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7)))
    sites = (
        WyckoffSite("i", FracVector([F(1, 7), F(2, 9), F(1, 5)]), "Si"),
        WyckoffSite("i", FracVector([F(3, 8), F(4, 11), F(5, 13)]), "Si"),
    )
    first = ASUStructure(cell, 2, sites, _species("Si"))
    second = ASUStructure(cell, 2, sites[:1], _species("Si"))

    def unexpected(*args: object) -> None:
        pytest.fail("incompatible structures must not prepare geometry")

    monkeypatch.setattr(_TravelMetric, "from_cells", unexpected)
    with pytest.raises(ValueError, match="incompatible site classes"):
        _pair_travel_score(first, second)
