"""Regression tests for the conservative structure-comparison grid."""

from fractions import Fraction

import pytest
from httk.core import FracVector

from httk.atomistic import ASUStructure, Cell, Prototype, Species, WyckoffSite, data, subgroup_representation
from httk.atomistic.models import _vector_guards
from httk.atomistic.symmetry.comparison_grid import StructureComparisonGrid
from httk.atomistic.symmetry.lift import _apply_normalizer
from httk.atomistic.symmetry.paths import structure_delta

F = Fraction


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _p1(value: F, *, basis: object = ((5, 0, 0), (0, 6, 0), (0, 0, 7))) -> ASUStructure:
    return ASUStructure(
        Cell(basis),
        1,
        (
            WyckoffSite("a", FracVector((0, 0, 0)), "C"),
            WyckoffSite("a", FracVector((value, value, F(1, 10))), "O"),
        ),
        _species("C", "O"),
    )


def _p_minus_one(value: F, *, basis: object = ((5, 0, 0), (0, 6, 0), (0, 0, 7))) -> ASUStructure:
    return ASUStructure(
        Cell(basis),
        2,
        (
            WyckoffSite("i", FracVector((value, F(2, 9), F(1, 7))), "Si"),
            WyckoffSite("i", FracVector((F(1, 3), F(4, 11), F(2, 5))), "O"),
        ),
        _species("Si", "O"),
    )


def _repeated_sites(coordinates: tuple[tuple[F, F, F], ...]) -> ASUStructure:
    return ASUStructure(
        Cell(((3, 0, 0), (F(1, 2), 4, 0), (0, 0, 5))),
        1,
        tuple(WyckoffSite("a", FracVector(coordinate), "Si") for coordinate in coordinates),
        _species("Si"),
    )


@pytest.mark.parametrize("strategy", ("first", "variance", "occupancy"))
@pytest.mark.parametrize("dimensions", (1, 2, 3))
def test_grid_keeps_close_pairs_for_each_projection_strategy(strategy: str, dimensions: int) -> None:
    first = _p1(F(1, 100))
    second = _p1(F(3, 100))
    delta = structure_delta(first, second, use_numpy=True) + 1e-9

    grid = StructureComparisonGrid((first, second), delta, dimensions=dimensions, strategy=strategy)

    assert grid.might_match(0, 1)
    assert grid.might_match(1, 0)
    assert grid.selected_axes
    assert len(grid.selected_axes) == dimensions
    assert grid.fallback_reason is None


def test_grid_handles_periodic_boundary_and_unequal_skew_cells() -> None:
    first = _p1(F(1, 100), basis=((5, 0, 0), (F(1, 3), 6, 0), (0, 0, 7)))
    second = _p1(F(99, 100), basis=((F(26, 5), 0, 0), (F(1, 2), F(31, 5), 0), (0, 0, F(69, 10))))
    delta = structure_delta(first, second, use_numpy=True) + 1e-9

    grid = StructureComparisonGrid((first, second), delta, dimensions=2, strategy="variance")

    assert grid.might_match(0, 1)
    assert grid.might_match(1, 0)


def test_grid_handles_hungarian_reordering_of_repeated_sites() -> None:
    coordinates = (
        (F(1, 10), F(1, 7), F(1, 9)),
        (F(4, 10), F(2, 7), F(2, 9)),
        (F(8, 10), F(4, 7), F(4, 9)),
    )
    first = _repeated_sites(coordinates)
    second = _repeated_sites((coordinates[2], coordinates[0], coordinates[1]))
    delta = 1e-9

    grid = StructureComparisonGrid((first, second), delta, dimensions=1, strategy="first")

    assert structure_delta(first, second, use_numpy=True) <= delta
    assert grid.might_match(0, 1)


def test_grid_handles_p_minus_one_normalizer_equivalence() -> None:
    parent = ASUStructure(
        Cell(((3, 0, 0), (F(3, 2), F(3, 2), 0), (0, 0, 5))),
        5,
        (WyckoffSite("a", FracVector((F(2, 17),)), "C"),),
        _species("C"),
    )
    first = subgroup_representation(parent, 3).asu
    record = data.affine_normalizer_coset_record(first.spacegroup.hall_entry)
    second = _apply_normalizer(first, record["affine_normalizer_cosets"][0])
    assert second is not None
    delta = 1e-8
    assert structure_delta(first, second, use_numpy=True) <= delta

    grid = StructureComparisonGrid((first, second), delta, dimensions=2, strategy="occupancy")

    assert grid.might_match(0, 1)


def test_grid_keeps_all_close_pairs_in_a_small_heterogeneous_batch() -> None:
    values = (
        _p1(F(1, 100)),
        _p1(F(2, 100)),
        _p1(F(99, 100)),
        _p1(F(3, 100), basis=((F(51, 10), 0, 0), (F(1, 4), F(59, 10), 0), (0, 0, F(69, 10)))),
    )
    delta = 1.0
    grid = StructureComparisonGrid(values, delta, dimensions=2, strategy="variance")

    for first in range(len(values)):
        for second in range(first + 1, len(values)):
            expected = structure_delta(values[first], values[second], use_numpy=True) <= delta
            if expected:
                assert grid.might_match(first, second)


def test_grid_falls_back_for_too_many_periodic_images() -> None:
    values = (_p1(F(1, 10)), _p1(F(2, 10)))
    grid = StructureComparisonGrid(values, 2.0, max_images=1)

    assert grid.might_match(0, 1)


def test_grid_falls_back_for_ill_conditioned_cells() -> None:
    basis = ((1, 0, 0), (1000, F(1, 1000), 0), (0, 0, 1))
    values = (_p1(F(1, 10), basis=basis), _p1(F(2, 10), basis=basis))
    grid = StructureComparisonGrid(values, 0.1)

    assert grid.fallback_reason is not None
    assert grid.might_match(0, 1)


def test_grid_falls_back_for_missing_prototype_representative() -> None:
    values = (Prototype(1, [("a", "A")]), Prototype(1, [("a", "A")]))
    grid = StructureComparisonGrid(values, 0.1)

    assert grid.fallback_reason is not None
    assert grid.might_match(0, 1)


def test_grid_excludes_a_pair_that_is_far_in_all_selected_coordinates() -> None:
    first = _p_minus_one(F(1, 10))
    second = _p_minus_one(F(2, 5))
    delta = 0.01
    assert structure_delta(first, second, use_numpy=True) > delta

    grid = StructureComparisonGrid((first, second), delta, dimensions=3, strategy="occupancy")

    assert not grid.might_match(0, 1)
    assert not grid.might_match(1, 0)


def test_grid_falls_back_when_point_cap_is_too_small() -> None:
    values = tuple(_p1(F(value, 100)) for value in range(4))
    grid = StructureComparisonGrid(values, 0.1, max_points=1)

    assert grid.fallback_reason is not None
    assert grid.might_match(0, 1)


def test_grid_reuses_sparse_queries_across_large_leader_scans(monkeypatch: pytest.MonkeyPatch) -> None:
    values = tuple(_p_minus_one(F(value, 1000)) for value in range(1, 71))
    grid = StructureComparisonGrid(values, 1e-5, dimensions=3)
    assert grid.fallback_reason is None
    neighbors = StructureComparisonGrid._neighbors
    calls: dict[int, int] = {}

    def counted_neighbors(self: StructureComparisonGrid, index: int) -> frozenset[int] | None:
        calls[index] = calls.get(index, 0) + 1
        return neighbors(self, index)

    monkeypatch.setattr(StructureComparisonGrid, "_neighbors", counted_neighbors)
    for candidate in range(1, len(values)):
        for leader in range(candidate):
            assert not grid.might_match(leader, candidate)

    assert len(calls) == len(values)
    assert set(calls.values()) == {1}


def test_grid_rejects_invalid_parameters() -> None:
    values = (_p1(F(1, 10)), _p1(F(1, 10)))
    for kwargs in (
        {"dimensions": 0},
        {"dimensions": 4},
        {"strategy": "unknown"},
        {"max_points": 0},
        {"max_images": 0},
    ):
        with pytest.raises(ValueError):
            StructureComparisonGrid(values, 0.1, **kwargs)
    with pytest.raises(ValueError):
        StructureComparisonGrid(values, -0.1)


def test_grid_reports_missing_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    values = (_p1(F(1, 10)), _p1(F(1, 10)))
    monkeypatch.setattr(_vector_guards, "numpy_available", lambda: False)

    with pytest.raises(ImportError, match="requires numpy"):
        StructureComparisonGrid(values, 0.1)
