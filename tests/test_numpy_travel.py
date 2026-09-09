"""Tests for the optional float64 structure-travel comparison path."""

import math
from fractions import Fraction

import pytest
from httk.core import FracVector
from httk.core.storage import content_id

from httk.atomistic import (
    ASUStructure,
    Cell,
    CellParams,
    FundamentalDomainTemplate,
    Protostructure,
    Prototype,
    Species,
    WyckoffSite,
)
from httk.atomistic.models import _vector_guards
from httk.atomistic.symmetry import _numpy_travel, structure_delta
from httk.atomistic.symmetry.paths import _cartesian_orbits, _pair_travel_score, _prepared_orbit_travel, _TravelMetric

pytest.importorskip("numpy")

F = Fraction


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _skew_structure(value: F, basis: object | None = None) -> ASUStructure:
    if basis is None:
        basis = ((1, 0, 0), (10, F(1, 100), 0), (0, 0, 1))
    return ASUStructure(
        Cell(basis),
        1,
        (
            WyckoffSite("a", FracVector((0, 0, 0)), "C"),
            WyckoffSite("a", FracVector((value, value, F(1, 10))), "O"),
        ),
        _species("C", "O"),
    )


def _repeated_site_structure(
    coordinates: tuple[tuple[F, F, F], ...],
    *,
    basis: object = ((3, 0, 0), (F(1, 2), 4, 0), (0, 0, 5)),
) -> ASUStructure:
    return ASUStructure(
        Cell(basis),
        1,
        tuple(WyckoffSite("a", FracVector(coordinate), "Si") for coordinate in coordinates),
        _species("Si"),
    )


def _p_minus_one_structure(value: F, basis: object) -> ASUStructure:
    return ASUStructure(
        Cell(basis),
        2,
        (WyckoffSite("i", FracVector((F(1, 7), F(2, 9), value)), "Si"),),
        _species("Si"),
    )


def _anonymous_template(value: F) -> FundamentalDomainTemplate:
    return FundamentalDomainTemplate(
        Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        1,
        (
            WyckoffSite("a", FracVector((0, 0, 0)), "A"),
            WyckoffSite("a", FracVector((value, value, value)), "B"),
        ),
    )


def test_numpy_travel_agrees_with_exact_for_unequal_skew_cells() -> None:
    first = _skew_structure(F(1, 100))
    second = _skew_structure(
        F(49, 100),
        ((F(6, 5), 0, 0), (8, F(1, 80), 0), (0, 0, F(11, 10))),
    )

    exact = structure_delta(first, second)
    numeric = structure_delta(first, second, use_numpy=True)

    assert numeric == pytest.approx(exact, rel=1e-12, abs=1e-12)
    assert structure_delta(second, first, use_numpy=True) == pytest.approx(numeric, rel=1e-12, abs=1e-12)


def test_numpy_travel_keeps_the_symmetric_skew_cell_minimum_image() -> None:
    first = _skew_structure(F(1, 100))
    second = _skew_structure(F(49, 100))

    # This displacement needs a non-local periodic image in the skew cell.
    expected = math.hypot(7 / 25, 3 / 625)
    numeric = structure_delta(first, second, use_numpy=True)

    assert numeric == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_numpy_pair_travel_preserves_species_assignment_and_hungarian_matching() -> None:
    first = _repeated_site_structure(
        (
            (F(1, 10), F(1, 7), F(1, 9)),
            (F(4, 10), F(2, 7), F(2, 9)),
            (F(8, 10), F(4, 7), F(4, 9)),
        )
    )
    second = _repeated_site_structure(
        (
            (F(8, 10), F(4, 7), F(4, 9)),
            (F(1, 10), F(1, 7), F(1, 9)),
            (F(4, 10), F(2, 7), F(2, 9)),
        )
    )

    exact_score, exact_pairs = _pair_travel_score(first, second)
    numeric_score, numeric_pairs = _pair_travel_score(first, second, use_numpy=True)

    assert numeric_score == pytest.approx(exact_score, rel=1e-12, abs=1e-12)
    assert numeric_pairs == exact_pairs
    assert numeric_pairs == ((0, 2), (1, 0), (2, 1))


def test_numpy_prepared_multi_atom_orbit_agrees_with_exact_skew_travel() -> None:
    basis = CellParams((3, 3, 5, 90, 90, 120)).basis
    first = _p_minus_one_structure(F(1, 100), basis)
    second = _p_minus_one_structure(F(49, 100), basis)

    exact_metric = _TravelMetric.from_cells(first.cell, second.cell)
    first_orbit = _cartesian_orbits(first)[0]
    second_orbit = _cartesian_orbits(second)[0]
    exact = _prepared_orbit_travel(first_orbit, second_orbit, exact_metric)
    numeric = _numpy_travel.prepare_travel(first, second)(0, 0)

    assert len(first_orbit) == len(second_orbit) == 2
    assert numeric == pytest.approx(exact, rel=1e-12, abs=1e-12)


def test_numpy_similarity_is_opt_in_and_does_not_mutate_exact_representatives() -> None:
    first = _skew_structure(F(1, 100))
    second = _skew_structure(F(49, 100))
    first_basis = first.cell.basis
    first_sites = first.wyckoff_sites
    second_basis = second.cell.basis
    second_sites = second.wyckoff_sites
    exact = structure_delta(first, second)
    first_id = content_id(first)
    second_id = content_id(second)

    first_value = Protostructure(representative=first)
    second_value = Protostructure(representative=second)

    assert first_value.similar(second_value, exact + 1e-10, use_numpy=True)
    assert not first_value.similar(second_value, max(0.0, exact - 1e-10), use_numpy=True)
    assert first.cell.basis == first_basis
    assert first.wyckoff_sites == first_sites
    assert second.cell.basis == second_basis
    assert second.wyckoff_sites == second_sites
    assert content_id(first) == first_id
    assert content_id(second) == second_id
    assert all(isinstance(value, F) for row in first.cell.basis.fractions for value in row)


def test_numpy_similarity_supports_anonymous_prototype_representatives() -> None:
    first = Prototype(representative=_anonymous_template(F(1, 10)))
    second = Prototype(representative=_anonymous_template(F(11, 100)))
    first_id = content_id(first)
    second_id = content_id(second)

    assert not first.similar(second, 0.05, use_numpy=True)
    assert first.similar(second, 0.2, use_numpy=True)
    assert first == Prototype(representative=first.representative)
    assert content_id(first) == first_id
    assert content_id(second) == second_id
    assert first.representative is not None
    assert all(
        isinstance(value, F)
        for site in first.representative.wyckoff_sites
        for value in site.free_params.to_fractions()
    )


def test_numpy_mode_reports_an_optional_dependency_without_affecting_exact_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _skew_structure(F(1, 100))
    second = _skew_structure(F(49, 100))
    exact = structure_delta(first, second)
    monkeypatch.setattr(_vector_guards, "numpy_available", lambda: False)

    with pytest.raises(ImportError, match="requires numpy"):
        structure_delta(first, second, use_numpy=True)
    assert structure_delta(first, second) == exact
