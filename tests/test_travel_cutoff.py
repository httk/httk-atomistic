"""Tests for bounded approximate travel decisions."""

from fractions import Fraction

import pytest
from httk.core import FracVector

from httk.atomistic import (
    ASUStructure,
    Cell,
    CellParams,
    Prototype,
    Species,
    WyckoffSite,
    data,
    subgroup_representation,
)
from httk.atomistic.models import _vector_guards
from httk.atomistic.models.prototype import derived
from httk.atomistic.symmetry import paths, structure_delta
from httk.atomistic.symmetry.comparison_cache import StructureComparisonCache
from httk.atomistic.symmetry.lift import _apply_normalizer

F = Fraction


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _p1(value: F, *, edge: int = 5) -> ASUStructure:
    return ASUStructure(
        Cell(((edge, 0, 0), (0, edge, 0), (0, 0, edge))),
        1,
        (
            WyckoffSite("a", FracVector((0, 0, 0)), "C"),
            WyckoffSite("a", FracVector((value, F(1, 3), F(1, 5))), "O"),
        ),
        _species("C", "O"),
    )


def _p_minus_one(value: F, *, edge: int = 5) -> ASUStructure:
    return ASUStructure(
        Cell(((edge, 0, 0), (0, edge, 0), (0, 0, edge))),
        2,
        (WyckoffSite("i", FracVector((F(1, 7), F(2, 9), value)), "Si"),),
        _species("Si"),
    )


def _repeated(values: tuple[F, ...]) -> ASUStructure:
    return ASUStructure(
        Cell(((3, 0, 0), (F(1, 2), 4, 0), (0, 0, 5))),
        1,
        tuple(WyckoffSite("a", FracVector((value, F(1, 7), F(1, 9))), "Si") for value in values),
        _species("Si"),
    )


def _within(first: ASUStructure, second: ASUStructure, delta: float, **kwargs: object) -> bool:
    return paths._structure_within_delta(first, second, delta, **kwargs)


def test_cutoff_decision_agrees_with_full_metric_for_different_cells_and_repeated_calls() -> None:
    numpy = pytest.importorskip("numpy")
    first = _p1(F(1, 10))
    second = _p1(F(11, 100), edge=6)
    full = structure_delta(first, second, use_numpy=True)

    for budget in (float(numpy.nextafter(full, 0.0)), full, float(numpy.nextafter(full, numpy.inf))):
        assert _within(first, second, budget, use_numpy=True) is (full <= budget)

    cache = StructureComparisonCache()
    assert _within(first, second, full + 1e-10, use_numpy=True, cache=cache)
    assert _within(first, second, full + 1e-10, use_numpy=True, cache=cache)


def test_cutoff_handles_zero_and_float_boundary_for_irrational_skew_orbits() -> None:
    numpy = pytest.importorskip("numpy")
    basis = CellParams((3, 3, 5, 90, 90, 120)).basis
    first = ASUStructure(
        Cell(basis),
        2,
        (WyckoffSite("i", FracVector((F(1, 7), F(2, 9), F(1, 100))), "Si"),),
        _species("Si"),
    )
    second = ASUStructure(
        Cell(basis),
        2,
        (WyckoffSite("i", FracVector((F(1, 7), F(2, 9), F(49, 100))), "Si"),),
        _species("Si"),
    )
    full = structure_delta(first, second, use_numpy=True)

    assert _within(first, first, 0.0, use_numpy=True)
    assert not _within(first, second, float(numpy.nextafter(full, 0.0)), use_numpy=True)
    assert _within(first, second, float(numpy.nextafter(full, numpy.inf)), use_numpy=True)


def test_pair_cutoff_prunes_expensive_orbit_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("numpy")
    values = tuple(F(index, 100) for index in range(6))
    reference = _repeated(values)
    candidate = _repeated(tuple(F(49, 100) + value for value in values))
    original = paths._prepare_travel

    def count_calls(use_cutoff: bool) -> int:
        calls = 0

        def counted(*args: object, **kwargs: object):
            nonlocal calls
            travel = original(*args, **kwargs)

            def wrapped(first_index: int, second_index: int) -> float:
                nonlocal calls
                calls += 1
                return travel(first_index, second_index)

            return wrapped

        monkeypatch.setattr(paths, "_prepare_travel", counted)
        if use_cutoff:
            with pytest.raises(paths._TravelCutoffExceeded):
                paths._pair_travel_score(candidate, reference, use_numpy=True, cutoff=0.1)
        else:
            paths._pair_travel_score(candidate, reference, use_numpy=True)
        return calls

    baseline = count_calls(False)
    pruned = count_calls(True)

    assert baseline == 36
    assert pruned < baseline


def test_cutoff_does_not_accept_a_non_assignment_row_minimum() -> None:
    pytest.importorskip("numpy")
    reference = _repeated((F(0), F(1, 10)))
    candidate = _repeated((F(0), F(0)))
    full = structure_delta(reference, candidate, use_numpy=True)

    assert full > 0.1
    assert not _within(reference, candidate, 0.1, use_numpy=True)


def test_recognized_group_over_cutoff_does_not_fall_back_to_a_lower_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numpy = pytest.importorskip("numpy")
    first = _p_minus_one(F(1, 100))
    second = _p_minus_one(F(49, 100))
    targets: list[int] = []
    original = paths.rerepresent

    def counted(structure: ASUStructure, target: object, **kwargs: object) -> ASUStructure:
        targets.append(target.it_number)
        return original(structure, target, **kwargs)

    monkeypatch.setattr(paths, "rerepresent", counted)
    assert not _within(first, second, float(numpy.nextafter(0.0, numpy.inf)), use_numpy=True)
    assert targets and set(targets) == {2}


def test_cutoff_continues_after_a_rejected_normalizer_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    numpy = pytest.importorskip("numpy")
    parent = ASUStructure(
        Cell(CellParams((3, 3, 5, 90, 90, 120)).basis),
        5,
        (WyckoffSite("a", FracVector((F(2, 17),)), "Si"),),
        _species("Si"),
    )
    original = subgroup_representation(parent, 3).asu
    record = data.affine_normalizer_coset_record(original.spacegroup.hall_entry)
    transformed = _apply_normalizer(original, record["affine_normalizer_cosets"][0])
    assert transformed is not None
    original_pair_score = paths._pair_travel_score
    rejected = 0
    accepted = 0

    def counted(candidate: ASUStructure, reference: ASUStructure, **kwargs: object):
        nonlocal accepted, rejected
        try:
            result = original_pair_score(candidate, reference, **kwargs)
        except paths._TravelCutoffExceeded:
            rejected += 1
            raise
        if kwargs.get("cutoff") is not None:
            accepted += 1
        return result

    monkeypatch.setattr(paths, "_pair_travel_score", counted)

    # One normalizer image can exceed a zero budget before a later equivalent image
    # reaches the exact coincidence. The alignment search must keep trying candidates.
    assert _within(original, transformed, float(numpy.nextafter(0.0, numpy.inf)), use_numpy=True)
    assert rejected >= 1
    assert accepted >= 1


@pytest.mark.parametrize("rejection", ("pair_cutoff", "final_score"))
def test_cutoff_fallback_continues_after_an_over_budget_subgroup(
    monkeypatch: pytest.MonkeyPatch,
    rejection: str,
) -> None:
    """An over-budget fallback candidate must not suppress later valid groups."""
    pytest.importorskip("numpy")
    first = _p_minus_one(F(1, 10))
    parent = ASUStructure(
        Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        5,
        (WyckoffSite("a", FracVector((F(2, 17),)), "Si"),),
        _species("Si"),
    )
    second = subgroup_representation(parent, 3).asu
    targets: list[int] = []
    active_target = 0

    def fake_closure(group: object, *, include_self: bool = True) -> set[int]:
        if group.it_number == 2:
            return {1, 2, 4}
        if group.it_number == 3:
            return {1, 3, 4}
        return {group.it_number}

    def fake_canonicalize(structure: ASUStructure, *args: object, **kwargs: object) -> ASUStructure:
        return structure

    def fake_rerepresent(structure: ASUStructure, target: object, **kwargs: object) -> ASUStructure:
        nonlocal active_target
        active_target = target.it_number
        targets.append(active_target)
        return structure

    def fake_aligned(candidate: ASUStructure, reference: ASUStructure, **kwargs: object):
        if active_target == 4 and rejection == "pair_cutoff":
            raise paths._TravelCutoffExceeded
        if active_target == 4 and rejection == "final_score":
            return paths._Alignment(reference, ((0, 0),))
        return paths._Alignment(reference, ())

    monkeypatch.setattr(paths, "subgroup_closure", fake_closure)
    monkeypatch.setattr(paths, "canonicalize_full", fake_canonicalize)
    monkeypatch.setattr(paths, "rerepresent", fake_rerepresent)
    monkeypatch.setattr(paths, "_aligned", fake_aligned)
    if rejection == "final_score":
        monkeypatch.setattr(paths, "_prepare_travel", lambda *args, **kwargs: lambda *_: 1.0)

    assert _within(first, second, 0.5, use_numpy=True)
    assert targets == [4, 4, 1, 1]


def test_cutoff_keeps_similarity_delta_validation_and_unsupported_guards() -> None:
    first = Prototype(representative=derived._anonymous_template_from_structure(_p1(F(1, 10))))
    second = Prototype(representative=derived._anonymous_template_from_structure(_p1(F(11, 100))))

    with pytest.raises(ValueError):
        first.similar(second, -1.0, use_numpy=True)
    with pytest.raises(ValueError):
        first.similar(second, float("inf"), use_numpy=True)
    with pytest.raises(TypeError):
        first.similar(second, "0", use_numpy=True)  # type: ignore[arg-type]

    original_numpy = _vector_guards.numpy_available
    try:
        _vector_guards.numpy_available = lambda: False  # type: ignore[assignment]
        assert structure_delta(_p1(F(1, 10)), _p1(F(11, 100))) >= 0.0
        with pytest.raises(ImportError, match="requires numpy"):
            _within(_p1(F(1, 10)), _p1(F(11, 100)), 1.0, use_numpy=True)
    finally:
        _vector_guards.numpy_available = original_numpy
