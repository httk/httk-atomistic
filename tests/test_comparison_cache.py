"""Tests for caller scoped structure comparison preparation caches."""

from fractions import Fraction

import pytest
from httk.core import FracVector

from httk.atomistic import (
    ASUStructure,
    Cell,
    FundamentalDomainStructure,
    FundamentalDomainTemplate,
    Prototype,
    Species,
    WyckoffSite,
)
from httk.atomistic.models import _vector_guards
from httk.atomistic.symmetry import _numpy_travel, paths, structure_delta
from httk.atomistic.symmetry.comparison_cache import StructureComparisonCache

F = Fraction


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _structure(edge: int, value: F) -> ASUStructure:
    return ASUStructure(
        Cell(((edge, 0, 0), (0, edge, 0), (0, 0, edge))),
        1,
        (
            WyckoffSite("a", FracVector((0, 0, 0)), "C"),
            WyckoffSite("a", FracVector((value, F(1, 3), F(1, 5))), "O"),
        ),
        _species("C", "O"),
    )


def _fundamental(structure: ASUStructure) -> FundamentalDomainStructure:
    return FundamentalDomainStructure(
        structure.cell,
        structure.spacegroup,
        structure.wyckoff_sites,
        structure.species,
        transform=structure.transform,
        coordinate_precision=structure.coordinate_precision,
        charge=structure.charge,
    )


def _template(value: F) -> FundamentalDomainTemplate:
    return FundamentalDomainTemplate(
        Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        1,
        (
            WyckoffSite("a", FracVector((0, 0, 0)), "A"),
            WyckoffSite("a", FracVector((value, F(1, 3), F(1, 5))), "B"),
        ),
    )


def test_cache_requires_positive_integer_capacities() -> None:
    for keyword in ("max_structures", "max_geometries"):
        for value in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match="positive integer"):
                StructureComparisonCache(**{keyword: value})


def test_cached_and_uncached_scores_match_for_repeated_comparisons() -> None:
    values = (_structure(5, F(1, 10)), _structure(5, F(11, 100)), _structure(6, F(1, 10)))
    uncached = tuple(structure_delta(left, right) for left, right in ((values[0], values[1]), (values[0], values[2])))
    cache = StructureComparisonCache()
    cached = tuple(
        structure_delta(left, right, cache=cache) for left, right in ((values[0], values[1]), (values[0], values[2]))
    )
    repeated = tuple(
        structure_delta(left, right, cache=cache) for left, right in ((values[0], values[1]), (values[0], values[2]))
    )

    assert cached == pytest.approx(uncached, rel=1e-12, abs=1e-12)
    assert repeated == pytest.approx(cached, rel=1e-12, abs=1e-12)


def test_cache_canonicalizes_each_unique_identity_once(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second, third, equal_first = (
        _structure(5, F(1, 10)),
        _structure(5, F(11, 100)),
        _structure(6, F(1, 10)),
        _structure(5, F(1, 10)),
    )
    cache = StructureComparisonCache()
    original = paths.canonicalize_full
    calls: list[object] = []

    def counted(structure: object, *args: object, **kwargs: object) -> object:
        calls.append(structure)
        return original(structure, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(paths, "canonicalize_full", counted)
    structure_delta(first, second, cache=cache)
    structure_delta(first, third, cache=cache)
    structure_delta(equal_first, second, cache=cache)
    structure_delta(first, second, cache=cache)

    assert sum(value is first for value in calls) == 1
    assert sum(value is second for value in calls) == 1
    assert sum(value is third for value in calls) == 1
    assert sum(value is equal_first for value in calls) == 1
    assert len(calls) == 4
    assert any(source is first for source, _, _ in cache._structures.values())
    assert any(source is equal_first for source, _, _ in cache._structures.values())


def test_tolerance_is_part_of_cached_canonicalization_key(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = _structure(5, F(1, 10)), _structure(5, F(11, 100))
    cache = StructureComparisonCache()
    original = paths.canonicalize_full
    calls = 0

    def counted(structure: object, *args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(structure, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(paths, "canonicalize_full", counted)
    structure_delta(first, second, tolerance=0.01, cache=cache)
    structure_delta(first, second, tolerance=0.01, cache=cache)
    structure_delta(first, second, tolerance=0.005, cache=cache)

    assert calls == 4


def test_numpy_cache_reuses_orbit_geometry_and_matches_uncached(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("numpy")
    first, second = _structure(5, F(1, 10)), _structure(5, F(11, 100))
    uncached = structure_delta(first, second, use_numpy=True)
    cache = StructureComparisonCache()
    cached = structure_delta(first, second, use_numpy=True, cache=cache)
    repeated = structure_delta(first, second, use_numpy=True, cache=cache)
    assert cached == pytest.approx(uncached, rel=1e-12, abs=1e-12)
    assert repeated == pytest.approx(cached, rel=1e-12, abs=1e-12)

    # The lower-level prepared travel call has exactly two source geometries, so
    # repeated preparation demonstrates the identity cache directly.
    geometry_cache = StructureComparisonCache()
    original = _numpy_travel._cartesian_orbits
    calls = 0

    def counted(structure: ASUStructure, numpy: object) -> object:
        nonlocal calls
        calls += 1
        return original(structure, numpy)

    monkeypatch.setattr(_numpy_travel, "_cartesian_orbits", counted)
    _numpy_travel.prepare_travel(first, second, cache=geometry_cache)(0, 0)
    _numpy_travel.prepare_travel(first, second, cache=geometry_cache)(0, 0)
    assert calls == 2
    assert len(geometry_cache._geometries) == 2


def test_prototype_cache_reuses_template_to_asu_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("numpy")
    first = Prototype(representative=_template(F(1, 10)))
    second = Prototype(representative=_template(F(11, 100)))
    cache = StructureComparisonCache()
    from httk.atomistic.models.prototype import derived

    original = derived._prototype_to_structure
    calls = 0

    def counted(prototype: object) -> object:
        nonlocal calls
        calls += 1
        return original(prototype)

    monkeypatch.setattr(derived, "_prototype_to_structure", counted)

    assert not first.similar(second, 0.05, use_numpy=True, cache=cache)
    assert first.similar(second, 0.2, use_numpy=True, cache=cache)
    assert first.similar(second, 0.2, use_numpy=True, cache=cache)
    assert calls == 2


def test_cache_accepts_unhashable_fundamental_domain_inputs() -> None:
    pytest.importorskip("numpy")
    first = _fundamental(_structure(5, F(1, 10)))
    second = _fundamental(_structure(5, F(11, 100)))
    cache = StructureComparisonCache()

    with pytest.raises(TypeError):
        hash(first)
    uncached = structure_delta(first, second, use_numpy=True)
    cached = structure_delta(first, second, use_numpy=True, cache=cache)

    assert cached == pytest.approx(uncached, rel=1e-12, abs=1e-12)


def test_cache_eviction_clear_and_failed_preparation() -> None:
    first, second = _structure(5, F(1, 10)), _structure(5, F(11, 100))
    cache = StructureComparisonCache(max_structures=1, max_geometries=1)
    attempts = 0

    def failing_then_working() -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient preparation failure")
        return object()

    with pytest.raises(RuntimeError):
        cache._structure(first, failing_then_working)
    result = cache._structure(first, failing_then_working)
    assert attempts == 2
    assert cache._structure(first, lambda: pytest.fail("should be cached")) is result

    geometry_attempts = 0

    def failing_geometry() -> object:
        nonlocal geometry_attempts
        geometry_attempts += 1
        if geometry_attempts == 1:
            raise RuntimeError("transient geometry failure")
        return object()

    with pytest.raises(RuntimeError):
        cache._geometry(first, failing_geometry)
    geometry_result = cache._geometry(first, failing_geometry)
    assert geometry_attempts == 2
    assert cache._geometry(first, lambda: pytest.fail("should be cached")) is geometry_result

    cache._structure(second, lambda: object())
    assert len(cache._structures) == 1
    cache._geometry(second, lambda: object())
    assert len(cache._geometries) == 1
    cache.clear()
    assert not cache._structures
    assert not cache._geometries


def test_exact_default_remains_available_when_numpy_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = _structure(5, F(1, 10)), _structure(5, F(11, 100))
    cache = StructureComparisonCache()
    exact = structure_delta(first, second, cache=cache)
    monkeypatch.setattr(_vector_guards, "numpy_available", lambda: False)

    assert structure_delta(first, second, cache=cache) == exact
    with pytest.raises(ImportError, match="requires numpy"):
        structure_delta(first, second, use_numpy=True, cache=cache)
