"""Tests for caller scoped structure comparison preparation caches."""

from fractions import Fraction

import pytest
from httk.core import FracVector

from httk.atomistic import (
    ASUStructure,
    Cell,
    FundamentalDomainStructure,
    FundamentalDomainTemplate,
    Protostructure,
    Prototype,
    Species,
    WyckoffSite,
)
from httk.atomistic.models import _vector_guards
from httk.atomistic.symmetry import _numpy_travel, paths, structure_delta
from httk.atomistic.symmetry.comparison_cache import StructureComparisonCache
from httk.atomistic.symmetry.comparison_grid import StructureComparisonGrid

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
    for keyword in ("max_structures", "max_geometries", "max_geometry_bytes"):
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


def test_normalizer_cache_is_separate_bounded_and_does_not_retain_failures() -> None:
    first, second = _structure(5, F(1, 10)), _structure(5, F(11, 100))
    cache = StructureComparisonCache(max_structures=1)
    canonical = cache._structure(first, object)
    def fail():
        raise RuntimeError("failed")

    with pytest.raises(RuntimeError):
        cache._normalizer_images(first, fail)
    images = cache._normalizer_images(first, lambda: (first,))
    assert cache._normalizer_images(first, lambda: pytest.fail("cache miss")) is images
    cache._normalizer_images(second, lambda: (second,))
    assert len(cache._normalizers) == 1
    assert cache._structure(first, lambda: pytest.fail("canonical entry evicted")) is canonical
    regenerated = cache._normalizer_images(first, lambda: (first,))
    assert regenerated == images and regenerated is not images
    cache.clear()
    assert not cache._normalizers


def test_geometry_byte_budget_evicts_lru_and_bypasses_oversized_arrays() -> None:
    numpy = pytest.importorskip("numpy")
    cache = StructureComparisonCache(max_geometries=3, max_geometry_bytes=48)
    first, second, third, large = (object() for _ in range(4))

    def prepare(source, count=1):
        return cache._geometry(
            source, lambda: numpy.zeros((count, 3)), nbytes=lambda array: array.nbytes
        )

    a, b = prepare(first), prepare(second)
    assert cache._geometry_bytes == 48
    assert prepare(first) is a  # Make the first entry most recently used.
    oversized = prepare(large, 3)
    assert oversized.nbytes == 72
    assert id(large) not in cache._geometries
    assert cache._geometry_bytes == 48
    assert cache._geometries[id(second)][1] is b  # Bypass does not flush useful entries.
    prepare(third)
    assert id(second) not in cache._geometries
    assert prepare(first) is a
    assert len(cache._geometries) == 2
    assert cache._geometry_bytes == 48
    cache.clear()
    assert cache._geometry_bytes == 0 and not cache._geometries


def test_geometry_budget_keeps_the_explicit_entry_limit() -> None:
    numpy = pytest.importorskip("numpy")
    cache = StructureComparisonCache(max_geometries=1, max_geometry_bytes=1024)
    for source in (object(), object()):
        cache._geometry(source, lambda: numpy.zeros(3), nbytes=lambda array: array.nbytes)
    assert len(cache._geometries) == 1
    assert cache._geometry_bytes == 24


def test_grid_populates_read_only_geometry_for_later_comparisons(monkeypatch: pytest.MonkeyPatch) -> None:
    numpy = pytest.importorskip("numpy")
    values = (_structure(5, F(1, 10)), _structure(5, F(11, 100)))
    cache = StructureComparisonCache()
    grid = StructureComparisonGrid(values, 0.01, dimensions=2, cache=cache)
    assert grid.fallback_reason is None
    assert cache._geometry_bytes == sum(
        orbit.nbytes for _, orbits, _ in cache._geometries.values() for orbit in orbits
    )
    assert cache._geometry_bytes > 0
    for source, orbits, _ in tuple(cache._geometries.values()):
        assert _numpy_travel._cached_cartesian_orbits(source, numpy, cache) is orbits
        assert all(not orbit.flags.writeable for orbit in orbits)

    def unexpected_expansion(*args, **kwargs):
        pytest.fail("a rejected comparison rebuilt geometry already prepared by the grid")

    monkeypatch.setattr(_numpy_travel, "_cartesian_orbits", unexpected_expansion)
    assert not paths._structure_within_delta(*values, 0.01, use_numpy=True, cache=cache)


@pytest.mark.parametrize("kind", ("structure", "prototype", "protostructure"))
def test_grid_and_comparisons_share_normalizers_without_recanonicalizing(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Grid preparation survives both directed comparisons for each representative-bearing family."""
    pytest.importorskip("numpy")
    from httk.atomistic.models.prototype.derived import _anonymous_template_from_structure

    structures = [
        ASUStructure(
            Cell(((edge, 0, 0), (0, edge, 0), (0, 0, edge))),
            225,
            (WyckoffSite("a", (), "Si"), WyckoffSite("b", (), "O")),
            _species("Si", "O"),
        )
        for edge in (F(5), F(51, 10))
    ]
    if kind == "prototype":
        values = [Prototype(representative=_anonymous_template_from_structure(value)) for value in structures]
    elif kind == "protostructure":
        values = [Protostructure(representative=value) for value in structures]
    else:
        values = structures

    def similar(left, right, budget, cache=None):
        if kind == "structure":
            return paths._structure_within_delta(left, right, budget, use_numpy=True, cache=cache)
        return left.similar(right, budget, use_numpy=True, cache=cache)

    budgets = (0.01, 1.0)
    expected = [similar(values[0], values[1], budget) for budget in budgets]
    assert expected == [False, True]
    calls = {"canonical": 0, "normalizers": 0}
    original_canonical = paths.canonicalize_full
    original_normalizers = paths._build_normalizer_candidates

    def canonical(*args, **kwargs):
        calls["canonical"] += 1
        return original_canonical(*args, **kwargs)

    def normalizers(*args, **kwargs):
        calls["normalizers"] += 1
        return original_normalizers(*args, **kwargs)

    monkeypatch.setattr(paths, "canonicalize_full", canonical)
    monkeypatch.setattr(paths, "_build_normalizer_candidates", normalizers)
    cache = StructureComparisonCache(max_structures=2 * len(values))
    grid = StructureComparisonGrid(values, 1.0, dimensions=2, strategy="variance", cache=cache)
    assert grid.fallback_reason is None
    assert calls == {"canonical": 2, "normalizers": 2}
    for left, right in (values, values[::-1], values):
        assert [similar(left, right, budget, cache) for budget in budgets] == expected
    assert calls == {"canonical": 2, "normalizers": 2}


def test_exact_default_remains_available_when_numpy_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = _structure(5, F(1, 10)), _structure(5, F(11, 100))
    cache = StructureComparisonCache()
    exact = structure_delta(first, second, cache=cache)
    monkeypatch.setattr(_vector_guards, "numpy_available", lambda: False)

    assert structure_delta(first, second, cache=cache) == exact
    with pytest.raises(ImportError, match="requires numpy"):
        structure_delta(first, second, use_numpy=True, cache=cache)
