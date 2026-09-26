"""Promotion contracts for the anonymous default and explicit legacy entry points."""

import datetime
from fractions import Fraction as F
from pathlib import Path

import pytest
from httk.core import FracVector, load

import httk.atomistic.symmetry.lift as lift_module
from httk import atomistic
from httk.atomistic import ASUStructure, Cell, ChemicalComposition, Species, WyckoffSite, symmetry
from httk.atomistic.symmetry.canonical_protostructure import (
    _canonical_protostructure_assignments_asu,
    _canonical_protostructure_asu,
)
from httk.atomistic.symmetry.canonical import canonicalize
from httk.atomistic.symmetry.lift import canonicalize_legacy


def _species(name: str, symbol: str, **kwargs: object) -> Species:
    return Species(name, (symbol,), (1,), **kwargs)


def _sg47() -> ASUStructure:
    return ASUStructure(
        Cell(((3, 0, 0), (0, 4, 0), (0, 0, 5))),
        47,
        (WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("c", FracVector(()), "O")),
        (_species("Na", "Na"), _species("O", "O")),
    )


def _anonymous_pattern(structure: ASUStructure) -> tuple[tuple[str, ...], ...]:
    classes: dict[str, list[str]] = {}
    for site in structure.wyckoff_sites:
        classes.setdefault(site.species, []).append(site.wyckoff)
    return tuple(sorted(tuple(sorted(letters)) for letters in classes.values()))


def test_legacy_sg47_golden_and_promoted_exact_default(monkeypatch) -> None:
    source = _sg47()
    legacy = canonicalize_legacy(source, tolerance=1e-3)
    assert legacy.spacegroup.it_number == 47
    assert _anonymous_pattern(legacy.asu) == (("a",), ("c",))

    monkeypatch.setattr(lift_module, "canonicalize_legacy", lambda *args, **kwargs: pytest.fail("unexpected BFS"))
    promoted = canonicalize(source, symmetry="declared")
    assert isinstance(promoted, ASUStructure)
    assert promoted == _canonical_protostructure_asu(source)
    assert _anonymous_pattern(promoted) == (("a",), ("b",))


def test_declared_default_retains_source_metadata(monkeypatch) -> None:
    timestamp = datetime.datetime(2026, 9, 25, tzinfo=datetime.UTC)
    source = ASUStructure(
        Cell(_sg47().cell.basis, precision=F(1, 1000)),
        47,
        _sg47().wyckoff_sites,
        (
            _species("Na", "Na", labels=("sodium",)),
            _species("O", "O", labels=("oxygen",)),
            _species("unused", "Cl", labels=("unused-chlorine",)),
        ),
        coordinate_precision=F(1, 2000),
        chemical_composition=ChemicalComposition({"Na": 1, "O": 1}, mode="full"),
        chemical_formula_descriptive="NaO",
        chemical_formula_hill="NaO",
        optimization_type="experimental",
        immutable_id="source-id",
        last_modified=timestamp,
        charge=F(2),
    )
    source = _canonical_protostructure_asu(source)
    monkeypatch.setattr(lift_module, "canonicalize_legacy", lambda *args, **kwargs: pytest.fail("unexpected BFS"))
    result = canonicalize(source, symmetry="declared")

    assert result.cell.precision == source.cell.precision
    assert result.coordinate_precision == source.coordinate_precision
    assert result.species == source.species
    assert result.charge == source.charge
    assert result.chemical_composition == source.chemical_composition
    assert result.chemical_formula_descriptive == source.chemical_formula_descriptive
    assert result.chemical_formula_hill == source.chemical_formula_hill
    assert result.optimization_type == source.optimization_type
    assert result.immutable_id == source.immutable_id
    assert result.last_modified == source.last_modified


def test_promoted_and_legacy_apis_are_exported() -> None:
    names = (
        "canonical_asu",
        "canonical_asu_legacy",
        "canonical_bare_protostructure",
        "canonical_bare_prototype",
        "canonical_protostructure",
        "canonical_prototype",
        "canonicalize",
        "canonicalize_legacy",
    )
    for name in names:
        assert getattr(atomistic, name) is getattr(symmetry, name)


def test_measured_default_matches_explicit_anonymous_api_and_differs_from_legacy() -> None:
    pytest.importorskip("spglib")
    source = atomistic.UnitcellStructureView(_sg47())
    kwargs = {"tolerance": 1e-3, "factors": (1,)}

    promoted = atomistic.canonical_asu(source, **kwargs)
    explicit = atomistic.canonical_asu_protostructure(source, **kwargs)
    legacy = atomistic.canonical_asu_legacy(source, **kwargs)

    assert promoted == explicit
    assert _anonymous_pattern(promoted) == (("a",), ("b",))
    assert _anonymous_pattern(legacy) == (("a",), ("c",))


@pytest.mark.extended
def test_sg157_fixed_points_preserve_cell_and_coordinate_precision() -> None:
    pytest.importorskip("spglib")
    fixture = Path(__file__).with_name("fixtures") / "structreading" / "157.cif"
    source = load(str(fixture), repair=True)
    reference = atomistic.canonical_asu(atomistic.UnitcellStructureView(source))
    expected_precision = (reference.cell.precision, reference.coordinate_precision)

    exact_family = _canonical_protostructure_assignments_asu(reference)
    measured_family = atomistic.canonical_asu_protostructure_assignments(reference)
    measured = atomistic.canonical_asu(reference)
    materialized = atomistic.UnitcellStructureView(reference).unview()
    copied = atomistic.canonical_asu(materialized)

    for result in (measured, copied):
        assert result == reference
        assert (result.cell.precision, result.coordinate_precision) == expected_precision
    for family in (exact_family, measured_family):
        assert any(result == reference for result in family)
        fixed = next(result for result in family if result == reference)
        assert (fixed.cell.precision, fixed.coordinate_precision) == expected_precision
