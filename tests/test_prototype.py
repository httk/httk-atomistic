"""Tests for standard-setting prototypes."""

import pickle
from fractions import Fraction
from types import SimpleNamespace

import pytest
from httk.core import FracVector

from httk.atomistic import (
    AnonymousStructure,
    AnonymousStructureView,
    ASUStructure,
    ASUStructureView,
    ChemicalFormulaView,
    CompositionView,
    Formulapattern,
    FormulapatternView,
    Prototype,
    PrototypeView,
    Spacegroup,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
)
from httk.atomistic.models.structure.backend import StructureBackend

CELL = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
EMPTY = FracVector(())


def _rocksalt_asu() -> ASUStructure:
    return ASUStructure(
        CELL,
        225,
        (WyckoffSite("a", EMPTY, "Na"), WyckoffSite("b", EMPTY, "Cl")),
        (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
    )


def _rocksalt_unitcell() -> UnitcellStructure:
    return UnitcellStructure(
        CELL,
        [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]],
        (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
        ("Na", "Cl"),
    )


class CountingStructureResolver(StructureBackend):
    def __init__(self, structure: UnitcellStructure) -> None:
        self.structure = structure
        self.resolve_calls = 0

    @property
    def cell(self):
        return self.structure.cell

    @property
    def sites(self):
        return self.structure.sites

    @property
    def species(self):
        return self.structure.species

    @property
    def species_at_sites(self):
        return self.structure.species_at_sites

    def resolve(self):
        self.resolve_calls += 1
        return self.structure

    def unwrap(self):
        return self


def test_prototype_standard_setting_and_canonical_site_order() -> None:
    species = (Species("A", ("X",), (1,), labels=("A",)), Species("B", ("X",), (1,), labels=("B",)))
    first = Prototype(CELL, 225, (WyckoffSite("b", EMPTY, "B"), WyckoffSite("a", EMPTY, "A")), species)
    second = Prototype(CELL, 225, tuple(reversed(first.wyckoff_sites)), species)
    assert first == second
    with pytest.raises(ValueError, match="standard setting"):
        Prototype(CELL, Spacegroup.from_setting("15:c1"), (WyckoffSite("e", FracVector([1, 3]), "A"),), (species[0],))


def test_prototype_rejects_representatives_moments_and_bad_free_count() -> None:
    with pytest.raises(ValueError, match="representative"):
        Prototype(CELL, 225, (WyckoffSite("a", EMPTY, "A", representative=FracVector([0, 0, 0])),), (dummy("A"),))
    with pytest.raises(ValueError, match="free parameter"):
        Prototype(CELL, 225, (WyckoffSite("a", FracVector([1]), "A"),), (dummy("A"),))


def dummy(label: str) -> Species:
    return Species(label, ("X",), (1,), labels=(label,))


def test_exact_asu_path_and_expansion() -> None:
    asu = _rocksalt_asu()
    prototype = PrototypeView(asu)
    assert PrototypeView(asu, kind="structure") == prototype
    with pytest.raises(TypeError):
        PrototypeView(asu, kind="bogus")
    assert prototype.spacegroup.is_standard_setting
    assert prototype.multiplicities() == (4, 4)
    assert len(AnonymousStructureView(prototype).sites) == prototype.nsites_conventional
    assert prototype.anonymous_formula == "AB"
    assert isinstance(FormulapatternView(prototype), str)
    with pytest.raises(ValueError):
        PrototypeView(asu, tolerance=1e-5)
    with pytest.raises(ValueError):
        CompositionView(prototype)
    with pytest.raises(ValueError):
        ChemicalFormulaView(prototype)
    with pytest.raises(TypeError):
        ChemicalFormulaView(prototype, kind="bogus")


def test_prototype_view_rewrap_rejects_arguments() -> None:
    view = PrototypeView(_rocksalt_asu())
    with pytest.raises(ValueError):
        PrototypeView(view, tolerance=0.123)
    with pytest.raises(ValueError):
        PrototypeView(view, setting="bogus")
    with pytest.raises(ValueError):
        PrototypeView(view, kind="bogus")


def test_anonymous_prototype_source_uses_original_labels() -> None:
    anonymous = AnonymousStructure(
        CELL, [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]], species_at_sites=("A", "B")
    )
    pytest.importorskip("spglib")
    prototype = PrototypeView(anonymous)
    assert prototype.anonymous_formula == AnonymousStructureView(anonymous).anonymous_formula


def test_formula_views_reduce_non_coprime_amounts_and_remain_parseable() -> None:
    asu = _rocksalt_asu()
    prototype = PrototypeView(asu)
    assert Formulapattern(str(prototype.anonymous_formula)) == "AB"

    real_structure = UnitcellStructureView(asu)
    assert Formulapattern(str(FormulapatternView(real_structure))) == "AB"
    assert ChemicalFormulaView(real_structure) == "ClNa"


def test_anonymous_formula_cross_consistency_for_non_coprime_counts() -> None:
    asu = _rocksalt_asu()
    assert AnonymousStructureView(asu).anonymous_formula == UnitcellStructureView(asu).chemical_formula_anonymous

    two_to_two = UnitcellStructureView(
        UnitcellStructure(
            CELL,
            [
                [0, 0, 0],
                [0, 0, 0],
                [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
                [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
            ],
            (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
            ("Na", "Na", "Cl", "Cl"),
        )
    )
    assert AnonymousStructureView(two_to_two).anonymous_formula == two_to_two.chemical_formula_anonymous


def test_recognition_path_is_spglib_gated() -> None:
    pytest.importorskip("spglib")
    asu = _rocksalt_asu()
    unitcell = UnitcellStructureView(asu)
    assert PrototypeView(unitcell) == PrototypeView(asu)


def test_prototype_view_resolves_nested_asu_source_once_across_value_operations() -> None:
    source = CountingStructureResolver(_rocksalt_unitcell())
    asu = ASUStructureView(source, setting=Spacegroup.standard(1))
    view = PrototypeView(asu)

    assert source.resolve_calls == 0
    assert view.unwrap() is source
    assert source.resolve_calls == 0

    _ = view.spacegroup
    assert source.resolve_calls == 1
    _ = view.wyckoff_sites
    same = view
    _ = view == same
    _ = repr(view)
    _ = view.unview()
    assert source.resolve_calls == 1


def test_prototype_view_retains_tolerance_and_denominator_and_resolves_source_once(monkeypatch) -> None:
    module = __import__("httk.atomistic.models.crystalpattern.fundamental_view", fromlist=["conventional_cell"])
    source = CountingStructureResolver(_rocksalt_unitcell())
    captured: dict[str, object] = {}

    def fake_conventional(structure: object, **options: object):
        captured["structure"] = structure
        captured.update(options)
        return SimpleNamespace(asu=_rocksalt_asu())

    monkeypatch.setattr(module, "conventional_cell", fake_conventional)
    view = PrototypeView(source, tolerance=0.125, limit_denominator=17)
    assert source.resolve_calls == 0
    _ = view.spacegroup
    assert source.resolve_calls == 1
    assert captured == {"structure": source.structure, "tolerance": 0.125, "limit_denominator": 17}


def test_prototype_view_unsupported_data_fails_atomically_on_first_access() -> None:
    mixed = UnitcellStructure(
        CELL,
        [[0, 0, 0]],
        [Species("mixed", ("Fe", "Ni"), (Fraction(1, 2), Fraction(1, 2)))],
        ("mixed",),
    )
    view = PrototypeView(mixed)
    with pytest.raises(ValueError, match="not a single real element"):
        _ = view.spacegroup
    assert view._resolved_prototype is None
    assert "_cell" not in view.__dict__
    assert "_derived" not in view._backend.__dict__


def test_prototype_view_pickle_preserves_unresolved_and_resolved_states() -> None:
    source = CountingStructureResolver(_rocksalt_unitcell())
    unresolved = PrototypeView(ASUStructureView(source, setting=Spacegroup.standard(1)))
    restored = pickle.loads(pickle.dumps(unresolved))
    restored_source = restored._backend._structure._source_backend
    assert restored._resolved_prototype is None
    assert restored._tolerance is None
    assert restored_source.resolve_calls == 0
    assert restored.unwrap() is restored_source
    assert restored_source.resolve_calls == 0
    assert restored.spacegroup.it_number == 1
    assert restored_source.resolve_calls == 1

    source = CountingStructureResolver(_rocksalt_unitcell())
    resolved = PrototypeView(ASUStructureView(source, setting=Spacegroup.standard(1)))
    _ = resolved.spacegroup
    restored = pickle.loads(pickle.dumps(resolved))
    assert restored._resolved_prototype is not None
    assert restored.unview() is restored._resolved_prototype
    assert restored._backend._structure._source_backend.resolve_calls == 1


def test_prototype_view_native_unview_preserves_identity() -> None:
    native = Prototype(
        CELL,
        225,
        (WyckoffSite("a", EMPTY, "A"),),
        (dummy("A"),),
    )
    assert PrototypeView(native).unview() is native


def test_prototype_datastream_path_is_not_parsed_at_construction(tmp_path, monkeypatch) -> None:
    import httk.core

    path = tmp_path / "source.cif"
    path.write_text("not parsed", encoding="utf-8")
    calls = 0
    real_load = httk.core.load

    def counted_load(filename: str):
        nonlocal calls
        calls += 1
        return real_load(filename)

    monkeypatch.setattr(httk.core, "load", counted_load)
    view = PrototypeView(str(path))
    assert calls == 0
    assert view.unwrap() == str(path)
    assert calls == 0
