"""Tests for standard-setting, geometry-free protostructures."""

import pickle
from fractions import Fraction
from types import SimpleNamespace

import pytest
from httk.core import FracVector

from httk.atomistic import (
    AnonymousStructure,
    Assembly,
    ASUStructure,
    CartesianSiteMoments,
    ChemicalComposition,
    ChemicalFormulaView,
    CompositionView,
    FundamentalDomainStructure,
    FormulapatternView,
    Protostructure,
    ProtostructureView,
    PrototypeView,
    SettingTransform,
    Spacegroup,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffOccupation,
    WyckoffSite,
)
from httk.atomistic.models.cell.numeric import NumericCell
from httk.atomistic.models.protostructure.backend import ProtostructureBackend
from httk.atomistic.models.protostructure.recognized import RecognizedProtostructure
from httk.atomistic.models.sites.numeric import NumericSites
from httk.atomistic.models.structure.backend import StructureBackend

CELL = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
EMPTY = FracVector(())


def _species() -> tuple[Species, Species]:
    return Species("Na", ("Na",), (1,), charges=(1,), labels=("sodium",)), Species(
        "Cl", ("Cl",), (1,), spins=(1,), labels=("chloride",)
    )


def _rocksalt_unitcell() -> UnitcellStructure:
    sodium, chlorine = _species()
    return UnitcellStructure(
        CELL,
        ([0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]),
        (sodium, chlorine),
        ("Na", "Cl"),
    )


def _rocksalt_asu(*, domain: type[FundamentalDomainStructure] = ASUStructure) -> FundamentalDomainStructure:
    sodium, chlorine = _species()
    return domain(
        CELL,
        225,
        (WyckoffSite("a", EMPTY, "Na"), WyckoffSite("b", EMPTY, "Cl")),
        (sodium, chlorine),
    )


class CountingStructureResolver(StructureBackend):
    def __init__(self, structure: UnitcellStructure | FundamentalDomainStructure) -> None:
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


class CustomProtostructureBackend(ProtostructureBackend):
    def __init__(self) -> None:
        self.spacegroup_calls = 0
        self.occupations_calls = 0

    @property
    def spacegroup(self):
        self.spacegroup_calls += 1
        return Spacegroup.standard(1)

    @property
    def occupations(self):
        self.occupations_calls += 1
        return (WyckoffOccupation("a", "He"),)

    def unwrap(self):
        return self


def test_construction_promotes_inputs_and_canonicalizes() -> None:
    first = Protostructure(225, [("a", "Na"), ("b", "Cl")])
    second = Protostructure(Spacegroup.standard(225), [("b", "Cl"), ("a", "Na")])
    assert first == second
    assert first.occupations[0].species.name == "Cl"
    assert first.multiplicities() == (4, 4)
    assert first.nsites_conventional == 8
    assert {first: "value"}[second] == "value"

    with pytest.raises(ValueError, match="standard setting"):
        Protostructure(Spacegroup.from_setting("15:c1"), [("e", "Na")])
    with pytest.raises(ValueError, match="non-empty"):
        Protostructure(225, [])
    with pytest.raises(ValueError, match="no Wyckoff letter"):
        Protostructure(225, [("zz", "Na")])


def test_same_name_requires_equal_species_and_unknown_symbols_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal Species"):
        Protostructure(
            225,
            [
                ("a", Species("Na", ("Na",), (1,))),
                ("b", Species("Na", ("Na",), (1,), charges=(1,))),
            ],
        )
    with pytest.raises(ValueError, match="unknown.*X"):
        WyckoffOccupation("a", Species("unknown", ("X",), (1,)))
    with pytest.raises(ValueError, match="unknown.*X"):
        WyckoffOccupation("a", Species("attached", ("Na",), (1,), attached=("X",), nattached=(1,)))


def test_disorder_and_vacancy_project_with_tabulated_multiplicity() -> None:
    mixed = Species("mixed", ("Fe", "Ni"), (Fraction(1, 2), Fraction(1, 2)))
    vacancy = Species("vacancy", ("vacancy",), (1,))
    value = Protostructure(221, [("a", mixed), ("b", vacancy)])
    assert CompositionView(value).amounts == (("Fe", Fraction(1, 2)), ("Ni", Fraction(1, 2)))
    assert ChemicalFormulaView(value) == "FeNi"
    assert FormulapatternView(value) == "AB"


def test_exact_paths_preserve_species_and_match_structure_formula() -> None:
    asu = _rocksalt_asu()
    value = ProtostructureView(asu)
    assert value.unwrap() is asu
    assert value.occupations[0].species in asu.species
    assert value.formula == UnitcellStructureView(asu).formula
    assert value.anonymous_formula == UnitcellStructureView(asu).chemical_formula_anonymous
    assert (
        ProtostructureView(
            FundamentalDomainStructure(
                CELL,
                225,
                asu.wyckoff_sites,
                asu.species,
            )
        )
        == value
    )
    with pytest.raises(ValueError):
        ProtostructureView(asu, tolerance=1e-5)
    assert ProtostructureView(value) is value
    with pytest.raises(ValueError):
        ProtostructureView(value, tolerance=1e-5)


def test_transform_scaled_source_uses_standard_conventional_scale() -> None:
    transform = Spacegroup.from_setting("166:R").transform_from_standard
    bi = Species("Bi", ("Bi",), (1,))
    oxygen = Species("O", ("O",), (1,))
    asu = ASUStructure(
        CELL,
        166,
        (WyckoffSite("a", EMPTY, "Bi"), WyckoffSite("b", EMPTY, "O")),
        (bi, oxygen),
        transform=transform,
    )
    source = UnitcellStructureView(asu)
    value = ProtostructureView(asu)
    direct = Protostructure(166, [("a", bi), ("b", oxygen)])

    assert value == direct
    assert CompositionView(value).elements_ratios == source.elements_ratios
    assert CompositionView(value).amounts == (("Bi", Fraction(3)), ("O", Fraction(3)))
    assert CompositionView(source).amounts == (("Bi", Fraction(1)), ("O", Fraction(1)))
    assert value.formula == source.chemical_formula_reduced


def test_recognition_probe_re_raises_errors_from_a_matched_structure_backend() -> None:
    pytest.importorskip("numpy")
    raw = SimpleNamespace(
        cell=NumericCell(CELL),
        sites=NumericSites([[0, 0, 0]]),
        species=(object(),),
        species_at_sites=("invalid",),
    )
    with pytest.raises(TypeError, match="SpeciesBackend"):
        ProtostructureView(raw)


def test_recognition_path_and_rejections() -> None:
    pytest.importorskip("spglib")
    asu = _rocksalt_asu()
    assert ProtostructureView(UnitcellStructureView(asu)) == ProtostructureView(asu)
    with pytest.raises(TypeError):
        ProtostructureView(asu, kind="bogus")

    anonymous = AnonymousStructure(CELL, [[0, 0, 0]], species_at_sites=("A",))
    with pytest.raises(TypeError, match="dummy species"):
        ProtostructureView(anonymous)
    with pytest.raises(TypeError):
        UnitcellStructureView(Protostructure(225, [("a", "Na")]))
    with pytest.raises(TypeError):
        PrototypeView(Protostructure(225, [("a", "Na")]))

    structure = UnitcellStructure(
        CELL,
        [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]],
        _species(),
        ("Na", "Cl"),
        assemblies=(Assembly(((0,),), (1,)),),
    )
    view = ProtostructureView(structure)
    with pytest.raises(ValueError, match="assemblies"):
        _ = view.spacegroup

    chemical = UnitcellStructure(
        CELL,
        [[0, 0, 0]],
        (_species()[0],),
        ("Na",),
        chemical_composition=ChemicalComposition({"Na": 1}),
    )
    view = ProtostructureView(chemical)
    with pytest.raises(ValueError, match="chemical_composition"):
        _ = view.spacegroup

    moments = UnitcellStructure(
        CELL,
        [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]],
        _species(),
        ("Na", "Cl"),
        site_moments=CartesianSiteMoments([[1, 0, 0], [0, 1, 0]]),
    )
    view = ProtostructureView(moments)
    with pytest.raises(ValueError, match="site_moments"):
        _ = view.spacegroup


def test_protostructure_view_resolves_counting_source_once_across_value_operations() -> None:
    source = CountingStructureResolver(_rocksalt_unitcell())
    view = ProtostructureView(source, setting=Spacegroup.standard(1))

    assert source.resolve_calls == 0
    assert view.unwrap() is source
    assert source.resolve_calls == 0

    _ = view.spacegroup
    assert source.resolve_calls == 1
    _ = view.occupations
    same = view
    _ = view == same
    _ = hash(view)
    _ = repr(view)
    _ = view.unview()
    assert source.resolve_calls == 1


def test_protostructure_view_rejects_options_at_deferred_asu_boundary() -> None:
    asu = _rocksalt_asu()
    backend = RecognizedProtostructure(asu, setting=Spacegroup.standard(225))
    with pytest.raises(ValueError, match="recognition arguments cannot be used with an existing ASU"):
        _ = backend.spacegroup

    source = CountingStructureResolver(_rocksalt_unitcell())
    inner = __import__("httk.atomistic.models.structure.asu_view", fromlist=["ASUStructureView"]).ASUStructureView(
        source, setting=Spacegroup.standard(1)
    )
    outer = UnitcellStructureView(inner)
    with pytest.raises(ValueError, match="recognition arguments cannot be used with an existing ASU"):
        ProtostructureView(outer, tolerance=1e-5)
    assert source.resolve_calls == 0


def test_protostructure_view_retains_and_passes_all_recognition_options(monkeypatch: pytest.MonkeyPatch) -> None:
    module = __import__("httk.atomistic.models.protostructure.recognized", fromlist=["recognize_asu"])
    captured: dict[str, object] = {}
    standard = Spacegroup.standard(225)
    transform = SettingTransform.identity()

    def fake_recognize(source: object, **options: object) -> ASUStructure:
        captured.update(options)
        return _rocksalt_asu()

    monkeypatch.setattr(module, "recognize_asu", fake_recognize)
    view = ProtostructureView(
        _rocksalt_unitcell(),
        standard=standard,
        transform=transform,
        tolerance=0.125,
        limit_denominator=17,
    )
    assert captured == {}
    _ = view.spacegroup
    assert captured == {
        "setting": None,
        "standard": standard,
        "transform": transform,
        "tolerance": 0.125,
        "limit_denominator": 17,
    }


def test_protostructure_unsupported_data_fails_atomically_on_first_access() -> None:
    unknown = UnitcellStructure(CELL, [[0, 0, 0]], [Species("unknown", ("X",), (1,))], ("unknown",))
    view = ProtostructureView(unknown)
    with pytest.raises(ValueError, match="unknown symbol 'X'"):
        _ = view.spacegroup
    assert view._resolved_protostructure is None
    assert "_spacegroup" not in view.__dict__
    assert "_derived" not in view._backend.__dict__


def test_protostructure_view_pickle_preserves_unresolved_and_resolved_states() -> None:
    unresolved = ProtostructureView(CountingStructureResolver(_rocksalt_unitcell()), setting=Spacegroup.standard(1))
    restored = pickle.loads(pickle.dumps(unresolved))
    assert restored._resolved_protostructure is None
    assert restored._setting == Spacegroup.standard(1)
    assert restored._backend._structure.resolve_calls == 0
    assert restored.spacegroup.it_number == 1
    assert restored._backend._structure.resolve_calls == 1

    resolved = ProtostructureView(CountingStructureResolver(_rocksalt_unitcell()), setting=Spacegroup.standard(1))
    _ = resolved.spacegroup
    restored = pickle.loads(pickle.dumps(resolved))
    assert restored._resolved_protostructure is not None
    assert restored.unview() is restored._resolved_protostructure
    assert restored._backend._structure.resolve_calls == 1


def test_protostructure_view_accepts_custom_backend_and_native_identity() -> None:
    custom = CustomProtostructureBackend()
    view = ProtostructureView(custom)
    assert custom.spacegroup_calls == 0
    assert view.unwrap() is custom
    assert custom.spacegroup_calls == 0
    value = view.unview()
    assert type(value) is Protostructure
    assert custom.spacegroup_calls == 1
    assert custom.occupations_calls == 1

    native = Protostructure(225, [("a", "Na")])
    assert ProtostructureView(native).unview() is native


def test_protostructure_datastream_path_is_not_parsed_at_construction(tmp_path, monkeypatch) -> None:
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
    view = ProtostructureView(str(path), setting=Spacegroup.standard(1))
    assert calls == 0
    assert view.unwrap() == str(path)
    assert calls == 0
