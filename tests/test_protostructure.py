"""Tests for standard-setting assigned-species classification keys."""

import datetime
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
    Cell,
    ChemicalComposition,
    ChemicalFormulaView,
    CompositionView,
    FormulatypeView,
    FundamentalDomainStructure,
    Protostructure,
    ProtostructureBackend,
    ProtostructureLabel,
    ProtostructureRecord,
    ProtostructureView,
    PrototypeView,
    RecognizedProtostructure,
    SettingTransform,
    Spacegroup,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffOccupation,
    WyckoffSite,
)
from httk.atomistic.models.cell.numeric import NumericCell
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


class IdentityCarryingProtostructureBackend(ProtostructureBackend):
    """Expose optional class identity through a non-value protostructure backend."""

    def __init__(self, value: Protostructure) -> None:
        self.value = value

    @property
    def spacegroup(self):
        return self.value.spacegroup

    @property
    def occupations(self):
        return self.value.occupations

    @property
    def representative(self):
        return self.value.representative

    @property
    def discriminator(self):
        return self.value.discriminator

    def unwrap(self):
        return self.value


def test_base_only_and_representative_only_are_valid() -> None:
    representative = _rocksalt_asu()
    derived = Protostructure(representative=representative)
    explicit = Protostructure(225, [("a", representative.species[0]), ("b", representative.species[1])])
    assert derived.spacegroup == explicit.spacegroup
    assert derived.occupations == explicit.occupations
    assert derived.similar(explicit, 0.0)
    assert derived.similar(derived, 0.0)
    assert derived.representative == representative
    assert derived.discriminator is None
    assert Protostructure(225, [("a", "Na"), ("b", "Cl")], discriminator="001").discriminator == "001"


def test_representative_validation_and_mismatch() -> None:
    with pytest.raises(ValueError, match="disagrees with its representative"):
        Protostructure(225, [("a", "Na")], representative=_rocksalt_asu())
    with pytest.raises(ValueError, match="non-empty string"):
        Protostructure(225, [("a", "Na")], discriminator="")
    asu = _rocksalt_asu()
    molecular = FundamentalDomainStructure(
        CELL,
        225,
        asu.wyckoff_sites,
        asu.species,
        molecular=True,
    )
    with pytest.raises(ValueError, match="cannot be molecular"):
        Protostructure(representative=molecular)
    with pytest.raises(ValueError, match="molecular"):
        ProtostructureView(molecular).unview()


def test_exact_equality_includes_optional_information_and_is_hashable() -> None:
    representative = _rocksalt_asu()
    assert Protostructure(representative=representative) == Protostructure(representative=representative)
    assert Protostructure(225, [("a", "Na"), ("b", "Cl")], discriminator="001") != Protostructure(
        225, [("a", "Na"), ("b", "Cl")], discriminator="002"
    )
    # Hashable over the base identity (space group, occupations, discriminator): equal values
    # hash equal and work as dict keys; a retained representative is excluded from the hash.
    first = Protostructure(225, [("a", "Na"), ("b", "Cl")], discriminator="001")
    same = Protostructure(225, [("b", "Cl"), ("a", "Na")], discriminator="001")
    assert hash(first) == hash(same)
    assert {first: "value"}[same] == "value"
    assert isinstance(hash(Protostructure(representative=representative)), int)


def test_similar_optional_fields_and_delta_validation() -> None:
    one = Protostructure(225, [("a", "Na"), ("b", "Cl")])
    two = Protostructure(225, [("a", "Na"), ("b", "Cl")], discriminator="001")
    assert one.similar(two, 0.0)
    assert not two.similar(Protostructure(225, [("a", "Na"), ("b", "Cl")], discriminator="002"), 0.0)
    # Both base (the recognition default): comparison is discrete over space group, occupations,
    # and discriminator only. No geometry is consulted, so two recognitions of geometrically
    # different structures compare similar even at a zero budget.
    tight = ProtostructureView(_rocksalt_asu()).unview()
    stretched = ProtostructureView(
        ASUStructure(
            [[Fraction(51, 10), 0, 0], [0, Fraction(51, 10), 0], [0, 0, Fraction(51, 10)]],
            225,
            (WyckoffSite("a", EMPTY, "Na"), WyckoffSite("b", EMPTY, "Cl")),
            _species(),
        )
    ).unview()
    assert tight.representative is None and stretched.representative is None
    assert tight.similar(stretched, 0.0)
    with pytest.raises(ValueError):
        one.similar(one, -1)
    with pytest.raises(ValueError):
        one.similar(one, float("inf"))
    with pytest.raises(TypeError):
        one.similar(one, object())


def test_similar_passes_fundamental_domain_representatives_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    representative = _rocksalt_asu(domain=FundamentalDomainStructure)
    one = Protostructure(representative=representative)
    two = Protostructure(representative=representative)
    received: list[FundamentalDomainStructure] = []

    def fake_structure_delta(first: FundamentalDomainStructure, second: FundamentalDomainStructure) -> float:
        received.extend((first, second))
        return 0.0

    paths = __import__("httk.atomistic.symmetry.paths", fromlist=["structure_delta"])
    monkeypatch.setattr(paths, "structure_delta", fake_structure_delta)

    assert one.similar(two, 0.0)
    assert received == [representative, representative]


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
    assert hash(first) == hash(second)
    assert {first: "value"}[second] == "value"
    assert first.similar(second, 0.0)

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
    assert FormulatypeView(value) == "AB"


def test_exact_paths_preserve_species_and_match_structure_formula() -> None:
    asu = _rocksalt_asu()
    value = ProtostructureView(asu)
    assert value.unwrap() is asu
    assert value.occupations[0].species in asu.species
    assert value.formula == UnitcellStructureView(asu).formula
    assert value.anonymous_formula == UnitcellStructureView(asu).chemical_formula_anonymous
    other = ProtostructureView(FundamentalDomainStructure(CELL, 225, asu.wyckoff_sites, asu.species))
    assert other.spacegroup == value.spacegroup
    assert other.occupations == value.occupations
    assert other.representative is None  # recognition returns a base value
    assert other.similar(value, 0.0)
    with pytest.raises(ValueError):
        ProtostructureView(asu, tolerance=1e-5)
    assert ProtostructureView(value) is value
    with pytest.raises(ValueError):
        ProtostructureView(value, tolerance=1e-5)


def test_protostructure_views_and_labels_preserve_optional_identity() -> None:
    representative = _rocksalt_asu()
    value = Protostructure(representative=representative, discriminator="003")

    recognized = RecognizedProtostructure(representative)
    view = ProtostructureView(recognized)
    assert view._resolved_protostructure is None
    base = view.unview()
    assert base.representative is None and base.discriminator is None  # recognition returns a base value
    assert ProtostructureLabel(recognized).unview() == base

    generic_backend = IdentityCarryingProtostructureBackend(value)
    assert ProtostructureView(generic_backend).unview() == value
    label = ProtostructureLabel(generic_backend)
    assert str(label) == "AB_cF8_225_a_b:Na-Cl"
    assert label.unview() == value


def test_transform_scaled_source_uses_standard_conventional_scale() -> None:
    transform = Spacegroup.from_setting("166:R").transform_from_standard
    bi = Species("Bi", ("Bi",), (1,))
    oxygen = Species("O", ("O",), (1,))
    timestamp = datetime.datetime(2026, 8, 23, tzinfo=datetime.UTC)
    asu = ASUStructure(
        Cell(CELL, precision=Fraction(1, 1_000)),
        166,
        (WyckoffSite("a", EMPTY, "Bi"), WyckoffSite("b", EMPTY, "O")),
        (bi, oxygen),
        transform=transform,
        coordinate_precision=Fraction(1, 10_000),
        chemical_formula_descriptive="BiO",
        chemical_formula_hill="BiO",
        optimization_type="local",
        immutable_id="source-1",
        last_modified=timestamp,
        charge=1,
    )
    source = UnitcellStructureView(asu)
    value = ProtostructureView(asu)
    direct = Protostructure(166, [("a", bi), ("b", oxygen)])

    assert value.spacegroup == direct.spacegroup
    assert value.occupations == direct.occupations
    # Recognition returns a base value: composition/formula use the standard conventional-cell
    # scale independently of the volume-scaled source, and no representative is attached.
    assert value.representative is None
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
    assert PrototypeView(Protostructure(225, [("a", "Na")])).unview().occupations[0].label == "A"

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
    assert hash(view) == hash(view.unview())
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


def test_protostructure_label_golden_and_aflow_divergence() -> None:
    calcite = Protostructure(167, [("a", "Ca"), ("b", "C"), ("e", "O")])
    assert str(calcite.label) == "ABC3_hR10_167_a_b_e:Ca-C-O"
    # AFLOW orders classes alphabetically by element, so both prefix and suffix reorder.
    assert calcite.aflow_label == "ABC3_hR10_167_b_a_e:C-Ca-O"


def test_protostructure_label_round_trips_element_pure_value() -> None:
    # Element-pure means Species(name, (name,), (1,)) exactly, as the parser rebuilds.
    pure = [(letter, Species(name, (name,), (1,))) for letter, name in (("a", "Ca"), ("b", "C"), ("e", "O"))]
    calcite = Protostructure(167, pure)
    assert ProtostructureView(str(calcite.label)) == calcite


def test_protostructure_label_string_dispatch() -> None:
    view = ProtostructureView("AB_cF8_225_a_b:Na-Cl")
    assert view.spacegroup == Spacegroup.standard(225)
    assert str(ProtostructureView("AB_cF8_225_a_b:Na-Cl").label) == "AB_cF8_225_a_b:Na-Cl"


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


def test_representative_rejections_ported_from_structuretype() -> None:
    # Ported from the retired structuretype suite: _validate_representative rejections that
    # now guard Protostructure's optional geometrical representative.
    from httk.atomistic.models.moments.collinear import CollinearSiteMoments

    non_standard = ASUStructure(
        CELL,
        Spacegroup.from_setting("15:c1"),
        (WyckoffSite("a", EMPTY, "Na"),),
        (Species("Na", ("Na",), (1,)),),
    )
    with pytest.raises(ValueError, match="IT standard setting"):
        Protostructure(representative=non_standard)

    transform = SettingTransform(FracVector.eye((3, 3)), (Fraction(1, 2), 0, 0))
    with_transform = ASUStructure(
        CELL,
        225,
        (WyckoffSite("a", EMPTY, "Na"), WyckoffSite("b", EMPTY, "Cl")),
        _species(),
        transform=transform,
    )
    with pytest.raises(ValueError, match="identity setting transform"):
        Protostructure(representative=with_transform)

    moments = ASUStructure(
        CELL,
        225,
        (
            WyckoffSite("a", EMPTY, "Na", moment=CollinearSiteMoments((1,))),
            WyckoffSite("b", EMPTY, "Cl", moment=CollinearSiteMoments((0,))),
        ),
        _species(),
    )
    with pytest.raises(ValueError, match="site moments"):
        Protostructure(representative=moments)

    assemblies = ASUStructure(
        CELL,
        221,
        (WyckoffSite("a", EMPTY, "Na"),),
        (Species("Na", ("Na",), (1,)),),
        assemblies=(Assembly(((0,),), (1,)),),
    )
    with pytest.raises(ValueError, match="assemblies"):
        Protostructure(representative=assemblies)


def test_lazy_erasure_arrows_recover_source_and_stay_lazy_through_pickle() -> None:
    # Ported from the retired structuretype suite: erasure arrows off a discriminator-carrying
    # source resolve lazily, recover the source via unwrap, and stay lazy across pickling.
    source = Protostructure(representative=_rocksalt_asu(), discriminator="001")

    prototype_view = PrototypeView(source)
    assert prototype_view._resolved_prototype is None
    assert prototype_view.unwrap() is source
    restored = pickle.loads(pickle.dumps(prototype_view))
    assert restored._resolved_prototype is None
    assert restored.unview().discriminator == "001"

    protostructure_view = ProtostructureView(source)
    assert protostructure_view._resolved_protostructure is None
    assert protostructure_view.unwrap() is source
    restored_proto = pickle.loads(pickle.dumps(protostructure_view))
    assert restored_proto._resolved_protostructure is None
    assert restored_proto.unview() == source


def test_protostructure_view_resolves_storage_record() -> None:
    from httk.core.storage import resolve_storage_record

    view = ProtostructureView(Protostructure(225, [("a", "Na")]))
    assert resolve_storage_record(view) is ProtostructureRecord


def test_similar_is_available_on_every_protostructure_backend() -> None:
    # Regression (P4): the similar body lived on the value class, so RecognizedProtostructure,
    # ProtostructureLabelString, and user backends raised NotImplementedError. It now lives on
    # ProtostructureAPI and is callable from every source.
    asu = _rocksalt_asu()
    view = ProtostructureView(asu)
    assert view.similar(ProtostructureView(asu), 0.0)

    recognized = RecognizedProtostructure(asu)
    assert recognized.similar(recognized, 0.0)

    label_view = ProtostructureView("AB_cF8_225_a_b:Na-Cl")
    assert label_view.similar(ProtostructureView("AB_cF8_225_a_b:Na-Cl"), 0.0)
    assert label_view._backend.similar(label_view._backend, 0.0)

    generic = IdentityCarryingProtostructureBackend(Protostructure(225, [("a", "Na"), ("b", "Cl")]))
    assert generic.similar(Protostructure(225, [("a", "Na"), ("b", "Cl")]), 0.0)
