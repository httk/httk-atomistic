"""Tests for the geometry-free and geometrical-class taxonomy storage records."""

from fractions import Fraction

import pytest
from httk.core import FracVector
from httk.core.storage import content_id, storage_identity_name

from httk.atomistic import (
    ASUStructure,
    Cell,
    FundamentalDomainPattern,
    FundamentalDomainPatternRecord,
    Protopattern,
    ProtopatternRecord,
    Protostructure,
    ProtostructureRecord,
    Prototype,
    PrototypeRecord,
    PrototypeView,
    Spacegroup,
    Species,
    Structuretype,
    StructuretypeRecord,
    StructuretypeView,
    WyckoffOccupationRecord,
    WyckoffSite,
)
from httk.atomistic.entries.prototypes import (
    ProtopatternEntry,
    ProtostructureEntry,
    PrototypeEntry,
    StructuretypeEntry,
)
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.storage.records import (
    _fundamental_domain_pattern_from_record,
    _fundamental_domain_pattern_record_from_value,
    _protopattern_from_record,
    _protopattern_record_from_value,
    _protostructure_from_record,
    _protostructure_record_from_value,
    _prototype_from_record,
    _prototype_record_from_value,
    _structuretype_from_record,
    _structuretype_record_from_value,
)

EMPTY: tuple[Fraction, ...] = ()
CELL = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]


def _rocksalt() -> Protostructure:
    return Protostructure(225, [("a", "Na"), ("b", "Cl")])


def _rocksalt_asu() -> ASUStructure:
    return ASUStructure(
        CELL,
        225,
        (WyckoffSite("a", EMPTY, "Na"), WyckoffSite("b", EMPTY, "Cl")),
        (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
    )


def _rocksalt_protopattern() -> Protopattern:
    return Protopattern(225, [("a", "A"), ("b", "B")])


def _hexagonal_fundamental_domain_pattern() -> FundamentalDomainPattern:
    dummy = Species("A", ("X",), (1,), labels=("A",))
    return FundamentalDomainPattern(
        Cell(CellParams((2, 2, 3, 90, 90, 120)).basis),
        191,
        (WyckoffSite("a", EMPTY, "A"),),
        (dummy,),
    )


# --- record identity names ---


def test_record_identity_names_are_storage_names() -> None:
    for record_type in (
        ProtostructureRecord,
        ProtopatternRecord,
        PrototypeRecord,
        StructuretypeRecord,
        FundamentalDomainPatternRecord,
        WyckoffOccupationRecord,
    ):
        identity_name = storage_identity_name(record_type)
        assert identity_name == record_type.__httk_storage__.storage_name
        assert "httk.atomistic" not in identity_name


def test_new_record_storage_names_are_suffix_free() -> None:
    assert ProtopatternRecord.__httk_storage__.storage_name == "atomistic_protopattern"
    assert PrototypeRecord.__httk_storage__.storage_name == "atomistic_prototype"
    assert StructuretypeRecord.__httk_storage__.storage_name == "atomistic_structuretype"
    assert FundamentalDomainPatternRecord.__httk_storage__.storage_name == "atomistic_fundamental_domain_pattern"


def test_taxonomy_record_indexes_cover_it_number_and_label() -> None:
    for record_type in (ProtostructureRecord, ProtopatternRecord, PrototypeRecord, StructuretypeRecord):
        assert record_type.__httk_storage__.indexes == (("spacegroup_it_number",), ("label",))


# --- protostructure (unchanged) ---


def test_protostructure_record_label_format_is_deterministic() -> None:
    record = _protostructure_record_from_value(_rocksalt())
    assert record.label == "AB_cF8_225_a_b:Na-Cl"


def test_protostructure_golden_content_id_is_layout_independent() -> None:
    record = _protostructure_record_from_value(_rocksalt())
    assert record.id == content_id(_rocksalt()) == "743eee98965ea3a28e00db23b8211d1d41659dbc1a8db2a0b34ace3f298896b5"


def test_equal_protostructures_share_content_id_including_permuted_species_order() -> None:
    first = Protostructure(225, [("a", "Na"), ("b", "Cl")])
    permuted = Protostructure(Spacegroup.standard(225), [("b", "Cl"), ("a", "Na")])
    assert first == permuted
    assert _protostructure_record_from_value(first).id == _protostructure_record_from_value(permuted).id


def test_disordered_species_protostructure_round_trips() -> None:
    mixed = Species("mixed", ("Fe", "Ni"), (Fraction(1, 2), Fraction(1, 2)))
    value = Protostructure(221, [("a", mixed), ("b", "Cl")])
    record = _protostructure_record_from_value(value)
    assert _protostructure_from_record(record) == value


def test_protostructure_record_rejects_out_of_range_it_number() -> None:
    good = _protostructure_record_from_value(_rocksalt())
    with pytest.raises(ValueError, match="in .1, 230."):
        ProtostructureRecord(
            spacegroup_it_number=231,
            spacegroup_hall_entry=good.spacegroup_hall_entry,
            occupations=good.occupations,
        )


# --- protopattern record ---


def test_protopattern_record_round_trips_and_labels() -> None:
    value = _rocksalt_protopattern()
    record = _protopattern_record_from_value(value)
    assert record.label == "AB_cF8_225_a_b"
    assert _protopattern_from_record(record) == value


def test_protopattern_golden_content_id_is_layout_independent() -> None:
    record = _protopattern_record_from_value(_rocksalt_protopattern())
    assert record.id == content_id(_rocksalt_protopattern())
    assert record.id == "1e3f5da5dbd250cc75e2939415dc1f5c1061fcd54a3b8509e18e9f11b1cd754a"


# --- fundamental-domain-pattern record (the renamed geometric record) ---


def test_fundamental_domain_pattern_record_round_trips_surd_cell_and_free_parameters() -> None:
    value = _hexagonal_fundamental_domain_pattern()
    record = _fundamental_domain_pattern_record_from_value(value)
    rebuilt = _fundamental_domain_pattern_from_record(record)
    assert rebuilt == value
    assert record.cell.basis  # surd basis retained as exact scalars


def test_fundamental_domain_pattern_golden_content_id_is_layout_independent() -> None:
    record = _fundamental_domain_pattern_record_from_value(_hexagonal_fundamental_domain_pattern())
    assert record.id == content_id(_hexagonal_fundamental_domain_pattern())
    assert record.id == "88b2a932cf8f4d83120b0eb1bee26240e1daac934e9252f493fc3f87a4994c8f"


# --- prototype record (new geometrical-class meaning) ---


def test_prototype_record_round_trips_representative_carrying() -> None:
    value = PrototypeView(_rocksalt_asu()).unview()
    record = _prototype_record_from_value(value)
    assert record.label == "AB_cF8_225_a_b"
    assert record.representative is not None
    assert record.discriminator is None
    assert _prototype_from_record(record) == value


def test_prototype_record_round_trips_discriminator_only() -> None:
    value = Prototype(_rocksalt_protopattern(), discriminator="001")
    record = _prototype_record_from_value(value)
    assert record.representative is None
    assert record.discriminator == "001"
    assert _prototype_from_record(record) == value


def test_prototype_record_requires_a_class_distinction() -> None:
    good = _prototype_record_from_value(Prototype(_rocksalt_protopattern(), discriminator="001"))
    with pytest.raises(ValueError, match="at least one of representative or discriminator"):
        PrototypeRecord(
            spacegroup_it_number=good.spacegroup_it_number,
            spacegroup_hall_entry=good.spacegroup_hall_entry,
            wyckoff_letters=good.wyckoff_letters,
            labels=good.labels,
        )


def test_prototype_golden_content_ids_are_layout_independent() -> None:
    representative_carrying = PrototypeView(_rocksalt_asu()).unview()
    rep_record = _prototype_record_from_value(representative_carrying)
    assert rep_record.id == content_id(representative_carrying)
    assert rep_record.id == "b30718b654cf19dc55cc6d065e361dff27c412d112288b634a9b963b99f897c5"

    discriminator_only = Prototype(_rocksalt_protopattern(), discriminator="001")
    disc_record = _prototype_record_from_value(discriminator_only)
    assert disc_record.id == content_id(discriminator_only)
    assert disc_record.id == "3ee6b735605d2804d0cfc58bc112a7696d031ed3b3ef267ccfb1fd45992da5a1"


# --- structuretype record ---


def test_structuretype_record_round_trips_representative_carrying() -> None:
    value = StructuretypeView(_rocksalt_asu()).unview()
    record = _structuretype_record_from_value(value)
    assert record.label == "AB_cF8_225_a_b:Na-Cl"
    assert record.representative is not None
    assert record.discriminator is None
    assert _structuretype_from_record(record) == value


def test_structuretype_record_round_trips_discriminator_only() -> None:
    protostructure = Protostructure(225, [("a", Species("Na", ("Na",), (1,))), ("b", Species("Cl", ("Cl",), (1,)))])
    value = Structuretype(protostructure, discriminator="001")
    record = _structuretype_record_from_value(value)
    assert record.representative is None
    assert record.discriminator == "001"
    assert _structuretype_from_record(record) == value


def test_structuretype_record_requires_a_class_distinction() -> None:
    good = _structuretype_record_from_value(
        Structuretype(
            Protostructure(225, [("a", Species("Na", ("Na",), (1,))), ("b", Species("Cl", ("Cl",), (1,)))]),
            discriminator="001",
        )
    )
    with pytest.raises(ValueError, match="at least one of representative or discriminator"):
        StructuretypeRecord(
            spacegroup_it_number=good.spacegroup_it_number,
            spacegroup_hall_entry=good.spacegroup_hall_entry,
            occupations=good.occupations,
        )


def test_structuretype_golden_content_ids_are_layout_independent() -> None:
    representative_carrying = StructuretypeView(_rocksalt_asu()).unview()
    rep_record = _structuretype_record_from_value(representative_carrying)
    assert rep_record.id == content_id(representative_carrying)
    assert rep_record.id == "7095708d994065dffe01bfe32d5382cad0aea827438a6d69a3775823778bb5dd"

    protostructure = Protostructure(225, [("a", Species("Na", ("Na",), (1,))), ("b", Species("Cl", ("Cl",), (1,)))])
    discriminator_only = Structuretype(protostructure, discriminator="001")
    disc_record = _structuretype_record_from_value(discriminator_only)
    assert disc_record.id == content_id(discriminator_only)
    assert disc_record.id == "67b214c92957cd65c840a3995cbec6e1b696bf1dd54736af9c35462a888d21eb"


# --- occupation record guard (unchanged) ---


def test_wyckoff_occupation_record_rejects_raw_species() -> None:
    with pytest.raises(TypeError, match="SpeciesRecord"):
        WyckoffOccupationRecord(wyckoff="a", species=Species("Na", ("Na",), (1,)))  # type: ignore[arg-type]


# --- SQL store ---


def test_sql_store_protostructure_dedup_and_it_number_filter() -> None:
    pytest.importorskip("sqlalchemy")
    from httk.store import Backend, SqlStore

    first = _protostructure_record_from_value(Protostructure(225, [("a", "Na"), ("b", "Cl")]))
    equal = _protostructure_record_from_value(Protostructure(Spacegroup.standard(225), [("b", "Cl"), ("a", "Na")]))
    low = _protostructure_record_from_value(Protostructure(1, [("a", "He")]))
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={ProtostructureEntry: ProtostructureRecord})
        first_sid = store.save(first)
        equal_sid = store.save(equal)
        low_sid = store.save(low)
        assert first_sid == equal_sid
        assert low_sid != first_sid

        searcher = store.searcher()
        variable = searcher.variable(ProtostructureRecord)
        searcher.add(variable.spacegroup_it_number > 2)
        assert searcher.count() == 1


def test_sql_store_prototype_dedup_and_label_query() -> None:
    pytest.importorskip("sqlalchemy")
    from httk.store import Backend, SqlStore

    first = _prototype_record_from_value(Prototype(_rocksalt_protopattern(), discriminator="001"))
    equal = _prototype_record_from_value(
        Prototype(Protopattern(Spacegroup.standard(225), [("b", "B"), ("a", "A")]), discriminator="001")
    )
    other = _prototype_record_from_value(Prototype(Protopattern(1, [("a", "A")]), discriminator="001"))
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={PrototypeEntry: PrototypeRecord})
        first_sid = store.save(first)
        equal_sid = store.save(equal)
        other_sid = store.save(other)
        assert first_sid == equal_sid  # content-id dedup
        assert other_sid != first_sid

        searcher = store.searcher()
        variable = searcher.variable(PrototypeRecord)
        searcher.add(variable.spacegroup_it_number > 2)
        assert searcher.count() == 1

        label_searcher = store.searcher()
        label_variable = label_searcher.variable(PrototypeRecord)
        label_searcher.add(label_variable.label == "AB_cF8_225_a_b")
        assert label_searcher.count() == 1

        fetched = store.fetch(PrototypeRecord, first_sid, eager=True)
        assert fetched.id == first.id
        assert fetched.label == "AB_cF8_225_a_b"


def test_sql_store_structuretype_representative_round_trips() -> None:
    pytest.importorskip("sqlalchemy")
    from httk.store import Backend, SqlStore

    value = StructuretypeView(_rocksalt_asu()).unview()
    record = _structuretype_record_from_value(value)
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={StructuretypeEntry: StructuretypeRecord})
        sid = store.save(record)
        fetched = store.fetch(StructuretypeRecord, sid, eager=True)

    assert fetched.id == record.id
    assert _structuretype_from_record(fetched) == value


def test_sql_store_protopattern_round_trips() -> None:
    pytest.importorskip("sqlalchemy")
    from httk.store import Backend, SqlStore

    record = _protopattern_record_from_value(_rocksalt_protopattern())
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={ProtopatternEntry: ProtopatternRecord})
        sid = store.save(record)
        fetched = store.fetch(ProtopatternRecord, sid, eager=True)

    assert fetched.id == record.id
    assert fetched.label == "AB_cF8_225_a_b"
