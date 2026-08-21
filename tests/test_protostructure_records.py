"""Tests for the geometry-free and geometrical-class taxonomy storage records."""

import dataclasses
from fractions import Fraction

import pytest
from httk.core.storage import content_id, storage_identity_name

from httk.atomistic import (
    ASUStructure,
    Cell,
    FundamentalDomainPattern,
    FundamentalDomainPatternRecord,
    Protochroma,
    ProtochromaRecord,
    Protostructure,
    ProtostructureRecord,
    Prototype,
    PrototypeRecord,
    PrototypeView,
    Spacegroup,
    Species,
    Crystallotype,
    CrystallotypeRecord,
    CrystallotypeView,
    WyckoffOccupationRecord,
    WyckoffSite,
)
from httk.atomistic.entries.prototypes import (
    ProtochromaEntry,
    ProtostructureEntry,
    PrototypeEntry,
    CrystallotypeEntry,
)
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.storage.records import (
    _fundamental_domain_pattern_from_record,
    _fundamental_domain_pattern_record_from_value,
    _protochroma_from_record,
    _protochroma_record_from_value,
    _protostructure_from_record,
    _protostructure_record_from_value,
    _prototype_from_record,
    _prototype_record_from_value,
    _crystallotype_from_record,
    _crystallotype_record_from_value,
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


def _rocksalt_protochroma() -> Protochroma:
    return Protochroma(225, [("a", "A"), ("b", "B")])


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
        ProtochromaRecord,
        PrototypeRecord,
        CrystallotypeRecord,
        FundamentalDomainPatternRecord,
        WyckoffOccupationRecord,
    ):
        identity_name = storage_identity_name(record_type)
        assert identity_name == record_type.__httk_storage__.storage_name
        assert "httk.atomistic" not in identity_name


def test_new_record_storage_names_are_suffix_free() -> None:
    assert ProtochromaRecord.__httk_storage__.storage_name == "atomistic_protochroma"
    assert PrototypeRecord.__httk_storage__.storage_name == "atomistic_prototype"
    assert CrystallotypeRecord.__httk_storage__.storage_name == "atomistic_crystallotype"
    assert FundamentalDomainPatternRecord.__httk_storage__.storage_name == "atomistic_fundamental_domain_pattern"


def test_taxonomy_record_indexes_cover_it_number_and_label() -> None:
    for record_type in (ProtostructureRecord, ProtochromaRecord, PrototypeRecord, CrystallotypeRecord):
        assert record_type.__httk_storage__.indexes == (("spacegroup_it_number",), ("label",))


# --- protostructure (unchanged) ---


def test_protostructure_record_label_format_is_deterministic() -> None:
    record = _protostructure_record_from_value(_rocksalt())
    assert record.label == "AB_cF8_225_a_b:Na-Cl"


def test_protostructure_golden_content_id_is_layout_independent() -> None:
    record = _protostructure_record_from_value(_rocksalt())
    assert record.id == content_id(_rocksalt()) == "329f155afa99629b803ffd35a25dd51876f193f00770d738ab83d65d9e206119"


def test_equal_protostructures_share_content_id_including_permuted_species_order() -> None:
    first = Protostructure(225, [("a", "Na"), ("b", "Cl")])
    permuted = Protostructure(Spacegroup.standard(225), [("b", "Cl"), ("a", "Na")])
    assert first == permuted
    assert _protostructure_record_from_value(first).id == _protostructure_record_from_value(permuted).id


def test_protostructure_record_rejects_reversed_occupations() -> None:
    good = _protostructure_record_from_value(_rocksalt())
    reordered = dataclasses.replace(good, occupations=tuple(reversed(good.occupations)))
    with pytest.raises(ValueError, match="not in canonical order"):
        ProtostructureRecord.__httk_validate__(reordered)


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


# --- protochroma record ---


def test_protochroma_record_round_trips_and_labels() -> None:
    value = _rocksalt_protochroma()
    record = _protochroma_record_from_value(value)
    assert record.label == "AB_cF8_225_a_b"
    assert _protochroma_from_record(record) == value


def test_protochroma_golden_content_id_is_layout_independent() -> None:
    record = _protochroma_record_from_value(_rocksalt_protochroma())
    assert record.id == content_id(_rocksalt_protochroma())
    assert record.id == "7fb1c7369fbdebeb076c4a30b80c09621f0a09883084f4758e412a79c0e7eef3"


def test_protochroma_record_rejects_non_canonical_field_order() -> None:
    good = _protochroma_record_from_value(_rocksalt_protochroma())
    swapped = dataclasses.replace(good, wyckoff_letters=tuple(reversed(good.wyckoff_letters)))
    with pytest.raises(ValueError, match="not in canonical order"):
        ProtochromaRecord.__httk_validate__(swapped)


# --- fundamental-domain-pattern record (the renamed geometric record) ---


def test_fundamental_domain_pattern_record_round_trips_surd_cell_and_free_parameters() -> None:
    value = _hexagonal_fundamental_domain_pattern()
    record = _fundamental_domain_pattern_record_from_value(value)
    rebuilt = _fundamental_domain_pattern_from_record(record)
    assert rebuilt == value
    assert record.cell.basis  # surd basis retained as exact scalars


def _two_site_fundamental_domain_pattern() -> FundamentalDomainPattern:
    a = Species("A", ("X",), (1,), labels=("A",))
    b = Species("B", ("X",), (1,), labels=("B",))
    return FundamentalDomainPattern(CELL, 225, (WyckoffSite("a", EMPTY, "A"), WyckoffSite("b", EMPTY, "B")), (a, b))


def test_fundamental_domain_pattern_record_rejects_permuted_sites() -> None:
    good = _fundamental_domain_pattern_record_from_value(_two_site_fundamental_domain_pattern())
    permuted = dataclasses.replace(good, wyckoff_sites=tuple(reversed(good.wyckoff_sites)))
    with pytest.raises(ValueError, match="not in canonical order"):
        FundamentalDomainPatternRecord.__httk_validate__(permuted)


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
    value = Prototype(_rocksalt_protochroma(), discriminator="001")
    record = _prototype_record_from_value(value)
    assert record.representative is None
    assert record.discriminator == "001"
    assert _prototype_from_record(record) == value


def test_prototype_record_requires_a_class_distinction() -> None:
    good = _prototype_record_from_value(Prototype(_rocksalt_protochroma(), discriminator="001"))
    with pytest.raises(ValueError, match="at least one of representative or discriminator"):
        PrototypeRecord(
            spacegroup_it_number=good.spacegroup_it_number,
            spacegroup_hall_entry=good.spacegroup_hall_entry,
            wyckoff_letters=good.wyckoff_letters,
            labels=good.labels,
        )


def test_prototype_record_rejects_permuted_class_labels() -> None:
    good = _prototype_record_from_value(Prototype(_rocksalt_protochroma(), discriminator="001"))
    permuted = dataclasses.replace(good, labels=tuple(reversed(good.labels)))
    with pytest.raises(ValueError, match="not in canonical order"):
        PrototypeRecord.__httk_validate__(permuted)


def test_prototype_golden_content_ids_are_layout_independent() -> None:
    representative_carrying = PrototypeView(_rocksalt_asu()).unview()
    rep_record = _prototype_record_from_value(representative_carrying)
    assert rep_record.id == content_id(representative_carrying)
    assert rep_record.id == "b30718b654cf19dc55cc6d065e361dff27c412d112288b634a9b963b99f897c5"

    discriminator_only = Prototype(_rocksalt_protochroma(), discriminator="001")
    disc_record = _prototype_record_from_value(discriminator_only)
    assert disc_record.id == content_id(discriminator_only)
    assert disc_record.id == "3ee6b735605d2804d0cfc58bc112a7696d031ed3b3ef267ccfb1fd45992da5a1"


# --- crystallotype record ---


def test_crystallotype_record_round_trips_representative_carrying() -> None:
    value = CrystallotypeView(_rocksalt_asu()).unview()
    record = _crystallotype_record_from_value(value)
    assert record.label == "AB_cF8_225_a_b:Na-Cl"
    assert record.representative is not None
    assert record.discriminator is None
    assert _crystallotype_from_record(record) == value


def test_crystallotype_record_round_trips_discriminator_only() -> None:
    protostructure = Protostructure(225, [("a", Species("Na", ("Na",), (1,))), ("b", Species("Cl", ("Cl",), (1,)))])
    value = Crystallotype(protostructure, discriminator="001")
    record = _crystallotype_record_from_value(value)
    assert record.representative is None
    assert record.discriminator == "001"
    assert _crystallotype_from_record(record) == value


def test_crystallotype_record_requires_a_class_distinction() -> None:
    good = _crystallotype_record_from_value(
        Crystallotype(
            Protostructure(225, [("a", Species("Na", ("Na",), (1,))), ("b", Species("Cl", ("Cl",), (1,)))]),
            discriminator="001",
        )
    )
    with pytest.raises(ValueError, match="at least one of representative or discriminator"):
        CrystallotypeRecord(
            spacegroup_it_number=good.spacegroup_it_number,
            spacegroup_hall_entry=good.spacegroup_hall_entry,
            occupations=good.occupations,
        )


def test_crystallotype_record_rejects_non_canonical_occupation_order() -> None:
    protostructure = Protostructure(225, [("a", Species("Na", ("Na",), (1,))), ("b", Species("Cl", ("Cl",), (1,)))])
    good = _crystallotype_record_from_value(Crystallotype(protostructure, discriminator="001"))
    reordered = dataclasses.replace(good, occupations=tuple(reversed(good.occupations)))
    with pytest.raises(ValueError, match="not in canonical order"):
        CrystallotypeRecord.__httk_validate__(reordered)


def test_crystallotype_golden_content_ids_are_layout_independent() -> None:
    representative_carrying = CrystallotypeView(_rocksalt_asu()).unview()
    rep_record = _crystallotype_record_from_value(representative_carrying)
    assert rep_record.id == content_id(representative_carrying)
    assert rep_record.id == "27cc7f99302ff11db3febbdcfd02de914073fe75beb38adf3dad55c93a21a80a"

    protostructure = Protostructure(225, [("a", Species("Na", ("Na",), (1,))), ("b", Species("Cl", ("Cl",), (1,)))])
    discriminator_only = Crystallotype(protostructure, discriminator="001")
    disc_record = _crystallotype_record_from_value(discriminator_only)
    assert disc_record.id == content_id(discriminator_only)
    assert disc_record.id == "66b2ac605e1a9a75493ad41b937e0e689724dabff1b12dd561df8384c4b7826b"


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

    first = _prototype_record_from_value(Prototype(_rocksalt_protochroma(), discriminator="001"))
    equal = _prototype_record_from_value(
        Prototype(Protochroma(Spacegroup.standard(225), [("b", "B"), ("a", "A")]), discriminator="001")
    )
    other = _prototype_record_from_value(Prototype(Protochroma(1, [("a", "A")]), discriminator="001"))
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


def test_sql_store_crystallotype_representative_round_trips() -> None:
    pytest.importorskip("sqlalchemy")
    from httk.store import Backend, SqlStore

    value = CrystallotypeView(_rocksalt_asu()).unview()
    record = _crystallotype_record_from_value(value)
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={CrystallotypeEntry: CrystallotypeRecord})
        sid = store.save(record)
        fetched = store.fetch(CrystallotypeRecord, sid, eager=True)

    assert fetched.id == record.id
    assert _crystallotype_from_record(fetched) == value


def test_sql_store_protochroma_round_trips() -> None:
    pytest.importorskip("sqlalchemy")
    from httk.store import Backend, SqlStore

    record = _protochroma_record_from_value(_rocksalt_protochroma())
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={ProtochromaEntry: ProtochromaRecord})
        sid = store.save(record)
        fetched = store.fetch(ProtochromaRecord, sid, eager=True)

    assert fetched.id == record.id
    assert fetched.label == "AB_cF8_225_a_b"
