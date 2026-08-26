"""Focused identity and round-trip tests for the retained prototype families."""

import dataclasses
from fractions import Fraction

import pytest
from httk.core.storage import content_id, storage_identity_name

from httk.atomistic import (
    ASUStructure,
    Cell,
    FundamentalDomainTemplate,
    FundamentalDomainTemplateRecord,
    Protostructure,
    ProtostructureRecord,
    ProtostructureView,
    Prototype,
    PrototypeOccupation,
    PrototypeRecord,
    PrototypeView,
    Species,
    WyckoffOccupationRecord,
    WyckoffSite,
)
from httk.atomistic.entries.prototypes import ProtostructureEntry, PrototypeEntry
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.storage.records import (
    _fundamental_domain_template_from_record,
    _fundamental_domain_template_record_from_value,
    _protostructure_from_record,
    _protostructure_record_from_value,
    _prototype_from_record,
    _prototype_record_from_value,
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


def _rocksalt_prototype() -> Prototype:
    return Prototype(225, [("a", "A"), ("b", "B")])


def _rocksalt_template() -> FundamentalDomainTemplate:
    return FundamentalDomainTemplate(
        CELL,
        225,
        (WyckoffSite("a", EMPTY, "A"), WyckoffSite("b", EMPTY, "B")),
        (Species("A", ("X",), (1,), labels=("A",)), Species("B", ("X",), (1,), labels=("B",))),
    )


def _hexagonal_fundamental_domain_template() -> FundamentalDomainTemplate:
    dummy = Species("A", ("X",), (1,), labels=("A",))
    return FundamentalDomainTemplate(
        Cell(CellParams((2, 2, 3, 90, 90, 120)).basis),
        191,
        (WyckoffSite("a", EMPTY, "A"),),
        (dummy,),
    )


def test_recognized_values_are_provenance_independent() -> None:
    # Recognition returns a base value, so a recognized protostructure/prototype equals,
    # hashes like, and has the same record content id as a hand-built or label-parsed one.
    asu = _rocksalt_asu()

    recognized = ProtostructureView(asu).unview()
    hand_built = Protostructure(225, [("a", Species("Na", ("Na",), (1,))), ("b", Species("Cl", ("Cl",), (1,)))])
    assert recognized.representative is None
    assert recognized == hand_built
    assert hash(recognized) == hash(hand_built)
    assert content_id(recognized) == content_id(hand_built)

    recognized_type = PrototypeView(asu).unview()
    assert recognized_type.representative is None
    assert recognized_type == _rocksalt_prototype()
    assert hash(recognized_type) == hash(_rocksalt_prototype())
    assert content_id(recognized_type) == content_id(_rocksalt_prototype())


def test_record_identity_names_are_suffix_free() -> None:
    for record_type in (ProtostructureRecord, PrototypeRecord, FundamentalDomainTemplateRecord):
        assert storage_identity_name(record_type) == record_type.__httk_storage__.storage_name
        assert "httk.atomistic" not in record_type.__httk_storage__.storage_name
    assert ProtostructureRecord.__httk_storage__.storage_name == "atomistic_protostructure"
    assert PrototypeRecord.__httk_storage__.storage_name == "atomistic_prototype"


def test_taxonomy_families_are_retained() -> None:
    assert ProtostructureEntry.type == "protostructures"
    assert PrototypeEntry.type == "prototypes"


def test_protostructure_record_optional_forms_have_distinct_pinned_identities() -> None:
    values = (
        _rocksalt(),
        Protostructure(representative=_rocksalt_asu()),
        Protostructure(225, [("a", "Na"), ("b", "Cl")], discriminator="001"),
        Protostructure(representative=_rocksalt_asu(), discriminator="001"),
    )
    expected_ids = (
        "606d20e7ed847d7598c68045ba8199565a734adeb78025c5b5cd9d51667b68a1",
        "800a86d40ac957e5b799d62d605236caf3fe66ecc5294c0e66d95efe5ca67f2e",
        "1a6fefdacc7afbceb08da454c14a57ee9930048909f4c8ae959ac9408d4b38d8",
        "7049adcd791a86369c89223537bac45b5a5064c7f4054eda01b7ea5c4703fe1d",
    )

    records = tuple(_protostructure_record_from_value(value) for value in values)
    assert tuple(record.id for record in records) == (None,) * len(records)
    assert tuple(content_id(value) for value in values) == expected_ids
    assert len(set(expected_ids)) == len(expected_ids)
    identified = dataclasses.replace(records[0], id="logical", immutable_id="immutable")
    assert content_id(identified) == content_id(records[0])
    assert records[0].label == "AB_cF8_225_a_b:Na-Cl"
    for value, record in zip(values, records, strict=True):
        assert _protostructure_from_record(record) == value


def test_protostructure_record_rejects_reversed_occupations() -> None:
    good = _protostructure_record_from_value(_rocksalt())
    reordered = dataclasses.replace(good, occupations=tuple(reversed(good.occupations)))
    with pytest.raises(ValueError, match="not in canonical order"):
        ProtostructureRecord.__httk_validate__(reordered)


def test_prototype_record_optional_forms_have_distinct_pinned_identities() -> None:
    representative = _rocksalt_template()
    values = (
        _rocksalt_prototype(),
        Prototype(representative=representative),
        Prototype(prototype=_rocksalt_prototype(), discriminator="001"),
        Prototype(representative=representative, discriminator="001"),
    )
    expected_ids = (
        "008c1ae747ddce0814286733971c0dcadd57bfd8ecc8b097db2604ff4e9ac99b",
        "459fb13715b8f4a66527dbd849ed40f154a2419a7ec7f1408b21d75b6d4fa6a1",
        "79f060d34eea0e203ea2985c19677e1a8d7390cdcedaf0d329e708c5ca74c562",
        "ae2ef2ecda278d7058bc620d0b6bfcf37ff4d05a4dedc0cc06621dfced74b746",
    )

    records = tuple(_prototype_record_from_value(value) for value in values)
    assert tuple(record.id for record in records) == (None,) * len(records)
    assert tuple(content_id(value) for value in values) == expected_ids
    assert len(set(expected_ids)) == len(expected_ids)
    for value, record in zip(values, records, strict=True):
        assert _prototype_from_record(record) == value


def test_sql_store_round_trips_all_optional_identity_forms() -> None:
    pytest.importorskip("sqlalchemy")
    from httk.store import Backend, EntryIdScheme, SqlStore

    protostructures = (
        _rocksalt(),
        Protostructure(representative=_rocksalt_asu()),
        Protostructure(225, [("a", "Na"), ("b", "Cl")], discriminator="001"),
        Protostructure(representative=_rocksalt_asu(), discriminator="001"),
    )
    representative = _rocksalt_template()
    prototypes = (
        _rocksalt_prototype(),
        Prototype(representative=representative),
        Prototype(prototype=_rocksalt_prototype(), discriminator="001"),
        Prototype(representative=representative, discriminator="001"),
    )

    with Backend.sqlite() as database:
        store = SqlStore(
            database,
            entry_ids=EntryIdScheme("httk.test", "1"),
            entry_records={
                ProtostructureEntry: ProtostructureRecord,
                PrototypeEntry: PrototypeRecord,
            },
        )
        for value in protostructures:
            record = _protostructure_record_from_value(value)
            sid = store.save(record)
            fetched = store.fetch(ProtostructureRecord, sid, eager=True)
            assert fetched.id is None
            assert content_id(fetched) == content_id(record)
            assert _protostructure_from_record(fetched) == value
        for value in prototypes:
            record = _prototype_record_from_value(value)
            sid = store.save(record)
            fetched = store.fetch(PrototypeRecord, sid, eager=True)
            assert fetched.id is None
            assert content_id(fetched) == content_id(record)
            assert _prototype_from_record(fetched) == value


def test_prototype_record_base_only_is_valid() -> None:
    value = _rocksalt_prototype()
    record = _prototype_record_from_value(value)
    assert record.label == "AB_cF8_225_a_b"
    assert record.representative is None
    assert record.discriminator is None


def test_prototype_record_rejects_reversed_occupations() -> None:
    good = _prototype_record_from_value(_rocksalt_prototype())
    reordered = dataclasses.replace(good, occupations=tuple(reversed(good.occupations)))
    with pytest.raises(ValueError, match="not in canonical order"):
        PrototypeRecord.__httk_validate__(reordered)


def test_fundamental_domain_template_record_round_trip() -> None:
    value = _hexagonal_fundamental_domain_template()
    record = _fundamental_domain_template_record_from_value(value)
    assert _fundamental_domain_template_from_record(record) == value
    assert record.id is None
    assert content_id(record) == content_id(value)


def test_wyckoff_occupation_record_rejects_raw_species() -> None:
    with pytest.raises(TypeError, match="SpeciesRecord"):
        WyckoffOccupationRecord(wyckoff="a", species=Species("Na", ("Na",), (1,)))  # type: ignore[arg-type]


def test_prototype_occupation_is_the_anonymous_record_vocabulary() -> None:
    assert PrototypeOccupation("a", "A") == PrototypeOccupation("a", "A")
