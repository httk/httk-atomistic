"""Focused identity and round-trip tests for the retained prototype families."""

import dataclasses
from fractions import Fraction

import pytest
from httk.core.storage import content_id, storage_identity_name

from httk.atomistic import (
    ASUStructure,
    BareProtostructure,
    BareProtostructureRecord,
    BareProtostructureView,
    BarePrototype,
    BarePrototypeRecord,
    BarePrototypeView,
    Cell,
    FundamentalDomainTemplate,
    FundamentalDomainTemplateRecord,
    Protostructure,
    ProtostructureRecord,
    Prototype,
    PrototypeOccupation,
    PrototypeRecord,
    Species,
    WyckoffOccupationRecord,
    WyckoffSite,
)
from httk.atomistic.entries.prototypes import (
    BareProtostructureEntry,
    BarePrototypeEntry,
    ProtostructureEntry,
    PrototypeEntry,
)
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.storage.records import (
    _bare_protostructure_from_record,
    _bare_protostructure_record_from_value,
    _bare_prototype_record_from_value,
    _fundamental_domain_template_from_record,
    _fundamental_domain_template_record_from_value,
    _protostructure_from_record,
    _protostructure_record_from_value,
    _prototype_from_record,
    _prototype_record_from_value,
)

EMPTY: tuple[Fraction, ...] = ()
CELL = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]


def _rocksalt() -> BareProtostructure:
    return BareProtostructure(225, [("a", "Na"), ("b", "Cl")])


def _rocksalt_asu() -> ASUStructure:
    return ASUStructure(
        CELL,
        225,
        (WyckoffSite("a", EMPTY, "Na"), WyckoffSite("b", EMPTY, "Cl")),
        (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
    )


def _rocksalt_prototype() -> BarePrototype:
    return BarePrototype(225, [("a", "A"), ("b", "B")])


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

    recognized = BareProtostructureView(asu).unview()
    hand_built = BareProtostructure(225, [("a", Species("Na", ("Na",), (1,))), ("b", Species("Cl", ("Cl",), (1,)))])
    assert recognized == hand_built
    assert hash(recognized) == hash(hand_built)
    assert content_id(recognized) == content_id(hand_built)

    recognized_type = BarePrototypeView(asu).unview()
    assert recognized_type == _rocksalt_prototype()
    assert hash(recognized_type) == hash(_rocksalt_prototype())
    assert content_id(recognized_type) == content_id(_rocksalt_prototype())


def test_record_identity_names_are_suffix_free() -> None:
    for record_type in (
        BareProtostructureRecord,
        BarePrototypeRecord,
        ProtostructureRecord,
        PrototypeRecord,
        FundamentalDomainTemplateRecord,
    ):
        assert storage_identity_name(record_type) == record_type.__httk_storage__.storage_name
        assert "httk.atomistic" not in record_type.__httk_storage__.storage_name
    assert ProtostructureRecord.__httk_storage__.storage_name == "atomistic_protostructure"
    assert PrototypeRecord.__httk_storage__.storage_name == "atomistic_prototype"


def test_taxonomy_families_are_retained() -> None:
    assert BareProtostructureEntry.type == "bare_protostructures"
    assert BarePrototypeEntry.type == "bare_prototypes"
    assert ProtostructureEntry.type == "protostructures"
    assert PrototypeEntry.type == "prototypes"


def test_protostructure_record_optional_forms_have_distinct_pinned_identities() -> None:
    values = (
        Protostructure(representative=_rocksalt_asu()),
        Protostructure(225, [("a", "Na"), ("b", "Cl")], discriminator="001"),
        Protostructure(representative=_rocksalt_asu(), discriminator="001"),
    )
    expected_ids = (
        "85c4c9fb94a5045e6600c9f57aeb1f75cf6005c1527f243cdfd52aa67bc78525",
        "bca5f75b0fa6c5166c52064d2ba6258a0945542efcca4f8dc6f5d96751b5a8f8",
        "20e109409265f99c5f44b90a90b0fae6dbd56e53f3c629f01caa865860672b11",
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
    good = _bare_protostructure_record_from_value(_rocksalt())
    reordered = dataclasses.replace(good, occupations=tuple(reversed(good.occupations)))
    with pytest.raises(ValueError, match="not in canonical order"):
        BareProtostructureRecord.__httk_validate__(reordered)


def test_prototype_record_optional_forms_have_distinct_pinned_identities() -> None:
    representative = _rocksalt_template()
    values = (
        Prototype(representative=representative),
        Prototype(prototype=_rocksalt_prototype(), discriminator="001"),
        Prototype(representative=representative, discriminator="001"),
    )
    expected_ids = (
        "843cc4f8995f4bfe69a7b6b3d2bebeefa05df5b3057407a0d59535375b33f297",
        "7d9def986fe49e305211cd2c8f9baf50d132ffec1839f52acdc4177ca2e200c4",
        "424ad8ed8cbef1e69bdc9fc19bf88b4a03b6c1b157192c008d2406825ff88a95",
    )

    records = tuple(_prototype_record_from_value(value) for value in values)
    assert tuple(record.id for record in records) == (None,) * len(records)
    assert tuple(content_id(value) for value in values) == expected_ids
    assert len(set(expected_ids)) == len(expected_ids)
    for value, record in zip(values, records, strict=True):
        assert _prototype_from_record(record) == value


def test_sql_store_round_trips_all_optional_identity_forms() -> None:
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("httk.store")
    from httk.store import Backend, EntryIdScheme, SqlStore

    protostructures = (
        Protostructure(representative=_rocksalt_asu()),
        Protostructure(225, [("a", "Na"), ("b", "Cl")], discriminator="001"),
        Protostructure(representative=_rocksalt_asu(), discriminator="001"),
    )
    representative = _rocksalt_template()
    prototypes = (
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


def test_bare_prototype_record_has_no_refinement() -> None:
    value = _rocksalt_prototype()
    record = _bare_prototype_record_from_value(value)
    assert record.label == "AB_cF8_225_a_b"
    assert not hasattr(record, "representative")
    assert not hasattr(record, "discriminator")


def test_prototype_record_rejects_reversed_occupations() -> None:
    good = _bare_prototype_record_from_value(_rocksalt_prototype())
    reordered = dataclasses.replace(good, occupations=tuple(reversed(good.occupations)))
    with pytest.raises(ValueError, match="not in canonical order"):
        BarePrototypeRecord.__httk_validate__(reordered)


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


def test_refined_records_require_refinement() -> None:
    for bare, refined in (
        (_bare_prototype_record_from_value(_rocksalt_prototype()), PrototypeRecord),
        (_bare_protostructure_record_from_value(_rocksalt()), ProtostructureRecord),
    ):
        with pytest.raises(ValueError, match="requires a representative or discriminator"):
            refined(bare.spacegroup_it_number, bare.spacegroup_hall_entry, bare.occupations)


@pytest.mark.parametrize("engine", ["sqlite", "duckdb"])
@pytest.mark.parametrize("bulk", [False, True])
def test_bare_and_refined_records_remain_separate_after_reopen(tmp_path, engine, bulk) -> None:
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("httk.store")
    if engine == "duckdb":
        pytest.importorskip("duckdb_engine")
    from httk.store import Backend, EntryIdScheme, SqlStore

    records = (
        _bare_protostructure_record_from_value(_rocksalt()),
        _bare_prototype_record_from_value(_rocksalt_prototype()),
        _protostructure_record_from_value(Protostructure(representative=_rocksalt_asu())),
        _prototype_record_from_value(Prototype(representative=_rocksalt_template())),
        _protostructure_record_from_value(Protostructure(225, [("a", "Na"), ("b", "Cl")], discriminator="other")),
        _prototype_record_from_value(Prototype(prototype=_rocksalt_prototype(), discriminator="other")),
    )
    declarations = {
        BareProtostructureEntry: BareProtostructureRecord,
        BarePrototypeEntry: BarePrototypeRecord,
        ProtostructureEntry: ProtostructureRecord,
        PrototypeEntry: PrototypeRecord,
    }
    path = tmp_path / f"classes.{engine}"
    factory = getattr(Backend, engine)
    with factory(path) as database:
        store = SqlStore(database, entry_ids=EntryIdScheme("httk.test", "1"), entry_records=declarations)
        if bulk:
            with store.bulk_ingest(finalize="deferred") as ingest:
                provisional = [ingest.save(record) for record in records]
            saved = [
                (type(record), ingest.resolved_sid(type(record), key), content_id(record))
                for record, key in zip(records, provisional, strict=True)
            ]
        else:
            saved = [(type(record), store.save(record), content_id(record)) for record in records]
        assert len({identity for _, _, identity in saved}) == 6
    with factory(path) as database:
        store = SqlStore(database, entry_ids=EntryIdScheme("httk.test", "1"), entry_records=declarations)
        for record_type, key, identity in saved:
            assert content_id(store.fetch(record_type, key, eager=True)) == identity
        for record_type, count in (
            (BarePrototypeRecord, 1),
            (BareProtostructureRecord, 1),
            (PrototypeRecord, 2),
            (ProtostructureRecord, 2),
        ):
            searcher = store.searcher()
            searcher.variable(record_type)
            assert searcher.count() == count


def test_bare_records_preserve_decorated_species_and_validate_spacegroup() -> None:
    charged = BareProtostructure(225, [("a", Species("Na", ("Na",), (1,), charges=(1,))), ("b", "Cl")])
    plain = _bare_protostructure_record_from_value(_rocksalt())
    decorated = _bare_protostructure_record_from_value(charged)
    assert decorated.label == plain.label
    assert content_id(decorated) != content_id(plain)
    assert _bare_protostructure_from_record(decorated) == charged
    for record in (plain, _bare_prototype_record_from_value(_rocksalt_prototype())):
        conflicting = dataclasses.replace(record, spacegroup_it_number=224)
        with pytest.raises(ValueError, match="contradicts"):
            type(record).__httk_validate__(conflicting)
        identified = dataclasses.replace(record, id="public", immutable_id="revision")
        assert content_id(identified) == content_id(record)


def test_bare_record_identity_pins_match_models() -> None:
    for value, project, expected in (
        (
            _rocksalt_prototype(),
            _bare_prototype_record_from_value,
            "7a0ab0f11ac5dd2115c28b5a6e99d5a591ae3cdc23b4255438788302603124dc",
        ),
        (
            _rocksalt(),
            _bare_protostructure_record_from_value,
            "5fe9c17665cb433c5e7f8ab681d76eeec7f13de31dc19acab1afb6cfca4e3316",
        ),
    ):
        assert content_id(value) == content_id(project(value)) == expected
