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
    SpeciesRecord,
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
        # representative species sorted: the stored representative lists its classes by name (Cl, Na)
        "e61a04be21f46d24121d06a68f9d5f4b1fa5f329d1befe8e526461277985402c",
        # dropped precision: the "Na"/"Cl" shorthand reads concentration 1.0 (precision 1/10), and
        # the occupations now store the species classes, which carry no precision
        "40ee32e63d8249bdba0d19f2b39edbb5f35d7140ab4d6abe081d04c9373859f8",
        # representative species sorted, as the first pin
        "1c4d636ddef9d386f9e4f9002967e16b52b29d657c90645d0b4a9287f80661ae",
    )

    records = tuple(_protostructure_record_from_value(value) for value in values)
    assert tuple(record.id for record in records) == (None,) * len(records)
    assert tuple(content_id(value) for value in values) == expected_ids
    assert len(set(expected_ids)) == len(expected_ids)
    identified = dataclasses.replace(records[0], id="logical", immutable_id="immutable")
    assert content_id(identified) == content_id(records[0])
    assert records[0].label == "AB_cF8_225_a_b:Na-Cl"
    # the first pin is that of the representative written with its species already sorted
    rocksalt = _rocksalt_asu()
    written_sorted = ASUStructure(CELL, 225, rocksalt.wyckoff_sites, tuple(reversed(rocksalt.species)))
    assert content_id(Protostructure(representative=written_sorted)) == expected_ids[0]
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
        "ca91ca7cf0cf59f6ec294b3d469a605203c619d657d85a6b9c145d54561dd768",
        "4590a246a1a691c959d7b3dcdea3a662fdf4f74640e265b6a94c1ff598959c5c",
        "e05593314a108364b609ea326f37888d65c47d9d79880b62f7080d83165c1065",
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


def test_bare_records_store_species_classes_and_validate_spacegroup() -> None:
    charged = BareProtostructure(225, [("a", Species("Na", ("Na",), (1,), charges=(1,))), ("b", "Cl")])
    plain = _bare_protostructure_record_from_value(_rocksalt())
    decorated = _bare_protostructure_record_from_value(charged)
    # dropped charges: the charged and the plain value store the same species classes
    assert decorated == plain
    assert content_id(decorated) == content_id(plain)
    assert all(occupation.species.charges is None for occupation in decorated.occupations)
    assert _bare_protostructure_from_record(decorated) == charged == _rocksalt()
    # a stored species that is not its class (an older row) does not round-trip and is rejected
    sodium = SpeciesRecord(**SpeciesRecord.__httk_project__(Species("Na", ("Na",), (1,), charges=(1,))))
    stale = dataclasses.replace(
        plain, occupations=(dataclasses.replace(plain.occupations[0], species=sodium), *plain.occupations[1:])
    )
    with pytest.raises(ValueError, match="not a reduced species class"):
        BareProtostructureRecord.__httk_validate__(stale)
    # the same for the representative of a refined record, and a representative whose classes are
    # not in name order (both as an older row could hold them)
    refined = _protostructure_record_from_value(Protostructure(representative=_rocksalt_asu()))
    assert refined.representative is not None
    decorated = tuple(sodium if species.name == "Na" else species for species in refined.representative.species)
    for representative, message in (
        (
            dataclasses.replace(refined.representative, species=decorated),
            "representative species 'Na' is not a reduced",
        ),
        (
            dataclasses.replace(refined.representative, species=tuple(reversed(refined.representative.species))),
            "representative are not in canonical order",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            ProtostructureRecord.__httk_validate__(dataclasses.replace(refined, representative=representative))
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
            "7a5755af2c69252cc9f20441803e474fff2b16a3db65af1d7269ad42bbbec999",
        ),
        (
            _rocksalt(),
            _bare_protostructure_record_from_value,
            # dropped precision: the "Na"/"Cl" shorthand's 1/10 concentration precision is not stored
            "9f3c4fd199315f2db7da3c61841cd2a5953ab293c1e9922bf7aeac77c4d79cb6",
        ),
    ):
        assert content_id(value) == content_id(project(value)) == expected
    # the shorthand's precision is gone: the pin is that of the exact integer-concentration species
    exact = BareProtostructure(225, [("a", Species("Na", ("Na",), (1,))), ("b", Species("Cl", ("Cl",), (1,)))])
    assert content_id(exact) == content_id(_rocksalt())
