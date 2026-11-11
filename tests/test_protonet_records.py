"""Durable records, entry families and registrations for protonettypes and protonetstructures."""

import dataclasses

import pytest
from httk.core.register import known_entry_families, resolve_entry_record
from httk.core.storage import content_id, resolve_storage_record, storage_identity_name
from test_protonet_models import _respelled
from test_protostructure_records import _rocksalt, _rocksalt_prototype

from httk.atomistic import (
    Protonetstructure,
    ProtonetstructureRecord,
    ProtonetstructureView,
    Protonettype,
    ProtonettypeRecord,
    ProtonettypeView,
    Protostructure,
    Prototype,
    Species,
)
from httk.atomistic.entries.prototypes import ProtonetstructureEntry, ProtonettypeEntry
from httk.atomistic.models.prototype.notation import parse_protonetstructure_label, parse_protonettype_label
from httk.atomistic.storage.records import (
    ProtonetBondRecord,
    _bare_protostructure_from_record,
    _bare_protostructure_record_from_value,
    _bare_prototype_from_record,
    _bare_prototype_record_from_value,
    _net_rows,
    _protonetstructure_from_record,
    _protonetstructure_record_from_value,
    _protonettype_from_record,
    _protonettype_record_from_value,
    _protostructure_from_record,
    _protostructure_record_from_value,
    _prototype_from_record,
    _prototype_record_from_value,
)
from httk.atomistic.symmetry._canonical_form import _CANONICAL_FORM_VERSION

ROWS = [(i, j, 1, 0, 0, 0) for i, j in ((1, 8), (1, 10), (2, 8), (2, 10), (3, 9), (3, 10), (4, 9))] + [
    (4, 10, 3, 0, 0, 0)
]
LETTERS = ["c"] * 4 + ["a", "b", "c", "a", "b", "c"]
CLASSES = ["A"] * 4 + ["B"] * 3 + ["C"] * 2 + ["D"]
NAMES = ["S"] * 4 + ["K"] * 3 + ["Ag"] * 2 + ["Sb"]
TIED_ROWS = [(1, 1, 2, 0, 0, 0), (1, 2, 1, 0, 0, 0)]


def _sp(name: str) -> Species:
    return Species(name, (name,), (1,))


def _anon(net=ROWS) -> Protonettype:
    return Protonettype(34, list(zip(LETTERS, CLASSES, strict=True)), net)


def _assigned(net=ROWS) -> Protonetstructure:
    return Protonetstructure(34, [(letter, _sp(name)) for letter, name in zip(LETTERS, NAMES, strict=True)], net)


def _tied(first: str = "Zn", second: str = "S") -> Protonetstructure:
    return Protonetstructure(4, [("a", _sp(first)), ("a", _sp(second))], TIED_ROWS)


CASES = [
    (_anon, _protonettype_record_from_value, _protonettype_from_record),
    (_assigned, _protonetstructure_record_from_value, _protonetstructure_from_record),
    (_tied, _protonetstructure_record_from_value, _protonetstructure_from_record),
]


@pytest.mark.parametrize(("build", "to_record", "from_record"), CASES)
def test_value_record_value_round_trip(build, to_record, from_record) -> None:
    value = build()
    record = to_record(value)
    assert from_record(record) == value
    assert record.label == str(value.label)
    assert content_id(record) == content_id(value)
    assert dataclasses.replace(record, id="public", immutable_id="revision") == record
    type(record).__httk_validate__(record)


def test_tied_classes_keep_the_value_order() -> None:
    zn_first, s_first = _tied(), _tied("S", "Zn")
    record = _protonetstructure_record_from_value(zn_first)
    assert [occupation.species.name for occupation in record.occupations] == ["Zn", "S"]
    assert record.label == "AB_mP4_4_a_a__1~1@2.2:Zn-S"
    assert _protonetstructure_from_record(record) == zn_first != s_first
    assert content_id(zn_first) != content_id(s_first)


@pytest.mark.parametrize(("build", "to_record", "from_record"), CASES[:2])
def test_validate_rejects_reordered_occupations_and_foreign_net(build, to_record, from_record) -> None:
    record = to_record(build())
    with pytest.raises(ValueError):
        type(record).__httk_validate__(dataclasses.replace(record, occupations=tuple(reversed(record.occupations))))
    with pytest.raises(ValueError, match="net rows are not in canonical order"):
        type(record).__httk_validate__(dataclasses.replace(record, net=tuple(reversed(record.net))))
    with pytest.raises(ValueError, match="contradicts"):
        type(record).__httk_validate__(dataclasses.replace(record, spacegroup_it_number=33))


def test_validate_rejects_a_non_canonical_net() -> None:
    rows = _respelled(ROWS)
    anonymous = Protonettype._of(34, list(zip(LETTERS, CLASSES, strict=True)), rows)
    assigned = Protonetstructure._of(34, [(x, _sp(name)) for x, name in zip(LETTERS, NAMES, strict=True)], rows)
    for record in (_protonettype_record_from_value(anonymous), _protonetstructure_record_from_value(assigned)):
        assert _net_rows(record) != _net_rows(_protonettype_record_from_value(_anon()))
        with pytest.raises(ValueError, match="net rows are not in canonical order"):
            type(record).__httk_validate__(record)


# every record with a canonical label, and how it converts back to a value
CANONICAL_LABEL_RECORDS = [
    (lambda: _bare_prototype_record_from_value(_rocksalt_prototype()), _bare_prototype_from_record),
    (lambda: _bare_protostructure_record_from_value(_rocksalt()), _bare_protostructure_from_record),
    (
        lambda: _prototype_record_from_value(Prototype(prototype=_rocksalt_prototype(), discriminator="001")),
        _prototype_from_record,
    ),
    (
        lambda: _protostructure_record_from_value(Protostructure(225, [("a", "Na"), ("b", "Cl")], discriminator="001")),
        _protostructure_from_record,
    ),
    (lambda: _protonettype_record_from_value(_anon()), _protonettype_from_record),
    (lambda: _protonetstructure_record_from_value(_assigned()), _protonetstructure_from_record),
]


@pytest.mark.parametrize(("build", "from_record"), CANONICAL_LABEL_RECORDS)
def test_records_carry_the_canonical_form_version(build, from_record) -> None:
    record = build()
    assert record.canonical_form == _CANONICAL_FORM_VERSION == 1
    assert type(record).__httk_project__(from_record(record))["canonical_form"] == _CANONICAL_FORM_VERSION
    type(record).__httk_validate__(record)
    with pytest.raises(TypeError, match="canonical_form must be an integer"):
        dataclasses.replace(record, canonical_form=True)
    for stale in (0, 2):
        old = dataclasses.replace(record, canonical_form=stale)
        assert content_id(old) != content_id(record)
        with pytest.raises(ValueError, match=r"canonical form .* changed; rebuild the database"):
            from_record(old)
        with pytest.raises(ValueError, match="rebuild the database"):
            type(record).__httk_validate__(old)


P1BAR_NET = [(1, 1, 2, 0, -1, 0), (1, 2, 1, 0, 0, 1), (1, 3, 2, 0, -1, 1), (1, 3, 2, 0, 0, 1), (2, 3, 2, -1, -1, 0)]


@pytest.mark.xfail(strict=True, reason="D7: P-1 action set not closed; fixed by Task 4.1")
@pytest.mark.parametrize("assigned", [False, True], ids=["protonettype", "protonetstructure"])
def test_p1bar_constructor_is_idempotent(assigned: bool) -> None:
    if assigned:
        value: Protonettype | Protonetstructure = Protonetstructure(
            2, [("d", _sp("Na")), ("e", _sp("K")), ("h", _sp("Cl"))], P1BAR_NET
        )
        parse, to_record = parse_protonetstructure_label, _protonetstructure_record_from_value
    else:
        value = Protonettype(2, [("d", "A"), ("e", "B"), ("h", "C")], P1BAR_NET)
        parse, to_record = parse_protonettype_label, _protonettype_record_from_value
    assert type(value)(value.spacegroup, value.occupations, value.net) == value
    assert parse(str(value.label)) == value
    record = to_record(value)  # type: ignore[arg-type]
    type(record).__httk_validate__(record)


def test_record_fields_are_checked() -> None:
    record = _protonettype_record_from_value(_anon())
    with pytest.raises(TypeError, match="ProtonetBondRecord"):
        dataclasses.replace(record, net=((1, 8, 1, 0, 0, 0),))
    with pytest.raises(TypeError, match="integers"):
        ProtonetBondRecord(1, 2, 1, 0, 0, True)


def test_content_identity_tracks_the_net() -> None:
    assert content_id(_anon()) != content_id(_anon(ROWS[:-1]))
    assert content_id(_assigned()) != content_id(_assigned(ROWS[:-1]))
    assert content_id(_anon()) != content_id(_assigned())


def test_storage_names_families_and_registry() -> None:
    assert storage_identity_name(ProtonettypeRecord) == "atomistic_protonettype"
    assert storage_identity_name(ProtonetstructureRecord) == "atomistic_protonetstructure"
    assert storage_identity_name(ProtonetBondRecord) == "atomistic_protonet_bond"
    assert ProtonettypeEntry.type == "protonettypes"
    assert ProtonetstructureEntry.type == "protonetstructures"
    assert {"protonettypes", "protonetstructures"} <= set(known_entry_families())
    assert resolve_entry_record("atomistic-protonettype") is ProtonettypeRecord
    assert resolve_entry_record("atomistic-protonetstructure") is ProtonetstructureRecord
    assert resolve_storage_record(ProtonettypeView(_anon())) is ProtonettypeRecord
    assert resolve_storage_record(ProtonetstructureView(_assigned())) is ProtonetstructureRecord


@pytest.mark.parametrize("engine", ["sqlite", "duckdb"])
def test_store_round_trip_after_reopen(tmp_path, engine) -> None:
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("httk.store")
    if engine == "duckdb":
        pytest.importorskip("duckdb_engine")
    from httk.store import Backend, EntryIdScheme, SqlStore

    values = (_anon(), _anon(ROWS[:-1]), _assigned(), _tied(), _tied("S", "Zn"))
    declarations = {ProtonettypeEntry: ProtonettypeRecord, ProtonetstructureEntry: ProtonetstructureRecord}
    path = tmp_path / f"protonets.{engine}"
    factory = getattr(Backend, engine)
    with factory(path) as database:
        store = SqlStore(database, entry_ids=EntryIdScheme("httk.test", "1"), entry_records=declarations)
        saved = [(value, store.save(value)) for value in values]
    with factory(path) as database:
        store = SqlStore(database, entry_ids=EntryIdScheme("httk.test", "1"), entry_records=declarations)
        for value, sid in saved:
            record_type = resolve_storage_record(value)
            fetched = store.fetch(record_type, sid, eager=True)
            from_record = (
                _protonettype_from_record if record_type is ProtonettypeRecord else _protonetstructure_from_record
            )
            assert from_record(fetched) == value
            assert content_id(fetched) == content_id(value)
            assert fetched.label == str(value.label)
        for record_type, count in ((ProtonettypeRecord, 2), (ProtonetstructureRecord, 3)):
            searcher = store.searcher()
            searcher.variable(record_type)
            assert searcher.count() == count
        search = store.searcher()
        v = search.variable(ProtonettypeRecord)
        search.add(v.net.operation == 3)  # child-record chaining reaches the bond rows
        assert [row.v.label for row in search.results(v=v)] == [str(_anon().label)]
