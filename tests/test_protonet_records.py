"""Durable records, entry families and registrations for protonettypes and protonetstructures."""

import contextlib
import dataclasses
import os
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from _protonet_support import default_shapes
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
from httk.atomistic.symmetry._canonical_form import _LABEL_FORM_VERSION, _NET_FORM_VERSION, _REFINED_FORM_VERSION

ROWS = [(i, j, 1, 0, 0, 0) for i, j in ((1, 8), (1, 10), (2, 8), (2, 10), (3, 9), (3, 10), (4, 9))] + [
    (4, 10, 3, 0, 0, 0)
]
LETTERS = ["c"] * 4 + ["a", "b", "c", "a", "b", "c"]
CLASSES = ["A"] * 4 + ["B"] * 3 + ["C"] * 2 + ["D"]
NAMES = ["S"] * 4 + ["K"] * 3 + ["Ag"] * 2 + ["Sb"]
TIED_ROWS = [(1, 1, 2, 0, 0, 0), (1, 2, 1, 0, 0, 0)]


def _sp(name: str) -> Species:
    return Species(name, (name,), (1,))


def _anon(net=ROWS, shapes=None) -> Protonettype:
    shapes = default_shapes(34, LETTERS, net) if shapes is None else shapes
    return Protonettype(34, list(zip(LETTERS, CLASSES, strict=True)), net, shapes)


def _assigned(net=ROWS) -> Protonetstructure:
    occupations = [(letter, _sp(name)) for letter, name in zip(LETTERS, NAMES, strict=True)]
    return Protonetstructure(34, occupations, net, default_shapes(34, LETTERS, net))


def _tied(first: str = "Zn", second: str = "S") -> Protonetstructure:
    return Protonetstructure(4, [("a", _sp(first)), ("a", _sp(second))], TIED_ROWS, ("TP3", None))


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
    rows, shapes = _respelled(ROWS), default_shapes(34, LETTERS, ROWS)
    anonymous = Protonettype._of(34, list(zip(LETTERS, CLASSES, strict=True)), rows, shapes)
    occupations = [(x, _sp(name)) for x, name in zip(LETTERS, NAMES, strict=True)]
    assigned = Protonetstructure._of(34, occupations, rows, shapes)
    for record in (_protonettype_record_from_value(anonymous), _protonetstructure_record_from_value(assigned)):
        assert _net_rows(record) != _net_rows(_protonettype_record_from_value(_anon()))
        with pytest.raises(ValueError, match="net rows are not in canonical order"):
            type(record).__httk_validate__(record)


# every record with a canonical label, how it converts back to a value, and its version family
CANONICAL_LABEL_RECORDS = [
    (lambda: _bare_prototype_record_from_value(_rocksalt_prototype()), _bare_prototype_from_record, "label"),
    (lambda: _bare_protostructure_record_from_value(_rocksalt()), _bare_protostructure_from_record, "label"),
    (
        lambda: _prototype_record_from_value(Prototype(prototype=_rocksalt_prototype(), discriminator="001")),
        _prototype_from_record,
        "refined",
    ),
    (
        lambda: _protostructure_record_from_value(Protostructure(225, [("a", "Na"), ("b", "Cl")], discriminator="001")),
        _protostructure_from_record,
        "refined",
    ),
    (lambda: _protonettype_record_from_value(_anon()), _protonettype_from_record, "net"),
    (lambda: _protonetstructure_record_from_value(_assigned()), _protonetstructure_from_record, "net"),
]
FORM_VERSIONS = {"label": _LABEL_FORM_VERSION, "refined": _REFINED_FORM_VERSION, "net": _NET_FORM_VERSION}


@pytest.mark.parametrize(("build", "from_record", "family"), CANONICAL_LABEL_RECORDS)
def test_records_carry_their_family_canonical_form_version(build, from_record, family) -> None:
    version = FORM_VERSIONS[family]
    record = build()
    assert record.canonical_form == version == {"label": 4, "refined": 9, "net": 7}[family]
    assert type(record).__httk_project__(from_record(record))["canonical_form"] == version
    type(record).__httk_validate__(record)
    with pytest.raises(TypeError, match="canonical_form must be an integer"):
        dataclasses.replace(record, canonical_form=True)
    for stale in (version - 1, version + 1):
        old = dataclasses.replace(record, canonical_form=stale)
        assert content_id(old) != content_id(record)
        message = rf"canonical {family} form {stale}, but this httk-atomistic uses {version}: .* changed; rebuild"
        with pytest.raises(ValueError, match=message):
            from_record(old)
        with pytest.raises(ValueError, match="rebuild the database"):
            type(record).__httk_validate__(old)


P1BAR_NET = [(1, 1, 2, 0, -1, 0), (1, 2, 1, 0, 0, 1), (1, 3, 2, 0, -1, 1), (1, 3, 2, 0, 0, 1), (2, 3, 2, -1, -1, 0)]


@pytest.mark.parametrize("assigned", [False, True], ids=["protonettype", "protonetstructure"])
def test_p1bar_constructor_is_idempotent(assigned: bool) -> None:
    shapes = default_shapes(2, ["d", "e", "h"], P1BAR_NET)
    if assigned:
        value: Protonettype | Protonetstructure = Protonetstructure(
            2, [("d", _sp("Na")), ("e", _sp("K")), ("h", _sp("Cl"))], P1BAR_NET, shapes
        )
        parse, to_record = parse_protonetstructure_label, _protonetstructure_record_from_value
    else:
        value = Protonettype(2, [("d", "A"), ("e", "B"), ("h", "C")], P1BAR_NET, shapes)
        parse, to_record = parse_protonettype_label, _protonettype_record_from_value
    assert type(value)(value.spacegroup, value.occupations, value.net, value.shapes) == value
    assert parse(str(value.label)) == value
    record = to_record(value)  # type: ignore[arg-type]
    type(record).__httk_validate__(record)


def test_record_fields_are_checked() -> None:
    record = _protonettype_record_from_value(_anon())
    with pytest.raises(TypeError, match="ProtonetBondRecord"):
        dataclasses.replace(record, net=((1, 8, 1, 0, 0, 0),))
    with pytest.raises(TypeError, match="integers"):
        ProtonetBondRecord(1, 2, 1, 0, 0, True)
    with pytest.raises(TypeError, match="shapes must contain str"):
        dataclasses.replace(record, shapes=(None,) * len(record.occupations))
    with pytest.raises(ValueError, match="one entry per occupation"):
        dataclasses.replace(record, shapes=("A2",))


SHAPED = ("A2",) * 4 + (None,) * 3 + ("I4", "SS4", "T4")  # the B8 shapes of K2AgSbS4


def test_records_store_every_shape_and_validate_them() -> None:
    value = _anon(shapes=SHAPED)
    record = _protonettype_record_from_value(value)
    assert record.shapes == ("A2",) * 4 + ("",) * 3 + ("I4", "SS4", "T4")  # defaults included
    assert _protonettype_from_record(record) == value and record.label == str(value.label)
    assert record.label.endswith("__8I4~1.2_9SS4~3.4_10~1.2.3.4@4")
    ProtonettypeRecord.__httk_validate__(record)
    assert content_id(record) != content_id(_protonettype_record_from_value(_anon()))  # T4 on both Ag sites
    with pytest.raises(ValueError, match="site 8"):
        _protonettype_from_record(dataclasses.replace(record, shapes=("A2",) * 4 + ("",) * 3 + ("I5", "SS4", "T4")))
    with pytest.raises(ValueError, match="shapes are missing"):
        _protonettype_from_record(dataclasses.replace(record, shapes=()))
    undetermined = _protonettype_record_from_value(Protonettype.undetermined("A4B2CD_oP32_34_X__X"))
    assert undetermined.shapes == () and undetermined.net == ()


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


@contextlib.contextmanager
def _store(engine: str, tmp_path: Any, name: str, *, drop: bool = False) -> Iterator[Any]:
    """A store over one database that ``name`` names again after a reopen; Mongo needs HTTK_TEST_MONGODB_URI.

    ``drop`` removes the Mongo database when the store closes.
    """
    pytest.importorskip("httk.store")
    from httk.store import EntryIdScheme

    ids = EntryIdScheme("httk.test", "1")
    declarations = {ProtonettypeEntry: ProtonettypeRecord, ProtonetstructureEntry: ProtonetstructureRecord}
    if engine != "mongo":
        pytest.importorskip("sqlalchemy")
        if engine == "duckdb":
            pytest.importorskip("duckdb_engine")
        from httk.store import Backend, SqlStore

        with getattr(Backend, engine)(tmp_path / f"{name}.{engine}") as database:
            yield SqlStore(database, entry_ids=ids, entry_records=declarations)
        return
    uri = os.environ.get("HTTK_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HTTK_TEST_MONGODB_URI is not set")
    pymongo = pytest.importorskip("pymongo")
    from httk.store.backend.mongo import MongoDatabase, MongoStore

    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=1000)
    try:
        client.admin.command("ping")
    except Exception as error:
        client.close()
        pytest.skip(f"MongoDB test server is unreachable: {error}")
    try:
        yield MongoStore(MongoDatabase(client, name), entry_ids=ids, entry_records=declarations)
    finally:
        if drop:
            client.drop_database(name)
        client.close()


@pytest.mark.parametrize("engine", ["sqlite", "duckdb", "mongo"])
def test_store_round_trip_after_reopen(tmp_path, engine) -> None:
    values = (_anon(), _anon(ROWS[:-1]), _anon(shapes=SHAPED), _assigned(), _tied(), _tied("S", "Zn"))
    name = f"httk_test_{uuid.uuid4().hex}"
    with _store(engine, tmp_path, name) as store:
        saved = [(value, store.save(value)) for value in values]
    with _store(engine, tmp_path, name, drop=True) as store:
        _check_reopened(store, saved)


def _check_reopened(store: Any, saved: list[Any]) -> None:
    for value, sid in saved:
        record_type = resolve_storage_record(value)
        fetched = store.fetch(record_type, sid, eager=True)
        from_record = _protonettype_from_record if record_type is ProtonettypeRecord else _protonetstructure_from_record
        assert from_record(fetched) == value
        assert content_id(fetched) == content_id(value)
        assert fetched.label == str(value.label)
    for record_type, count in ((ProtonettypeRecord, 3), (ProtonetstructureRecord, 3)):
        searcher = store.searcher()
        searcher.variable(record_type)
        assert searcher.count() == count
    search = store.searcher()
    v = search.variable(ProtonettypeRecord)
    search.add(v.net.operation == 4)  # child-record chaining reaches the bond rows
    labels = sorted(row.v.label for row in search.results(v=v))
    assert labels == sorted(str(value.label) for value in (_anon(), _anon(shapes=SHAPED)))
