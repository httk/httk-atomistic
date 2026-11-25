"""Tests for the structure import records: validation, identity, registration, stores and bulk ingest."""

import os
import uuid
from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction as F
from typing import Any, cast

import pytest
from httk.core import FileRecord, ImportEntry, ImportRecord, ImportReport
from httk.core.register import entry_record_info, known_entry_records, resolve_entry_record
from httk.core.storage import content_id, project_storage_record

from httk.atomistic import (
    ASUStructure,
    ASUStructureImportRecord,
    ASUStructureRecord,
    CIFStructure,
    CIFStructureImportRecord,
    CIFStructureRecord,
    FundamentalDomainStructure,
    FundamentalDomainStructureRecord,
    Species,
    StructureEntry,
    UnitcellStructure,
    UnitcellStructureImportRecord,
    UnitcellStructureRecord,
    WyckoffSite,
    distance_bonds,
)
from httk.atomistic.io.cif.cif_items import CIFDataItem

NA, CL = Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))
REPORTS = (
    ImportReport("warning", "Dropped an auxiliary loop.", "cif", "httk.atomistic.io.cif"),
    ImportReport("info", "Inferred symbols from labels."),
)
FILE = FileRecord("https://example.org/1000000.cif", "1000000.cif", sha256="0" * 64)
RECORDS: dict[type, tuple[type, ...]] = {
    StructureEntry: (ASUStructureRecord, FundamentalDomainStructureRecord, UnitcellStructureRecord, CIFStructureRecord),
    ImportEntry: (ImportRecord, ASUStructureImportRecord, UnitcellStructureImportRecord, CIFStructureImportRecord),
}


def _asu_structure(a: object = F(564, 100), kind: type[Any] = ASUStructure) -> Any:
    return kind(
        [[a, 0, 0], [0, a, 0], [0, 0, a]], 225, [WyckoffSite("a", (), "Na"), WyckoffSite("b", (), "Cl")], [NA, CL]
    )


def _asu(a: object = F(564, 100)) -> ASUStructureRecord:
    return ASUStructureRecord.from_structure(_asu_structure(a))


def _unitcell() -> UnitcellStructureRecord:
    return UnitcellStructureRecord.from_structure(
        UnitcellStructure([[4, 0, 0], [0, 4, 0], [0, 0, 4]], [[0, 0, 0]], (NA,), ("Na",))
    )


def _cif() -> CIFStructureRecord:
    block = CIFStructure("a", (CIFDataItem("_a", "x"),))
    return CIFStructureRecord(**project_storage_record(CIFStructureRecord, block))  # type: ignore[arg-type]


def _imported(source_uri: str = "cif/1/00/00/1000000.cif", **changes: Any) -> ASUStructureImportRecord:
    record = ASUStructureImportRecord(
        source_uri,
        "build-cod 1.0",
        "imported",
        source_item="1000000",
        source_file=FILE,
        reports=REPORTS,
        repaired=True,
        structure=_asu(),
    )
    return replace(record, **changes)


def _failed() -> ASUStructureImportRecord:
    return ASUStructureImportRecord("cif/1/00/00/1000001.cif", "build-cod 1.0", "failed", error="ValueError: no cell")


# ---------------------------------------------------------------------------------------------
# validation and identity
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("cls", "structure"),
    [
        (ASUStructureImportRecord, _asu),
        (UnitcellStructureImportRecord, _unitcell),
        (CIFStructureImportRecord, _cif),
    ],
)
def test_structure_is_required_exactly_when_imported(cls: type[Any], structure: Any) -> None:
    record = cls("x", "i", "imported", structure=structure())
    assert (record.type, record.status) == ("imports", "imported")
    assert cls("x", "i", "excluded", error="duplicate").structure is None
    with pytest.raises(ValueError, match="'structure' must be set exactly when status is 'imported'"):
        cls("x", "i", "imported")
    with pytest.raises(ValueError, match="'structure' must be set exactly when status is 'imported'"):
        cls("x", "i", "failed", error="boom", structure=structure())
    with pytest.raises(TypeError, match="'structure' must be a"):
        cls("x", "i", "imported", structure=object())


def test_wrong_structure_types_and_base_validation() -> None:
    with pytest.raises(TypeError, match="must be a ASUStructureRecord or None, got UnitcellStructureRecord"):
        ASUStructureImportRecord("x", "i", "imported", structure=cast(Any, _unitcell()))
    with pytest.raises(TypeError, match="must be a UnitcellStructureRecord or None, got ASUStructure"):
        UnitcellStructureImportRecord(
            "x", "i", "imported", structure=cast(Any, ASUStructure([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 1, [], []))
        )
    with pytest.raises(TypeError, match="'error' must be a str"):
        ASUStructureImportRecord("x", "i", "failed")
    with pytest.raises(ValueError, match="'error' must be None"):
        ASUStructureImportRecord("x", "i", "imported", error="boom", structure=_asu())
    with pytest.raises(ValueError, match="status"):
        ASUStructureImportRecord("x", "i", "done", structure=_asu())
    with pytest.raises(ValueError, match="repaired"):
        ASUStructureImportRecord("x", "i", "failed", error="boom", repaired=True)


def test_golden_content_ids() -> None:
    assert content_id(_imported()) == "a69ce56be6a21b5ce11f216b5979517379029969c5ff21639cd16808b4592fc8"
    assert content_id(_failed()) == "4604afaed70ae30ebb355c368a1f2de59f99ebebab8f7f1dea1e1df368d9d8d3"
    base = ImportRecord("cif/1/00/00/1000001.cif", "build-cod 1.0", "failed", error="ValueError: no cell")
    assert content_id(base) != content_id(_failed())
    assert content_id(_imported(structure=_asu(F(565, 100)))) != content_id(_imported())
    assert content_id(_imported(reports=REPORTS[:1])) != content_id(_imported())


# ---------------------------------------------------------------------------------------------
# record factories
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("record_type", "kind"),
    [(ASUStructureRecord, ASUStructure), (FundamentalDomainStructureRecord, FundamentalDomainStructure)],
)
def test_from_structure_builds_the_record_the_store_builds(store: Any, record_type: type[Any], kind: type[Any]) -> None:
    structure = distance_bonds(_asu_structure(kind=kind))
    record = record_type.from_structure(structure)
    assert type(record) is record_type
    assert content_id(record) == content_id(structure)
    assert store.fetch(record_type, store.save(structure), eager=True) == record


def test_from_structure_follows_the_storage_opt_in() -> None:
    assert type(FundamentalDomainStructureRecord.from_structure(_asu_structure())) is ASUStructureRecord
    with pytest.raises(TypeError, match="ASUStructureRecord does not record a FundamentalDomainStructure"):
        ASUStructureRecord.from_structure(_asu_structure(kind=FundamentalDomainStructure))
    unitcell = UnitcellStructure([[4, 0, 0], [0, 4, 0], [0, 0, 4]], [[0, 0, 0]], (NA,), ("Na",))
    with pytest.raises(TypeError, match="ASUStructureRecord does not record a UnitcellStructure"):
        ASUStructureRecord.from_structure(cast(Any, unitcell))
    with pytest.raises(TypeError, match="FundamentalDomainStructureRecord does not record a str"):
        FundamentalDomainStructureRecord.from_structure(cast(Any, "Na"))
    with pytest.raises(TypeError, match="does not record a ASUStructureRecord"):
        ASUStructureRecord.from_structure(cast(Any, _asu()))


# ---------------------------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------------------------
def test_registration() -> None:
    expected = {
        "atomistic-asu-structure-import": ASUStructureImportRecord,
        "atomistic-unitcell-structure-import": UnitcellStructureImportRecord,
        "atomistic-cif-structure-import": CIFStructureImportRecord,
    }
    assert set(expected) | {"core-import"} <= set(known_entry_records(family="imports"))
    for name, cls in expected.items():
        assert resolve_entry_record(name) is cls
        assert entry_record_info(name)[1] == "imports"


# ---------------------------------------------------------------------------------------------
# stores
# ---------------------------------------------------------------------------------------------
@pytest.fixture(params=["sqlite", "duckdb", "mongo"])
def store(request: pytest.FixtureRequest) -> Iterator[Any]:
    pytest.importorskip("httk.store")
    from httk.store import EntryIdScheme

    ids = EntryIdScheme("httk.test", "1")
    if request.param in ("sqlite", "duckdb"):
        pytest.importorskip("sqlalchemy")
        if request.param == "duckdb":
            pytest.importorskip("duckdb_engine")
        from httk.store import Backend, SqlStore

        with getattr(Backend, request.param)() as database:
            yield SqlStore(database, entry_ids=ids, entry_records=RECORDS)
        return
    uri = os.environ.get("HTTK_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HTTK_TEST_MONGODB_URI is not set")
    pymongo = pytest.importorskip("pymongo")
    from httk.store.backend.mongo import MongoDatabase, MongoStore

    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=1000)
    name = f"httk_test_{uuid.uuid4().hex}"
    try:
        client.admin.command("ping")
    except Exception as error:
        client.close()
        pytest.skip(f"MongoDB test server is unreachable: {error}")
    try:
        yield MongoStore(MongoDatabase(client, name), entry_ids=ids, entry_records=RECORDS)
    finally:
        client.drop_database(name)
        client.close()


def test_a_store_round_trips_import_records(store: Any) -> None:
    records = (
        _imported(),
        _failed(),
        UnitcellStructureImportRecord("u", "i", "imported", structure=_unitcell()),
        CIFStructureImportRecord("c", "i", "imported", structure=_cif()),
        ImportRecord("g", "i", "failed", error="unknown kind"),
    )
    for record in records:
        sid = store.save(record)
        fetched = store.fetch(type(record), sid, eager=True)
        assert fetched == record and content_id(fetched) == content_id(record)
        structure = getattr(record, "structure", None)
        if structure is not None:
            assert fetched.structure.id is not None
            assert store.fetch(type(structure), store.sid_of(fetched.structure), eager=True) == structure


@pytest.fixture(params=[("sqlite", "parity"), ("sqlite", "deferred"), ("duckdb", "parity"), ("duckdb", "deferred")])
def bulk_store(request: pytest.FixtureRequest) -> Iterator[tuple[Any, str]]:
    pytest.importorskip("sqlalchemy")
    backend, finalize = request.param
    if backend == "duckdb":
        pytest.importorskip("duckdb_engine")
    from httk.store import Backend, EntryIdScheme, SqlStore

    with getattr(Backend, backend)() as database:
        yield SqlStore(database, entry_ids=EntryIdScheme("httk.test", "1"), entry_records=RECORDS), finalize


def test_bulk_ingest_promotes_the_structure(bulk_store: tuple[Any, str]) -> None:
    store, finalize = bulk_store
    first, second = _imported(), _imported("cif/9/00/00/9000000.cif")
    with store.bulk_ingest(finalize=finalize) as bulk:
        for record in (first, second, _failed()):
            bulk.save(record, promote=ASUStructureRecord)

    searcher = store.searcher()
    structure = searcher.variable(ASUStructureRecord)
    (stored,) = [row.values[0] for row in searcher.results(structure=structure)]
    assert stored == _asu() and stored.id is not None
    searcher = store.searcher()
    imported = searcher.variable(ASUStructureImportRecord)
    searcher.add(imported.status == "failed")
    assert [row.values[0] for row in searcher.results(imported=imported)] == [_failed()]

    searcher = store.searcher()
    imported = searcher.variable(ASUStructureImportRecord)
    searcher.add(imported.structure.spacegroup_it_number == 225)
    assert sorted(row.values[0].source_uri for row in searcher.results(imported=imported)) == sorted(
        (first.source_uri, second.source_uri)
    )

    searcher = store.searcher()
    imported = searcher.variable(ASUStructureImportRecord)
    structure = searcher.variable(ASUStructureRecord)
    searcher.add(imported.structure == structure)
    searcher.add(imported.source_uri == second.source_uri)
    assert [row.values[0] for row in searcher.results(structure=structure)] == [_asu()]

    referring = store.referring(ASUStructureImportRecord, field="structure", to=stored, eager=True)
    assert sorted(record.source_uri for record in referring) == sorted((first.source_uri, second.source_uri))


@pytest.mark.parametrize("mode", ["save", "bulk", "bulk-promote"])
def test_the_embedded_structure_is_a_main_entry_only_when_promoted(bulk_store: tuple[Any, str], mode: str) -> None:
    import sqlalchemy
    from httk.store.backend.schema import resolve_schema

    store, finalize = bulk_store
    if mode == "save":
        store.save(_imported())
    else:
        with store.bulk_ingest(finalize=finalize) as bulk:
            bulk.save(_imported(), promote=ASUStructureRecord if mode == "bulk-promote" else None)
    # Every mode stores the structure once, with an entry id, fetchable and searchable on its own ...
    structure = store.fetch_entry(StructureEntry, content_id(_asu()), eager=True)
    assert structure == _asu() and structure.id is not None
    searcher = store.searcher()
    searcher.variable(ASUStructureRecord)
    assert searcher.count() == 1
    # ... but only a promoted one is a main row (role 1) rather than a dependency of the import record.
    table = store._table(resolve_schema(ASUStructureRecord).table_name)
    with store._read_connection() as connection:
        roles = connection.execute(sqlalchemy.select(table.c["_httk_role"])).scalars().all()
    assert roles == [1 if mode == "bulk-promote" else 0]
