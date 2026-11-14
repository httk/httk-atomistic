"""Storage of bond networks: unit-cell network records, bond sources, identity and store upgrades.

A unit-cell network is a referenced ``UnitcellBondNetworkRecord`` child; a fundamental domain keeps
its flat ``bonds``/``shapes`` fields. The sources are ``IdentitySkip`` references on the structure
record, so they never enter a content id and equal networks share one child row.
"""

import contextlib
import gzip
import logging
import os
import shutil
import uuid
from collections.abc import Callable, Iterator
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import pytest
from httk.core import FracVector
from httk.core.storage import content_id

from httk.atomistic import (
    ASUStructure,
    BondNetwork,
    BondSource,
    FundamentalDomainStructure,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    distance_bonds,
)
from httk.atomistic.entries.structures import StructureEntry
from httk.atomistic.models.structure.record import RecordStructure
from httk.atomistic.storage.records import (
    ASUStructureRecord,
    BondSourceRecord,
    FundamentalDomainStructureRecord,
    UnitcellBondNetworkRecord,
    UnitcellStructureRecord,
    _domain_structure_record_from_value,
    _structure_from_record,
    _unitcell_record_from_structure,
    validate_structure_record,
)

FIXTURES = Path(__file__).parent / "fixtures"
NA, CL = Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))
SOURCE = BondSource("hand-assigned", "test", "1")
SHAPES_SOURCE = BondSource(software="shapes")
RECORDS = {StructureEntry: (UnitcellStructureRecord, FundamentalDomainStructureRecord, ASUStructureRecord)}


def _rocksalt(a: object = F(564, 100)) -> ASUStructure:
    plain = ASUStructure(
        [[a, 0, 0], [0, a, 0], [0, 0, a]], 225, [WyckoffSite("a", (), "Na"), WyckoffSite("b", (), "Cl")], [NA, CL]
    )
    return distance_bonds(plain)


def _with(structure: Any, shapes: Any = None, source: Any = None, shapes_source: Any = None) -> Any:
    assert structure.bonds is not None
    return structure._with_bonds(BondNetwork(structure.bonds.pairs, shapes, source, shapes_source))


def _domain(kind: type[Any], network: bool = True) -> Any:
    asu = _rocksalt()
    return kind(asu.cell, asu.spacegroup, asu.wyckoff_sites, asu.species, bonds=asu.bonds if network else None)


def _cell(network: bool = True) -> UnitcellStructure:
    cell = UnitcellStructureView(_rocksalt()).unview()
    return cell if network else cell._with_bonds(None)


def _sources(network: BondNetwork | None) -> tuple[object, object]:
    return (None, None) if network is None else (network.source, network.shapes_source)


def _variants(structure: Any) -> list[Any]:
    """No network, an empty one (unit cell), and the network with each shape state and sources."""
    nsites = len(structure.wyckoff_sites) if isinstance(structure, FundamentalDomainStructure) else structure.nsites
    partial = ("OC6",) + (None,) * (nsites - 1)
    values = [
        structure._with_bonds(None),
        _with(structure),
        _with(structure, partial, SOURCE),
        _with(structure, ("OC6",) * nsites, SOURCE, SHAPES_SOURCE),
    ]
    if not isinstance(structure, FundamentalDomainStructure):
        values.append(structure._with_bonds(BondNetwork((), source=SOURCE)))
    return values


ALL = {
    "unit cell": lambda: _cell(),
    "fundamental domain": lambda: _domain(FundamentalDomainStructure),
    "asymmetric unit": lambda: _domain(ASUStructure),
}


def _record(structure: Any) -> Any:
    if isinstance(structure, FundamentalDomainStructure):
        return _domain_structure_record_from_value(structure)
    return _unitcell_record_from_structure(structure)


# ---------------------------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL)
def test_records_round_trip_networks_shapes_and_sources(name: str) -> None:
    for value in _variants(ALL[name]()):
        record = _record(value)
        type(record).__httk_validate__(record)
        back = _structure_from_record(record)
        assert back == value and _sources(back.bonds) == _sources(value.bonds)
        assert content_id(record) == content_id(value)
        assert RecordStructure(record).bonds == UnitcellStructureView(value).bonds


def test_a_unit_cell_record_stores_its_network_compactly() -> None:
    cell = _with(_cell(), None, SOURCE)
    record = _unitcell_record_from_structure(cell)
    assert isinstance(record.bonds, UnitcellBondNetworkRecord)
    assert record.bonds.pairs.to_fractions() == [
        [bond.site, bond.partner, *bond.operation.vector.to_fractions()] for bond in cell.bonds.pairs
    ]
    assert record.bonds_source == BondSourceRecord("hand-assigned", "test", "1") and record.shapes_source is None
    empty = _unitcell_record_from_structure(cell._with_bonds(BondNetwork(())))
    assert empty.bonds == UnitcellBondNetworkRecord(FracVector(())) and empty.bonds_source is None


def test_a_unit_cell_record_with_an_invalid_stored_shape_serves_null(caplog: pytest.LogCaptureFixture) -> None:
    from httk.atomistic.storage.stored_properties import _response_value

    nsites = _cell().nsites
    record = replace(_unitcell_record_from_structure(_with(_cell(), ("OC6",) * nsites, SOURCE)), id="x")
    assert _response_value(record, "_httk_bonds", "unitcell")["shapes"] == {
        "sites": list(range(nsites)),
        "symbols": ["OC6"] * nsites,
    }
    # a symbol outside the vocabulary, or one for the wrong bond count
    for shape, message in (("I6", "'I6' is no longer a shape symbol; write 'I'"), ("T4", "is for 4 bonds")):
        stale = replace(record, bonds=replace(record.bonds, shapes=(shape,) + ("OC6",) * (nsites - 1)))
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="httk.atomistic.storage.stored_properties"):
            assert _response_value(stale, "_httk_bonds", "unitcell") is None
        assert "the bond network of entry x is served as null: shapes[0]: " in caplog.text and message in caplog.text


@pytest.mark.parametrize("kind", [FundamentalDomainStructure, ASUStructure])
def test_a_domain_record_with_an_invalid_stored_shape_serves_null(
    kind: type[Any], caplog: pytest.LogCaptureFixture
) -> None:
    from httk.atomistic.storage.stored_properties import _response_value

    record = replace(_domain_structure_record_from_value(_with(_domain(kind), ("OC6", "OC6"), SOURCE)), id="x")
    assert _response_value(record, "_httk_bonds", "domain")["shapes"] == {"sites": [0, 1], "symbols": ["OC6"] * 2}
    stale = replace(record, shapes=("I6", "OC6"))
    with caplog.at_level(logging.WARNING, logger="httk.atomistic.storage.stored_properties"):
        assert _response_value(stale, "_httk_bonds", "domain") is None
    assert (
        "the bond network of entry x is served as null: " in caplog.text
        and "'I6' is no longer a shape symbol" in caplog.text
    )


def test_shape_states_are_stored_as_in_the_domain_fields() -> None:
    cell = _cell()
    nsites = cell.nsites
    stored = [
        _unitcell_record_from_structure(_with(cell, shapes)).bonds.shapes
        for shapes in (None, ("OC6",) + (None,) * (nsites - 1), ("OC6",) * nsites)
    ]
    assert stored == [None, ("OC6",) + ("",) * (nsites - 1), ("OC6",) * nsites]


def test_content_identity_ignores_sources_and_follows_pairs_and_shapes() -> None:
    for make in ALL.values():
        plain = make()._with_bonds(None)
        values = _variants(make())
        assert content_id(values[0]) == content_id(plain)
        assert content_id(_with(values[1], None, SOURCE)) == content_id(values[1]) != content_id(plain)
        assert content_id(_with(values[2], values[2].bonds.shapes, BondSource(software="other"))) == content_id(
            values[2]
        )
        assert len({content_id(value) for value in values}) == len(values)
    # without a network a unit cell keeps the id it had before networks were stored (httk-atomistic a2db444)
    cod = UnitcellStructureView(distance_bonds(_load_cod("1000008"))).unview()._with_bonds(None)
    assert content_id(cod) == "010ca29aa0f7a136a4b784be449518accb5a1d1f61d444d73bc6a2bc0deea5e4"


def test_a_shapes_source_equal_to_the_source_is_one_spelling() -> None:
    cell = _cell()
    shapes = ("OC6",) * cell.nsites
    network = BondNetwork(cell.bonds.pairs, shapes, SOURCE, BondSource("hand-assigned", "test", "1"))
    assert network.shapes_source is None and network == BondNetwork(cell.bonds.pairs, shapes, SOURCE)
    for value in (_with(cell, shapes, SOURCE, SOURCE), _with(_domain(ASUStructure), ("OC6", "OC6"), SOURCE, SOURCE)):
        assert value.bonds.shapes_source is None
        record = _record(value)
        assert record.shapes_source is None
        type(record).__httk_validate__(record)
        with pytest.raises(ValueError, match="shapes_source is not in its normal form"):
            type(record).__httk_validate__(replace(record, shapes_source=record.bonds_source))


def _load_cod(name: str) -> Any:
    import httk.core

    return httk.core.load(str(FIXTURES / "cod" / f"{name}.cif"))


def test_bad_records_are_rejected() -> None:
    cell = _with(_cell(), ("OC6",) * _cell().nsites, SOURCE, SHAPES_SOURCE)
    record = _unitcell_record_from_structure(cell)
    rows = record.bonds.pairs.to_fractions()
    i, j, *n = rows[0]
    reversed_first = [[j, i, *(-x for x in n)], *rows[1:]]
    with pytest.raises(ValueError, match="not the normalized bond network"):
        validate_structure_record(replace(record, bonds=replace(record.bonds, pairs=FracVector(reversed_first))))
    with pytest.raises(ValueError, match="not the normalized bond network"):
        validate_structure_record(replace(record, bonds=replace(record.bonds, pairs=FracVector(rows[::-1]))))
    with pytest.raises(ValueError, match="shapes are not in their normal form"):
        validate_structure_record(
            replace(record, shapes_source=None, bonds=replace(record.bonds, shapes=("",) * cell.nsites))
        )
    with pytest.raises(ValueError, match="integers"):
        UnitcellBondNetworkRecord(FracVector([[0, 1, F(1, 2), 0, 0]]))
    with pytest.raises(ValueError, match="non-negative"):
        UnitcellBondNetworkRecord(FracVector([[0, -1, 0, 0, 0]]))
    with pytest.raises(ValueError, match="five columns"):
        UnitcellBondNetworkRecord(FracVector([[0, 1, 0, 0]]))
    with pytest.raises(ValueError, match="shapes_source needs shapes"):
        replace(record, bonds=replace(record.bonds, shapes=None))
    with pytest.raises(ValueError, match="bonds_source needs bonds"):
        replace(record, bonds=None, shapes_source=None)
    with pytest.raises(ValueError, match="at least one field"):
        BondSourceRecord()
    with pytest.raises(ValueError, match="blank"):
        BondSourceRecord(software=" ")
    with pytest.raises(TypeError, match="BondSourceRecord"):
        replace(record, bonds_source="test")
    domain = _domain_structure_record_from_value(_with(_domain(ASUStructure), ("OC6", "OC6"), SOURCE, SHAPES_SOURCE))
    ASUStructureRecord.__httk_validate__(domain)
    with pytest.raises(ValueError, match="shapes_source needs shapes"):
        replace(domain, shapes=None)
    with pytest.raises(ValueError, match="bonds_source needs bonds"):
        replace(domain, bonds=None, shapes=None, shapes_source=None)
    with pytest.raises(ValueError, match="shapes are not in their normal form"):
        ASUStructureRecord.__httk_validate__(replace(domain, shapes=("", ""), shapes_source=None))


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


def _record_type(value: Any) -> type[Any]:
    if isinstance(value, ASUStructure):
        return ASUStructureRecord
    return (
        FundamentalDomainStructureRecord if isinstance(value, FundamentalDomainStructure) else UnitcellStructureRecord
    )


def test_a_store_round_trips_every_record_type_with_networks_and_sources(store: Any) -> None:
    for make in ALL.values():
        for value in _variants(make()):
            sid = store.save(value)
            for fetched in (store.fetch(_record_type(value), sid, eager=True), store.fetch(_record_type(value), sid)):
                back = _structure_from_record(fetched)
                assert back == value and _sources(back.bonds) == _sources(value.bonds)
                assert content_id(fetched) == content_id(value)


def test_equal_networks_with_different_sources_both_save(store: Any) -> None:
    # two different crystals with the same unit-cell network: the network row is shared, the sources are not
    first = _with(UnitcellStructureView(_rocksalt()).unview(), None, BondSource(software="one"))
    second = _with(UnitcellStructureView(_rocksalt(F(560, 100))).unview(), None, BondSource(software="two"))
    assert first.bonds == second.bonds and content_id(first) != content_id(second)
    for value in (first, second):
        back = _structure_from_record(store.fetch(UnitcellStructureRecord, store.save(value), eager=True))
        assert back.bonds is not None and back.bonds.source == value.bonds.source


def test_resaving_equal_content_with_another_source_raises(store: Any) -> None:
    from httk.store.store_common import EntryMetadataConflictError

    for make in ALL.values():
        value = _with(make(), None, BondSource(software="one"))
        other = _with(value, None, BondSource(software="two"))
        record_type = _record_type(value)
        sid = store.save(value)
        with pytest.raises(EntryMetadataConflictError, match="bonds_source"):
            store.save(other)
        with pytest.raises(EntryMetadataConflictError, match="bonds_source"):
            store.replace(store.fetch(record_type, sid), other)
        if hasattr(store, "bulk_ingest"):
            with pytest.raises(EntryMetadataConflictError, match="bonds_source"), store.bulk_ingest() as bulk:
                bulk.save(other)
        stored = _structure_from_record(store.fetch(record_type, sid, eager=True)).bonds
        assert stored is not None and stored.source == BondSource(software="one")


# ---------------------------------------------------------------------------------------------
# upgrading a develop-era (protocol-3) store
# ---------------------------------------------------------------------------------------------
# Built at httk-atomistic a2db444 (httk-core 6cbb302, httk-store d4ed802) by fixtures/stores/build_a2db444_store.py,
# one fixture per backend with the same entries (see fixtures/stores/README.md).
_A2DB444 = {
    "asu_network": (ASUStructureRecord, 1, "4a61584d6a062827e197cd228b05c58786cf531473ec7fb8afb0e8121d101d3e", 11),
    "asu_network_2": (ASUStructureRecord, 2, "eecfc84f8957d7fc909823a424ed9e49c7f2462870089929ed0c2f5f3b89f43a", 3),
    "fd_network_shapes": (
        FundamentalDomainStructureRecord,
        1,
        "41c05493ef875def99cf34de8f5218e0502f13f4df312a077fecec078c9219f2",
        3,
    ),
    "asu_plain": (ASUStructureRecord, 3, "3b84530e7bb57778d85585aba1316711877628ddc01d05e212622830fc85bd9c", None),
    "unitcell_plain": (
        UnitcellStructureRecord,
        1,
        "010ca29aa0f7a136a4b784be449518accb5a1d1f61d444d73bc6a2bc0deea5e4",
        None,
    ),
}


@contextlib.contextmanager
def _a2db444(backend: str, tmp_path: Path) -> Iterator[Callable[[bool], contextlib.AbstractContextManager[Any]]]:
    """Restore the a2db444 fixture of ``backend`` and yield a function opening it (with or without ``upgrade``)."""
    pytest.importorskip("httk.store")
    from httk.store import EntryIdScheme

    ids = EntryIdScheme("httk.test", "1")
    fixture = FIXTURES / "stores" / f"a2db444_domain_networks.{backend if backend != 'mongo' else 'mongo.json'}.gz"
    if backend in ("sqlite", "duckdb"):
        pytest.importorskip("sqlalchemy")
        if backend == "duckdb":
            pytest.importorskip("duckdb_engine")
        from httk.store import Backend, SqlStore

        path = tmp_path / f"a2db444.{backend}"
        with gzip.open(fixture) as source, path.open("wb") as target:
            shutil.copyfileobj(source, target)

        @contextlib.contextmanager
        def sql(upgrade: bool) -> Iterator[Any]:
            with getattr(Backend, backend)(path) as database:
                yield SqlStore(database, entry_ids=ids, entry_records=RECORDS, upgrade=upgrade)

        yield sql
        return
    uri = os.environ.get("HTTK_TEST_MONGODB_URI")
    if not uri:
        pytest.skip("HTTK_TEST_MONGODB_URI is not set")
    pymongo = pytest.importorskip("pymongo")
    from bson import json_util
    from httk.store.backend.mongo import MongoDatabase, MongoStore

    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=1000)
    name = f"httk_test_{uuid.uuid4().hex}"
    try:
        client.admin.command("ping")
    except Exception as error:
        client.close()
        pytest.skip(f"MongoDB test server is unreachable: {error}")
    try:
        with gzip.open(fixture, "rt") as handle:
            dump = json_util.loads(handle.read())
        for collection, content in dump.items():
            if content["documents"]:
                client[name][collection].insert_many(content["documents"])
            else:
                client[name].create_collection(collection)
            for index in content["indexes"]:
                if index["name"] != "_id_":
                    options = {key: value for key, value in index.items() if key not in ("v", "key", "ns")}
                    client[name][collection].create_index(list(index["key"].items()), **options)

        @contextlib.contextmanager
        def mongo(upgrade: bool) -> Iterator[Any]:
            yield MongoStore(MongoDatabase(client, name), entry_ids=ids, entry_records=RECORDS, upgrade=upgrade)

        yield mongo
    finally:
        client.drop_database(name)
        client.close()


@pytest.mark.parametrize("backend", ["sqlite", "duckdb", "mongo"])
def test_a_protocol_3_store_with_domain_networks_upgrades_additively(backend: str, tmp_path: Path) -> None:
    from httk.store.storage_layout import StorageLayoutUpgradeRequiredError

    with _a2db444(backend, tmp_path) as opened:
        with pytest.raises(StorageLayoutUpgradeRequiredError, match="purely additive"), opened(False) as store:
            store.fetch(ASUStructureRecord, 1)
        with opened(True) as store:
            for record_type, sid, expected, nbonds in _A2DB444.values():
                for eager in (True, False):
                    record = store.fetch(record_type, sid, eager=eager)
                    structure = _structure_from_record(record)
                    assert content_id(record) == content_id(structure) == expected
                    assert (None if structure.bonds is None else len(structure.bonds.pairs)) == nbonds
                    assert _sources(structure.bonds) == (None, None)
            shaped = _structure_from_record(store.fetch(FundamentalDomainStructureRecord, 1, eager=True)).bonds
            assert shaped is not None and shaped.shapes is not None and None in shaped.shapes
            # the upgraded store takes unit-cell networks and sources
            cell = _with(_cell(), None, SOURCE)
            back = _structure_from_record(store.fetch(UnitcellStructureRecord, store.save(cell), eager=True))
            assert back == cell and back.bonds is not None and back.bonds.source == SOURCE
        with opened(False) as store:  # and reopens without another upgrade
            store.fetch(ASUStructureRecord, 1)
