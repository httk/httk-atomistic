"""The dimensions of the bond network: per site, per structure, in records, served and filtered."""

import contextlib
import dataclasses
import itertools
import os
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httk.core
import pytest
from httk.core.storage import content_id
from test_bond_network_optimade import SERVED

from httk.atomistic import (
    AffineOperation,
    ASUStructure,
    ASUStructureView,
    Cell,
    CIFStructure,
    FundamentalDomainStructure,
    NumericUnitcellStructure,
    SiteBond,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    distance_bonds,
    site_bonds_dimensions,
    validate_structure_record,
)
from httk.atomistic.entries.bonds import bond_properties
from httk.atomistic.entries.structures import StructureEntry
from httk.atomistic.models.structure.record import RecordStructure
from httk.atomistic.storage.records import (
    ASUStructureRecord,
    FundamentalDomainStructureRecord,
    NumericUnitcellStructureRecord,
    UnitcellStructureRecord,
    _domain_structure_record_from_value,
    _unitcell_record_from_structure,
)

FIXTURES = Path(__file__).parent / "fixtures"
C = Species("C", ("C",), (1,))
RECORDS = {StructureEntry: (UnitcellStructureRecord, FundamentalDomainStructureRecord, ASUStructureRecord)}


def _bond(site: int, partner: int, t: str = "x,y,z") -> SiteBond:
    return SiteBond(site, partner, AffineOperation(t))


def _cell(
    nsites: int, *bonds: SiteBond, periodicity: tuple[bool, bool, bool] = (True, True, True)
) -> UnitcellStructure:
    """``nsites`` carbon atoms along the cell diagonal of a 4 x 4 x 20 Å cell, with the given bonds."""
    cell = Cell([[4, 0, 0], [0, 4, 0], [0, 0, 20]], periodicity=periodicity)
    sites = [[i / 10, i / 10, i / 10] for i in range(nsites)]
    return UnitcellStructure(cell, sites, (C,), ("C",) * nsites, bonds=bonds)


def _rocksalt() -> ASUStructure:
    plain = ASUStructure(
        Cell([[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]),
        225,
        [WyckoffSite("a", (), "Na"), WyckoffSite("b", (), "Cl")],
        [Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))],
    )
    return distance_bonds(plain)


LAYER = _cell(1, _bond(0, 0, "x+1,y,z"), _bond(0, 0, "x,y+1,z"))
CHAIN = _cell(2, _bond(0, 1), _bond(1, 0, "x+1,y,z"))  # a ring closed only through a translation
MOLECULE = _cell(2, _bond(0, 1))
# a framework atom, a two-atom molecule and an unbonded atom
MIXED = _cell(4, *(_bond(0, 0, t) for t in ("x+1,y,z", "x,y+1,z", "x,y,z+1")), _bond(1, 2))


@pytest.mark.parametrize(
    ("structure", "dimensions", "per_site"),
    (
        (LAYER, (2,), (2,)),
        (CHAIN, (1,), (1, 1)),
        (MOLECULE, (0,), (0, 0)),
        (MIXED, (0, 3), (3, 0, 0, None)),
        # three translations spanning only a plane
        (_cell(1, *(_bond(0, 0, t) for t in ("x+1,y,z", "x,y+1,z", "x+1,y+1,z"))), (2,), (2,)),
        # a slab: no translation along its non-periodic direction, so at most 2
        (_cell(1, _bond(0, 0, "x+1,y,z"), _bond(0, 0, "x,y+1,z"), periodicity=(True, True, False)), (2,), (2,)),
        (_cell(2), (), (None, None)),  # an empty network
    ),
)
def test_unit_cell_dimensions(structure: UnitcellStructure, dimensions: tuple[int, ...], per_site: tuple[Any]) -> None:
    assert structure.bonds_dimensions == dimensions
    assert site_bonds_dimensions(structure) == per_site
    assert UnitcellStructureView(structure).bonds_dimensions == dimensions


def test_a_structure_without_a_network() -> None:
    plain = MIXED._with_bonds(None)
    assert plain.bonds_dimensions is None
    with pytest.raises(ValueError, match="no bond network"):
        site_bonds_dimensions(plain)


def test_a_domain_and_its_unit_cell_agree() -> None:
    asu = _rocksalt()
    assert asu.bonds_dimensions == (3,)
    assert site_bonds_dimensions(asu) == (3, 3)  # per Wyckoff site
    cell = UnitcellStructureView(asu)
    assert cell.bonds_dimensions == (3,) and site_bonds_dimensions(cell) == (3,) * 8  # per site
    domain = FundamentalDomainStructure(asu.cell, asu.spacegroup, asu.wyckoff_sites, asu.species, bonds=asu.bonds)
    assert domain.bonds_dimensions == (3,)
    assert asu._with_bonds(None).bonds_dimensions is None


def test_a_molecular_crystal_and_a_recognised_view() -> None:
    loaded = httk.core.load(str(FIXTURES / "cod" / "7217521.cif"))
    view = ASUStructureView(loaded._with_bonds(None) if loaded.bonds is not None else loaded)
    bonded = distance_bonds(view)
    assert bonded.bonds_dimensions == (0,)
    assert set(site_bonds_dimensions(bonded)) == {0}
    assert ASUStructureView(bonded).bonds_dimensions == (0,)


DOMAINS = {
    "rock salt": _rocksalt,
    **{name: make for name, make in SERVED.items() if name != "a unit cell"},  # settings, transforms, members
    **{
        entry: (
            lambda entry=entry: distance_bonds(ASUStructureView(httk.core.load(str(FIXTURES / "cod" / f"{entry}.cif"))))
        )
        for entry in ("1535669", "1539945", "7219077")  # layers, chains, a molecule on a 2-fold axis
    },
}


@pytest.mark.parametrize("name", DOMAINS)
def test_a_domain_gives_what_its_unit_cell_expansion_gives(name: str) -> None:
    domain = DOMAINS[name]()
    per_site = site_bonds_dimensions(UnitcellStructureView(domain))
    counts = domain.multiplicities()
    starts = list(itertools.accumulate(counts, initial=0))
    blocks = [set(per_site[start:end]) for start, end in itertools.pairwise(starts)]
    assert all(len(block) == 1 for block in blocks)  # one dimension per Wyckoff site
    assert site_bonds_dimensions(domain) == tuple(block.pop() for block in blocks)
    assert domain.bonds_dimensions == UnitcellStructureView(domain).bonds_dimensions


def test_a_cif_with_a_bond_loop() -> None:
    cif = CIFStructure.from_cif(FIXTURES / "cif_bonds" / "ordered.cif")
    assert cif.bonds is not None
    assert cif.bonds_dimensions == UnitcellStructureView(cif).bonds_dimensions == cif.resolve().bonds_dimensions
    assert len(site_bonds_dimensions(cif)) == len(cif.sites.reduced_coords.to_fractions())


# ---------------------------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------------------------
def test_records_store_the_dimensions_outside_the_content_identity() -> None:
    cell = _unitcell_record_from_structure(MIXED)
    domain = _domain_structure_record_from_value(_rocksalt())
    assert (cell.bonds_dimensions_mask, domain.bonds_dimensions_mask) == (0b1001, 0b1000)  # bit d: dimension d
    assert _unitcell_record_from_structure(_cell(2)).bonds_dimensions_mask == 0  # no bonds
    assert _unitcell_record_from_structure(MIXED._with_bonds(None)).bonds_dimensions_mask is None
    assert RecordStructure(cell).bonds_dimensions == (0, 3) and RecordStructure(domain).bonds_dimensions == (3,)
    assert site_bonds_dimensions(RecordStructure(domain)) == (3,) * 8  # a record presents its unit cell
    for record, structure in ((cell, MIXED), (domain, _rocksalt())):
        validate_structure_record(record)
        assert content_id(record) == content_id(structure)
        wrong = dataclasses.replace(record, bonds_dimensions_mask=0b10)
        assert content_id(wrong) == content_id(record)
        with pytest.raises(ValueError, match="bonds_dimensions_mask does not hold the dimensions"):
            validate_structure_record(wrong)
    with pytest.raises(ValueError, match="bonds_dimensions_mask needs bonds"):
        dataclasses.replace(cell, bonds=None, bonds_source=None, shapes_source=None)
    with pytest.raises(TypeError, match="bonds_dimensions_mask must be an integer"):
        dataclasses.replace(cell, bonds_dimensions_mask=True)
    with pytest.raises(ValueError, match=r"bonds_dimensions_mask must be in 0\.\.15"):
        dataclasses.replace(cell, bonds_dimensions_mask=16)


# ---------------------------------------------------------------------------------------------
# serving and filtering
# ---------------------------------------------------------------------------------------------
def test_the_served_value_carries_the_dimensions() -> None:
    assert bond_properties(MIXED)["_httk_bonds"]["dimensions"] == [0, 3]
    assert bond_properties(_rocksalt())["_httk_bonds"]["dimensions"] == [3]
    assert bond_properties(_cell(2))["_httk_bonds"]["dimensions"] == []


STORED = {
    "rocksalt domain": _rocksalt,
    "layer": lambda: LAYER,
    "chain": lambda: CHAIN,
    "molecule": lambda: MOLECULE,
    "mixed": lambda: MIXED,
    "empty": lambda: _cell(3),
    "plain": lambda: MIXED._with_bonds(None),
}


def _saved(store: Any) -> dict[str, str]:
    """Save every structure of ``STORED`` and return the map from entry id to name."""
    saved = {}
    for name, make in STORED.items():
        structure = make()
        kind = ASUStructureRecord if isinstance(structure, ASUStructure) else UnitcellStructureRecord
        saved[store.fetch(kind, store.save(structure)).id] = name
    return saved


@contextlib.contextmanager
def _store(backend: str, directory: Path) -> Iterator[tuple[Any, Callable[[], None]]]:
    """Open an empty store of ``backend`` and a function that nulls the stored dimensions of every row."""
    pytest.importorskip("httk.store")
    from httk.store import EntryIdScheme

    ids = EntryIdScheme("httk.test", "1")
    tables = ("atomistic_unitcell_structure", "atomistic_asu_structure")
    if backend in ("sqlite", "duckdb"):
        sqlalchemy = pytest.importorskip("sqlalchemy")
        if backend == "duckdb":
            pytest.importorskip("duckdb_engine")
        from httk.store import Backend, SqlStore

        with getattr(Backend, backend)(directory / f"store.{backend}") as database:

            def forget() -> None:
                with database.engine.begin() as connection:
                    for table in tables:
                        connection.execute(sqlalchemy.text(f"UPDATE {table} SET bonds_dimensions_mask = NULL"))

            yield SqlStore(database, entry_ids=ids, entry_records=RECORDS), forget
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

    def forget() -> None:
        for table in tables:
            client[name][table].update_many({}, {"$unset": {"f.bonds_dimensions_mask": ""}})

    try:
        yield MongoStore(MongoDatabase(client, name), entry_ids=ids, entry_records=RECORDS), forget
    finally:
        client.drop_database(name)
        client.close()


def _plan(store: Any) -> Any:
    from httk.store.backend.mongo import stored_property_mongo_plan
    from httk.store.backend.sql import SqlStore, stored_property_sql_plan

    return (stored_property_sql_plan if isinstance(store, SqlStore) else stored_property_mongo_plan)(
        store, StructureEntry
    )


BACKENDS = ("sqlite", "duckdb", "mongo")


@pytest.fixture(params=BACKENDS, scope="module")
def stored(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[Any, dict[str, str]]]:
    """A store holding every structure of ``STORED`` and the map from entry id to name."""
    with _store(request.param, tmp_path_factory.mktemp("dimensions")) as (store, _forget):
        yield store, _saved(store)


@pytest.mark.parametrize(
    ("filter_string", "expected"),
    (
        ("_httk_bonds.dimensions HAS 2", {"layer"}),
        ("_httk_bonds.dimensions HAS 3", {"rocksalt domain", "mixed"}),
        ("_httk_bonds.dimensions HAS 0 AND NOT _httk_bonds.dimensions HAS 3", {"molecule"}),
        ("NOT _httk_bonds.dimensions HAS 3", {"layer", "chain", "molecule", "empty"}),
        ("_httk_bonds.dimensions HAS ONLY 0", {"molecule", "empty"}),
        ("_httk_bonds.dimensions HAS ONLY 0, 5", {"molecule", "empty"}),
        ("_httk_bonds.dimensions HAS 5", set()),
        ("NOT _httk_bonds.dimensions HAS 5", set(STORED) - {"plain"}),
        ("_httk_bonds.dimensions HAS ALL 0, 3", {"mixed"}),
        ("_httk_bonds.dimensions HAS ANY 1, 2", {"layer", "chain"}),
        ("_httk_bonds.dimensions LENGTH 0", {"empty"}),
        ("_httk_bonds.dimensions LENGTH >= 2", {"mixed"}),
        ("_httk_bonds.dimensions LENGTH != 1", {"empty", "mixed"}),
        ("_httk_bonds.dimensions IS UNKNOWN", {"plain"}),
        ("_httk_bonds.dimensions IS KNOWN", set(STORED) - {"plain"}),
    ),
)
def test_stored_dimensions_filter(stored: tuple[Any, dict[str, str]], filter_string: str, expected: set[str]) -> None:
    store, names = stored
    plan = _plan(store)
    found = {names[row[0].id] for searcher in plan.filter_searchers(filter_string) for row in searcher.results()}
    assert found == expected


def test_stored_rows_serve_the_in_memory_dimensions(stored: tuple[Any, dict[str, str]]) -> None:
    store, names = stored
    plan = _plan(store)
    served = {names[row["id"]]: row["_httk_bonds"] for row in plan.records()}
    assert served == {name: bond_properties(make())["_httk_bonds"] for name, make in STORED.items()}


@pytest.mark.parametrize("backend", BACKENDS)
def test_rows_without_stored_dimensions_serve_null_and_are_unknown(backend: str, tmp_path: Path) -> None:
    """As in a store upgraded to the dimensions column: the served value is the one the filters read."""
    with _store(backend, tmp_path) as (store, forget):
        names = _saved(store)
        forget()
        plan = _plan(store)
        rows = {names[row["id"]]: row["_httk_bonds"] for row in plan.records()}
        for name in ("mixed", "rocksalt domain"):  # a unit-cell and a domain row
            assert rows[name]["pairs"] and rows[name]["dimensions"] is None

        def found(filter_string: str) -> set[str]:
            return {names[row[0].id] for searcher in plan.filter_searchers(filter_string) for row in searcher.results()}

        assert found("_httk_bonds.dimensions IS UNKNOWN") == set(STORED)
        assert found("_httk_bonds.dimensions HAS 3") == found("NOT _httk_bonds.dimensions HAS 3") == set()


def test_a_numeric_structure_answers_unknown() -> None:
    pytest.importorskip("sqlalchemy")
    from httk.store import Backend, EntryIdScheme, SqlStore
    from httk.store.backend.sql import stored_property_sql_plan

    records = {StructureEntry: (UnitcellStructureRecord, NumericUnitcellStructureRecord)}
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_ids=EntryIdScheme("httk.test", "1"), entry_records=records)
        store.save(MIXED)
        store.save(NumericUnitcellStructure(MIXED))
        plan = stored_property_sql_plan(store, StructureEntry)
        for filter_string, counts in (
            ("_httk_bonds.dimensions HAS 3", [1, 0]),
            ("NOT _httk_bonds.dimensions HAS 3", [0, 0]),
            ("_httk_bonds.dimensions IS UNKNOWN", [0, 1]),
        ):
            assert [searcher.count() for searcher in plan.filter_searchers(filter_string)] == counts
