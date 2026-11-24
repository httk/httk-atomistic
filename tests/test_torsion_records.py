"""Storage of torsion statements: the ``BondTorsionsRecord``, content identity, stores and upgrades.

A network's torsions are one ``BondTorsionsRecord`` row referenced by the unit-cell network record
or the fundamental-domain record (``torsions``, ``IdentitySkipWhenNone``); ``torsions_source`` is an
``IdentitySkip`` source on the structure record, like the other sources.
"""

import json
from dataclasses import replace
from fractions import Fraction as F
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from httk.core import FracVector
from httk.core.storage import content_id
from test_bond_network_records import (
    RECORDS,
    _a2db444,
    _cell,
    _record,
    _record_type,
    store,  # noqa: F401  (the fixture)
)
from test_bond_network_torsions import FIXTURE_DOMAINS, T

from httk.atomistic import (
    AffineOperation,
    BondSource,
    FundamentalDomainStructure,
    SiteTorsion,
    UnitcellStructureView,
)
from httk.atomistic.models.structure.record import RecordStructure
from httk.atomistic.storage.records import (
    BondTorsionsRecord,
    UnitcellStructureRecord,
    _structure_from_record,
    _unitcell_record_from_structure,
    validate_structure_record,
)

ONE = AffineOperation.identity()
TORSIONS_SOURCE = BondSource("hand-stated", "test", "2")


def _stated(structure: Any, rows: Any, source: BondSource | None = TORSIONS_SOURCE) -> Any:
    assert structure.bonds is not None
    return structure._with_bonds(replace(structure.bonds, torsions=rows, torsions_source=source))


def _cell_rows() -> list[SiteTorsion]:
    """Statements of every kind on the rock-salt unit cell (they need not be its geometry)."""
    return [
        SiteTorsion((0, 1, 2, 3), (ONE, ONE, ONE, T(1, 0, 0)), "g"),  # a symbol only
        SiteTorsion((1, 2, 3, 4), (ONE,) * 4, angle=-60.25),  # an angle only
        SiteTorsion((2, 3, 4, 5), (ONE,) * 4, "-sc", -62.5, 0.25),  # everything
        SiteTorsion((3, 4, 5, 6), (ONE,) * 4, "none"),
        SiteTorsion((4, 5, 6, 7), (ONE, ONE, T(0, 0, 1), ONE), angle=120.0, uncertainty=1.5),
    ]


@cache
def _sarcosine() -> Any:
    return FIXTURE_DOMAINS["sarcosine (Sohncke)"]()


def _domain_rows() -> list[SiteTorsion]:
    """The sarcosine statements varied: symbols only, angles only, with uncertainties, and ``none``."""
    rows = _sarcosine().bonds.torsions
    assert rows is not None and len(rows) >= 4
    signed = ("+sc" if rows[2].angle > 0 else "-sc") if rows[2].angle else "sc"
    return [
        replace(rows[0], angle=None, symbol="t"),
        rows[1],
        replace(rows[2], symbol=signed, uncertainty=0.5),
        replace(rows[3], angle=None, symbol="none"),
        *rows[4:],
    ]


@cache
def _variants() -> dict[str, Any]:
    """Structures of every record kind with and without torsions and their source."""
    cell = _cell()
    sarcosine = _sarcosine()
    domain = FundamentalDomainStructure(
        sarcosine.cell,
        sarcosine.spacegroup,
        sarcosine.wyckoff_sites,
        sarcosine.species,
        sarcosine.transform,
        bonds=sarcosine.bonds,
    )
    return {
        "unit cell without torsions": cell,
        "unit cell with torsions": _stated(cell, _cell_rows()),
        "unit cell with torsions from its source": _stated(cell, _cell_rows()[1:], None),
        "unit cell of sarcosine": UnitcellStructureView(sarcosine).unview(),
        "asymmetric unit without torsions": _stated(sarcosine, None, None),
        "asymmetric unit with torsions": _stated(sarcosine, _domain_rows()),
        "fundamental domain with torsions": _stated(domain, _domain_rows()),
    }


def _torsions(value: Any) -> tuple[object, object]:
    network = value.bonds
    return (None, None) if network is None else (network.torsions, network.torsions_source)


# ---------------------------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------------------------
NAMES = [
    "unit cell without torsions",
    "unit cell with torsions",
    "unit cell with torsions from its source",
    "unit cell of sarcosine",
    "asymmetric unit without torsions",
    "asymmetric unit with torsions",
    "fundamental domain with torsions",
]


@pytest.mark.parametrize("name", NAMES)
def test_records_round_trip_torsions_and_their_source(name: str) -> None:
    assert list(_variants()) == NAMES
    value = _variants()[name]
    record = _record(value)
    type(record).__httk_validate__(record)
    torsions = record.bonds.torsions if isinstance(record, UnitcellStructureRecord) else record.torsions
    assert (torsions is None) is (value.bonds.torsions is None)
    back = _structure_from_record(record)
    assert back == value and _torsions(back) == _torsions(value)
    assert content_id(record) == content_id(value)
    assert RecordStructure(record).bonds == UnitcellStructureView(value).bonds


def test_a_torsions_record_holds_aligned_columns_in_the_general_form() -> None:
    record = _unitcell_record_from_structure(_stated(_cell(), _cell_rows())).bonds.torsions
    assert isinstance(record, BondTorsionsRecord)
    rows = record.atoms.to_fractions()
    assert len(rows) == 5 and all(len(row) == 52 for row in rows)
    identity = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    assert all(row[13 * k + 1 : 13 * k + 10] == identity for row in rows for k in range(4))  # unit-cell rotations
    assert sorted(record.symbols, key=str) == sorted(["g", None, "-sc", "none", None], key=str)
    assert sorted(x for x in record.angles if x is not None) == [-62.5, -60.25, 120.0]
    assert sorted(x for x in record.uncertainties if x is not None) == [0.25, 1.5]


def test_content_identity_follows_torsions_and_ignores_their_source() -> None:
    for plain, stated, rows in (
        (_cell(), _stated(_cell(), _cell_rows()), _cell_rows()),
        (_stated(_sarcosine(), None, None), _sarcosine(), _domain_rows()),
    ):
        assert content_id(plain) != content_id(stated)
        assert content_id(_stated(stated, stated.bonds.torsions, BondSource(software="other"))) == content_id(stated)
        assert content_id(_stated(plain, rows)) != content_id(_stated(plain, rows[1:]))
        changed = [replace(rows[1], angle=rows[1].angle + 0.01), *rows[2:], rows[0]]
        assert content_id(_stated(plain, rows)) != content_id(_stated(plain, changed))


def test_equal_networks_with_equal_torsions_share_the_network_row() -> None:
    from test_bond_network_records import _rocksalt

    first = _stated(UnitcellStructureView(_rocksalt()).unview(), _cell_rows())
    second = _stated(UnitcellStructureView(_rocksalt(F(560, 100))).unview(), _cell_rows(), BondSource(software="x"))
    one, two = (_unitcell_record_from_structure(value).bonds for value in (first, second))
    assert content_id(first) != content_id(second)
    assert content_id(one) == content_id(two) != content_id(_unitcell_record_from_structure(_cell()).bonds)


def test_bad_torsion_records_are_rejected() -> None:
    cell = _stated(_cell(), _cell_rows())
    record = _unitcell_record_from_structure(cell)
    torsions = record.bonds.torsions
    rows = torsions.atoms.to_fractions()
    translated = [[x + 1 if c % 13 == 10 else x for c, x in enumerate(rows[0])], *rows[1:]]  # every atom moved by a
    with pytest.raises(ValueError, match="torsions are not in their normal form"):
        validate_structure_record(
            replace(record, bonds=replace(record.bonds, torsions=replace(torsions, atoms=FracVector(translated))))
        )
    swapped = FracVector(torsions.atoms.to_fractions()[::-1])
    with pytest.raises(ValueError, match="torsions are not in their normal form"):
        validate_structure_record(
            replace(record, bonds=replace(record.bonds, torsions=replace(torsions, atoms=swapped)))
        )
    with pytest.raises(ValueError, match="torsions_source is not in its normal form"):
        validate_structure_record(replace(record, torsions_source=record.bonds_source))
    with pytest.raises(ValueError, match="torsions_source needs torsions"):
        replace(record, bonds=replace(record.bonds, torsions=None))
    with pytest.raises(TypeError, match="BondTorsionsRecord"):
        replace(record, bonds=replace(record.bonds, torsions="rows"))
    with pytest.raises(ValueError, match="52 columns"):
        BondTorsionsRecord(FracVector(()), (), (), ())
    with pytest.raises(ValueError, match="one entry per row"):
        replace(torsions, symbols=torsions.symbols[1:])
    with pytest.raises(ValueError, match="non-negative integers"):
        replace(torsions, atoms=FracVector([[F(1, 2)] + row[1:] for row in torsions.atoms.to_fractions()]))
    with pytest.raises(ValueError, match="finite"):
        replace(torsions, angles=(float("nan"),) * 5)
    domain = _record(_sarcosine())
    validate_structure_record(domain)
    with pytest.raises(ValueError, match="torsions_source needs torsions"):
        replace(domain, torsions=None)
    with pytest.raises(ValueError, match="torsions need bonds"):
        replace(domain, bonds=None, shapes=None, bonds_source=None, shapes_source=None, bonds_dimensions_mask=None)


# ---------------------------------------------------------------------------------------------
# stores
# ---------------------------------------------------------------------------------------------
def test_a_store_round_trips_torsions_of_every_record_kind(store: Any) -> None:  # noqa: F811
    for value in _variants().values():
        sid = store.save(value)
        for fetched in (store.fetch(_record_type(value), sid, eager=True), store.fetch(_record_type(value), sid)):
            back = _structure_from_record(fetched)
            assert back == value and _torsions(back) == _torsions(value)
            assert content_id(fetched) == content_id(value)


@pytest.mark.parametrize("backend", ["sqlite", "duckdb"])
def test_a_deferred_bulk_ingest_stores_torsions(backend: str) -> None:
    pytest.importorskip("httk.store")
    pytest.importorskip("sqlalchemy")
    if backend == "duckdb":
        pytest.importorskip("duckdb_engine")
    from httk.store import Backend, EntryIdScheme, SqlStore

    values = list(_variants().values())
    with getattr(Backend, backend)() as database:
        target = SqlStore(database, entry_ids=EntryIdScheme("httk.test", "1"), entry_records=RECORDS)
        with target.bulk_ingest(finalize="deferred") as ingest:
            provisional = [ingest.save(value) for value in values]
        for value, sid in zip(values, provisional, strict=True):
            record_type = _record_type(value)
            back = _structure_from_record(target.fetch(record_type, ingest.resolved_sid(record_type, sid), eager=True))
            assert back == value and _torsions(back) == _torsions(value)


def test_the_torsion_fields_are_an_additive_upgrade() -> None:
    """A layout without the torsion fields and table (stage 1) upgrades to this one additively."""
    pytest.importorskip("httk.store")
    from httk.store.storage_layout import (
        AdditiveUpgradePlan,
        classify_schema_upgrade,
        normalize_entry_records,
        schema_fingerprint_json,
    )

    current = schema_fingerprint_json(normalize_entry_records(RECORDS))
    before = json.loads(current)
    del before["tables"]["atomistic_bond_torsions"]
    for table in before["tables"].values():
        for name in ("torsions", "torsions_source"):
            table["fields"].pop(name, None)
    plan = classify_schema_upgrade(json.dumps(before), current)
    assert isinstance(plan, AdditiveUpgradePlan), plan
    assert {table: [column.name for column in columns] for table, columns in plan.added_columns.items()} == {
        "atomistic_asu_structure": ["torsions_sid", "torsions_source_sid"],
        "atomistic_fundamental_domain_structure": ["torsions_sid", "torsions_source_sid"],
        "atomistic_unitcell_bond_network": ["torsions_sid"],
        "atomistic_unitcell_structure": ["torsions_source_sid"],
    }


@pytest.mark.parametrize("backend", ["sqlite", "duckdb", "mongo"])
def test_an_upgraded_protocol_3_store_takes_torsions(backend: str, tmp_path: Path) -> None:
    with _a2db444(backend, tmp_path) as opened, opened(True) as upgraded:
        for value in _variants().values():
            back = _structure_from_record(upgraded.fetch(_record_type(value), upgraded.save(value), eager=True))
            assert back == value and _torsions(back) == _torsions(value)
