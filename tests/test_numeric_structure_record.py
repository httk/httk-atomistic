"""Tests for the all-float NumericUnitcellStructureRecord family and its store round trip."""

import math
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

numpy = pytest.importorskip("numpy")

from httk.core.storage import content_id

from httk.atomistic import (
    NumericUnitcellStructure,
    NumericUnitcellStructureRecord,
    NumericUnitcellStructureView,
    Species,
    StructureEntry,
    StructureEntryProvider,
    UnitcellStructure,
    UnitcellStructureRecord,
    UnitcellStructureView,
)
from httk.atomistic.storage.records import (
    NumericCellRecord,
    NumericCompositionAmountRecord,
    NumericSitesRecord,
    NumericSpeciesConstituentRecord,
    numeric_structure_from_record,
)


def _nacl() -> NumericUnitcellStructure:
    return NumericUnitcellStructure(
        [[0.0, 2.8, 2.8], [2.8, 0.0, 2.8], [2.8, 2.8, 0.0]],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        species_at_sites=[Species("Na", ("Na",), (1,), mass=(22.99,)), Species("Cl", ("Cl",), (1,))],
        charge=-1.5,
        chemical_formula_hill="ClNa",
        optimization_type="local",
    )


def _disordered() -> NumericUnitcellStructure:
    exact = UnitcellStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]],
        species=[
            Species(
                "A",
                ("Fe", "Ni"),
                (Fraction(1, 3), Fraction(2, 3)),
                charges=(Fraction(1, 3), None),
                spins=(Fraction(1, 2), None),
                labels=("x", None),
                concentration_precision=(Fraction(1, 100), None),
            ),
            Species("O", ("O",), (1,)),
        ],
        species_at_sites=["A", "O"],
        charge=Fraction(1, 3),
    )
    return NumericUnitcellStructure(exact)


@pytest.mark.parametrize("make", (_nacl, _disordered))
def test_record_round_trips_value_and_content_id(make: Any) -> None:
    value = make()
    record = NumericUnitcellStructureRecord.from_structure(value)
    assert numeric_structure_from_record(record) == value
    assert content_id(record) == content_id(value) == value.id
    NumericUnitcellStructureRecord.__httk_validate__(record)
    viewed = UnitcellStructureView(record)
    assert viewed.unwrap() is record
    assert viewed.species == value.species
    assert NumericUnitcellStructure(viewed) == value


def test_numeric_and_exact_copies_are_different_content() -> None:
    value = _nacl()
    assert value.id != UnitcellStructureView(value).unview().id


def test_post_init_normalises_numpy_and_negative_zero_and_rejects_non_finite() -> None:
    cell = NumericCellRecord(numpy.array([[4.0, -0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]), None, (1, 1, 1))
    assert cell.basis == ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))
    assert math.copysign(1.0, cell.basis[0][1]) == 1.0
    assert all(type(x) is float for row in cell.basis for x in row)
    assert cell.periodicity == (True, True, True)
    sites = NumericSitesRecord(numpy.zeros((2, 3)) * -1.0, numpy.float64(1e-3))
    assert sites.reduced_coords == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert type(sites.precision) is float
    assert str(NumericSpeciesConstituentRecord("Fe", 1.0, charge=-0.0).charge) == "0.0"
    assert NumericSitesRecord((), None).reduced_coords == ()
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError):
            NumericCellRecord([[bad, 0, 0], [0, 1, 0], [0, 0, 1]], None, None)
        with pytest.raises(ValueError):
            NumericSpeciesConstituentRecord("Fe", bad)
        with pytest.raises(ValueError):
            NumericCompositionAmountRecord("Fe", 1.0, bad)
        with pytest.raises(ValueError):
            replace(NumericUnitcellStructureRecord.from_structure(_nacl()), charge=bad)


def test_validation_rejects_contradicting_composition() -> None:
    record = NumericUnitcellStructureRecord.from_structure(_nacl())
    wrong = replace(
        record,
        normalized_composition=NumericUnitcellStructureRecord.from_structure(_disordered()).normalized_composition,
    )
    with pytest.raises(ValueError, match="contradicts"):
        NumericUnitcellStructureRecord.__httk_validate__(wrong)


def test_provider_serves_numeric_values_and_records() -> None:
    value = _disordered()
    provider = StructureEntryProvider({"v": value, "r": NumericUnitcellStructureRecord.from_structure(value)})
    served = {entry["__id"]: entry for entry in provider.records("structures")}
    for entry in served.values():
        assert entry["nsites"] == 2
        assert entry["elements"] == ["Fe", "Ni", "O"]
    assert served["v"]["lattice_vectors"] == served["r"]["lattice_vectors"]


def test_numeric_view_is_not_storable() -> None:
    with pytest.raises(TypeError, match="NumericUnitcellStructureView is not storable"):
        content_id(NumericUnitcellStructureView(_nacl()))


def _store(database: Any, records: Any) -> Any:
    from httk.store import EntryIdScheme, SqlStore

    return SqlStore(database, entry_ids=EntryIdScheme("httk.test", "1"), entry_records={StructureEntry: records})


def _exact_columns(database: Any) -> list[str]:
    import sqlalchemy

    inspector = sqlalchemy.inspect(database.engine)
    return [
        f"{table}.{column['name']}"
        for table in inspector.get_table_names()
        for column in inspector.get_columns(table)
        if column["name"].endswith("_exact")
    ]


def _backend(dialect: str, path: Path | None = None) -> Any:
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("httk.store")
    if dialect == "duckdb":
        pytest.importorskip("duckdb_engine")
    from httk.store import Backend

    return Backend.duckdb(path) if dialect == "duckdb" else Backend.sqlite(path)


@pytest.mark.parametrize("dialect", ("sqlite", "duckdb"))
def test_store_round_trip_has_no_exact_columns(dialect: str) -> None:
    with _backend(dialect) as database:
        store = _store(database, NumericUnitcellStructureRecord)
        for value in (_nacl(), _disordered()):
            sid = store.save(value)
            assert store.save(value) == sid
            fetched = store.fetch(NumericUnitcellStructureRecord, sid)
            assert numeric_structure_from_record(fetched) == value
            assert content_id(fetched) == content_id(value)
            assert NumericUnitcellStructure(UnitcellStructureView(fetched)) == value
            assert UnitcellStructureView(fetched).species == value.species
        assert _exact_columns(database) == []


def test_duckdb_file_store_keeps_content_id_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "numeric.duckdb"
    value = _disordered()
    database = _backend("duckdb", path)
    sid = _store(database, NumericUnitcellStructureRecord).save(value)
    database.dispose()
    with _backend("duckdb", path) as reopened:
        fetched = _store(reopened, NumericUnitcellStructureRecord).fetch(NumericUnitcellStructureRecord, sid)
        assert content_id(fetched) == content_id(value)
        assert numeric_structure_from_record(fetched) == value


def test_mixed_store_dispatches_to_the_right_record() -> None:
    exact = UnitcellStructure([[4, 0, 0], [0, 4, 0], [0, 0, 4]], [[0, 0, 0]], species_at_sites=["Fe"])
    numeric = NumericUnitcellStructure(exact)
    with _backend("sqlite") as database:
        store = _store(database, (UnitcellStructureRecord, NumericUnitcellStructureRecord))
        store.save(exact)
        store.save(numeric)
        assert isinstance(store.fetch_entry(StructureEntry, content_id(exact)), UnitcellStructureRecord)
        assert isinstance(store.fetch_entry(StructureEntry, numeric.id), NumericUnitcellStructureRecord)


def _rocksalt() -> UnitcellStructure:
    half = Fraction(1, 2)
    na = [[0, 0, 0], [0, half, half], [half, 0, half], [half, half, 0]]
    cl = [[half, half, half], [half, 0, 0], [0, half, 0], [0, 0, half]]
    return UnitcellStructure([[5.6, 0, 0], [0, 5.6, 0], [0, 0, 5.6]], na + cl, species_at_sites=["Na"] * 4 + ["Cl"] * 4)


@pytest.fixture(params=("sqlite", "duckdb"))
def mixed_plan(request: pytest.FixtureRequest) -> Any:
    from httk.store.backend.sql import stored_property_sql_plan

    with _backend(request.param) as database:
        store = _store(database, (UnitcellStructureRecord, NumericUnitcellStructureRecord))
        store.save(_rocksalt())
        store.save(NumericUnitcellStructure(_rocksalt()))
        store.save(_disordered())
        yield stored_property_sql_plan(store, StructureEntry)


@pytest.mark.parametrize(
    ("filter_string", "expected"),
    (
        ('elements HAS "Na"', [1, 1]),
        ('elements HAS "Fe"', [0, 1]),
        ("nelements=2", [1, 1]),
        ("nelements=3", [0, 1]),
        ('chemical_formula_reduced="ClNa"', [1, 1]),
        ('chemical_formula_anonymous="AB"', [1, 1]),
        ("nsites=8", [1, 1]),
        ("nsites=2", [0, 1]),
        ("elements_ratios HAS 0.5", [1, 2]),
        ("elements_ratios HAS 0.3333333333333333", [0, 1]),
        ('elements:elements_ratios HAS "O":0.5', [0, 1]),
        ('structure_features HAS "disorder"', [0, 1]),
        ("structure_features LENGTH 0", [1, 1]),
        ("species.concentration HAS 0.3333333333333333", [0, 1]),
        ('species.chemical_symbols:species.concentration HAS "Ni":0.6666666666666666', [0, 1]),
        ("_httk_charge=0.3333333333333333", [0, 1]),
        ("_httk_charge IS UNKNOWN", [1, 1]),
        ("space_group_it_number IS UNKNOWN", [1, 2]),
        ("space_group_it_number=225", [0, 0]),
        ("NOT space_group_it_number=225", [0, 0]),
        ("nperiodic_dimensions=3", [1, 2]),
        ('site_coordinate_span="unit_cell"', [1, 2]),
    ),
)
def test_mixed_store_filters_select_both_kinds(mixed_plan: Any, filter_string: str, expected: list[int]) -> None:
    assert [searcher.count() for searcher in mixed_plan.filter_searchers(filter_string)] == expected


def test_mixed_store_serves_numeric_rows(mixed_plan: Any) -> None:
    rows = sorted(mixed_plan.records(), key=lambda row: (row["nsites"], row["structure_features"]))
    assert [(row["nsites"], row["_httk_natoms"], row["_httk_nformula_units"]) for row in rows] == [
        (2, None, None),
        (8, 8, 4),
        (8, 8, 4),
    ]
    disordered, *nacl = rows
    assert disordered["chemical_formula_reduced"] != "FeNi2O3"  # the documented float-amount ceiling
    assert disordered["elements_ratios"] == [1 / 6, 1 / 3, 0.5]
    assert disordered["space_group_it_number"] is None and disordered["assemblies"] is None
    assert nacl[0]["cartesian_site_positions"] == nacl[1]["cartesian_site_positions"]
    assert (
        nacl[0]["lattice_vectors"] == nacl[1]["lattice_vectors"] == [[5.6, 0.0, 0.0], [0.0, 5.6, 0.0], [0.0, 0.0, 5.6]]
    )
