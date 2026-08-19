"""Tests for the geometry-free protostructure and dummy-species prototype records."""

from fractions import Fraction

import pytest
from httk.core.storage import content_id, storage_identity_name

from httk.atomistic import (
    Cell,
    Protostructure,
    ProtostructureRecord,
    Prototype,
    PrototypeRecord,
    Spacegroup,
    Species,
    WyckoffOccupationRecord,
    WyckoffSite,
)
from httk.atomistic.entries.prototypes import ProtostructureEntry, PrototypeEntry
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.storage.records import (
    _protostructure_from_record,
    _protostructure_record_from_value,
    _prototype_from_record,
    _prototype_record_from_value,
)

EMPTY: tuple[Fraction, ...] = ()


def _rocksalt() -> Protostructure:
    return Protostructure(225, [("a", "Na"), ("b", "Cl")])


def _hexagonal_prototype() -> Prototype:
    dummy = Species("A", ("X",), (1,), labels=("A",))
    return Prototype(
        Cell(CellParams((2, 2, 3, 90, 90, 120)).basis),
        191,
        (WyckoffSite("a", EMPTY, "A"),),
        (dummy,),
    )


def test_record_identity_names_are_storage_names() -> None:
    for record_type in (ProtostructureRecord, PrototypeRecord, WyckoffOccupationRecord):
        identity_name = storage_identity_name(record_type)
        assert identity_name == record_type.__httk_storage__.storage_name
        assert "httk.atomistic" not in identity_name


def test_protostructure_indexes_cover_it_number_and_label() -> None:
    assert ProtostructureRecord.__httk_storage__.indexes == (("spacegroup_it_number",), ("label",))


def test_protostructure_record_label_format_is_deterministic() -> None:
    record = _protostructure_record_from_value(_rocksalt())
    # The httk protostructure label: classes ordered by Wyckoff letters, then species names.
    assert record.label == "AB_cF8_225_a_b:Na-Cl"


def test_protostructure_golden_content_id_is_layout_independent() -> None:
    record = _protostructure_record_from_value(_rocksalt())
    # A future storage-layout move must not change this pinned identity.
    assert record.id == content_id(_rocksalt()) == "743eee98965ea3a28e00db23b8211d1d41659dbc1a8db2a0b34ace3f298896b5"


def test_prototype_golden_content_id_is_layout_independent() -> None:
    record = _prototype_record_from_value(_hexagonal_prototype())
    assert record.id == content_id(_hexagonal_prototype())
    assert record.id == "1862b718d99f026f99b2034dc9087555b1939a9db448fa630ebdf7d98c2dbace"


def test_equal_protostructures_share_content_id_including_permuted_species_order() -> None:
    first = Protostructure(225, [("a", "Na"), ("b", "Cl")])
    permuted = Protostructure(Spacegroup.standard(225), [("b", "Cl"), ("a", "Na")])
    assert first == permuted
    first_record = _protostructure_record_from_value(first)
    permuted_record = _protostructure_record_from_value(permuted)
    assert first_record == permuted_record
    assert first_record.id == permuted_record.id


def test_unequal_protostructures_differ_in_content_id() -> None:
    first = _protostructure_record_from_value(Protostructure(225, [("a", "Na"), ("b", "Cl")]))
    other = _protostructure_record_from_value(Protostructure(225, [("a", "Na")]))
    assert first.id != other.id


def test_disordered_species_protostructure_round_trips() -> None:
    mixed = Species("mixed", ("Fe", "Ni"), (Fraction(1, 2), Fraction(1, 2)))
    value = Protostructure(221, [("a", mixed), ("b", "Cl")])
    record = _protostructure_record_from_value(value)
    assert _protostructure_from_record(record) == value


def test_prototype_record_round_trips_surd_cell_and_free_parameters() -> None:
    value = _hexagonal_prototype()
    record = _prototype_record_from_value(value)
    rebuilt = _prototype_from_record(record)
    assert rebuilt == value
    assert record.cell.basis  # surd basis retained as exact scalars


def test_wyckoff_occupation_record_rejects_raw_species() -> None:
    with pytest.raises(TypeError, match="SpeciesRecord"):
        WyckoffOccupationRecord(wyckoff="a", species=Species("Na", ("Na",), (1,)))  # type: ignore[arg-type]


def test_protostructure_record_rejects_out_of_range_it_number() -> None:
    good = _protostructure_record_from_value(_rocksalt())
    with pytest.raises(ValueError, match="in .1, 230."):
        ProtostructureRecord(
            spacegroup_it_number=231,
            spacegroup_hall_entry=good.spacegroup_hall_entry,
            occupations=good.occupations,
        )


def test_sql_store_protostructure_dedup_and_it_number_filter() -> None:
    pytest.importorskip("sqlalchemy")
    from httk.store import Backend, SqlStore

    first = _protostructure_record_from_value(Protostructure(225, [("a", "Na"), ("b", "Cl")]))
    equal = _protostructure_record_from_value(Protostructure(Spacegroup.standard(225), [("b", "Cl"), ("a", "Na")]))
    low = _protostructure_record_from_value(Protostructure(1, [("a", "He")]))
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={ProtostructureEntry: ProtostructureRecord})
        first_sid = store.save(first)
        equal_sid = store.save(equal)
        low_sid = store.save(low)
        assert first_sid == equal_sid  # content-id dedup: two equal protostructures -> one row
        assert low_sid != first_sid

        searcher = store.searcher()
        variable = searcher.variable(ProtostructureRecord)
        searcher.add(variable.spacegroup_it_number > 2)
        assert searcher.count() == 1

        # The derived label column is queryable.
        label_searcher = store.searcher()
        label_variable = label_searcher.variable(ProtostructureRecord)
        label_searcher.add(label_variable.label == "AB_cF8_225_a_b:Na-Cl")
        assert label_searcher.count() == 1

        fetched = store.fetch(ProtostructureRecord, first_sid, eager=True)
        assert fetched.id == first.id
        assert fetched.label == "AB_cF8_225_a_b:Na-Cl"
        assert _protostructure_from_record(fetched) == Protostructure(225, [("a", "Na"), ("b", "Cl")])


def test_sql_store_prototype_round_trips_surd_hexagonal_cell() -> None:
    pytest.importorskip("sqlalchemy")
    from httk.store import Backend, SqlStore

    value = _hexagonal_prototype()
    record = _prototype_record_from_value(value)
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={PrototypeEntry: PrototypeRecord})
        sid = store.save(record)
        # Asserted after the in-memory database is disposed, so materialize now.
        fetched = store.fetch(PrototypeRecord, sid, eager=True)

    assert fetched.id == record.id
    assert _prototype_from_record(fetched) == value
