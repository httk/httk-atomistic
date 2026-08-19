"""Cross-package SQL execution for atomistic stored-property declarations."""

import datetime
from fractions import Fraction
from typing import Any

import pytest

pytest.importorskip("httk.store.backend.sql")

from httk.core import FracVector
from httk.store import Backend, SqlStore
from httk.store.backend.sql import stored_property_sql_plan
from httk.store.query.optimade_filters import FilterTranslationError

from httk.atomistic import (
    WyckoffSite,
    ASUStructure,
    ASUStructureRecord,
    Cell,
    FundamentalDomainStructure,
    FundamentalDomainStructureRecord,
    Sites,
    Spacegroup,
    Species,
    UnitcellStructure,
    StructureEntry,
    StructureEntryProvider,
    UnitcellStructureRecord,
)
from httk.atomistic.models.structure.semantics import StructureSymmetry


def _species() -> tuple[Species, Species]:
    return Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))


def _unitcell(
    *,
    last_modified: datetime.datetime | None = None,
    coordinate_precision: Fraction | None = None,
    basis_precision: Fraction | None = None,
    **metadata: Any,
) -> UnitcellStructure:
    return UnitcellStructure(
        Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]], precision=basis_precision),
        Sites([[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]], precision=coordinate_precision),
        _species(),
        ("Na", "Cl"),
        last_modified=last_modified,
        **metadata,
    )


def _domain(record_type: type[Any], *, molecular: bool = False) -> FundamentalDomainStructure:
    return record_type(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        225,
        (WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("b", FracVector(()), "Cl")),
        _species(),
        molecular=molecular,
    )


def _collapsed_orbit_asu() -> ASUStructure:
    rhombohedral = Spacegroup.from_setting("166:R")
    return ASUStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 12]],
        166,
        (WyckoffSite("a", FracVector(()), "Bi"),),
        (Species("Bi", ("Bi",), (1,)),),
        transform=rhombohedral.transform_from_standard,
    )


def _zero_site_unitcell() -> UnitcellStructure:
    return UnitcellStructure([[4, 0, 0], [0, 4, 0], [0, 0, 4]], [], (), ())


def _incomplete_unitcell() -> UnitcellStructure:
    return UnitcellStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        [[0, 0, 0]],
        (Species("C_X", ("C",), (1,), attached=("X",), nattached=(1,)),),
        ("C_X",),
    )


def _unitcell_with_unused_disordered_species() -> UnitcellStructure:
    return UnitcellStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]],
        (*_species(), Species("unused-half", ("Na",), (Fraction(1, 2),))),
        ("Na", "Cl"),
    )


def _database_for(request):
    if request.param == "duckdb":
        pytest.importorskip("duckdb_engine")
        return Backend.duckdb()
    return Backend.sqlite()


@pytest.fixture(params=("sqlite", "duckdb"))
def structure_plan(request):
    database = _database_for(request)
    timestamp = datetime.datetime(2026, 8, 2, 10, 30, tzinfo=datetime.UTC)
    sources = (
        _unitcell(last_modified=timestamp),
        _incomplete_unitcell(),
        _unitcell_with_unused_disordered_species(),
        _domain(FundamentalDomainStructure),
        _domain(ASUStructure),
    )
    with database:
        store = SqlStore(
            database,
            entry_records={
                StructureEntry: (
                    UnitcellStructureRecord,
                    FundamentalDomainStructureRecord,
                    ASUStructureRecord,
                )
            },
        )
        for source in sources:
            # The store resolves each natural representation through its exact
            # atomistic record binding; no hand-built storage record is used.
            store.save(source)
        yield stored_property_sql_plan(store, StructureEntry), sources


@pytest.fixture(params=("sqlite", "duckdb"))
def response_plan(request):
    database = _database_for(request)
    source = _unitcell(last_modified=datetime.datetime(2026, 8, 2, 10, 30, tzinfo=datetime.UTC))
    with database:
        store = SqlStore(database, entry_records={StructureEntry: UnitcellStructureRecord})
        store.save(source)
        yield stored_property_sql_plan(store, StructureEntry), source


@pytest.fixture(params=("sqlite", "duckdb"))
def span_plan(request):
    database = _database_for(request)
    sources = (
        _unitcell(),
        _unitcell(molecular=True),
        _domain(FundamentalDomainStructure),
        _domain(FundamentalDomainStructure, molecular=True),
        _domain(ASUStructure),
        _domain(ASUStructure, molecular=True),
    )
    with database:
        store = SqlStore(
            database,
            entry_records={
                StructureEntry: (
                    UnitcellStructureRecord,
                    FundamentalDomainStructureRecord,
                    ASUStructureRecord,
                )
            },
        )
        for source in sources:
            store.save(source)
        yield stored_property_sql_plan(store, StructureEntry)


@pytest.fixture(params=("sqlite", "duckdb"))
def scoped_scalar_plan(request):
    database = _database_for(request)
    sources = (
        _unitcell(coordinate_precision=Fraction(1, 500), basis_precision=Fraction(1, 200), charge=Fraction(3, 2)),
        _unitcell(
            symmetry=StructureSymmetry(225),
            coordinate_precision=Fraction(1, 1000),
            basis_precision=Fraction(1, 100),
        ),
    )
    with database:
        store = SqlStore(database, entry_records={StructureEntry: UnitcellStructureRecord})
        for source in sources:
            store.save(source)
        yield stored_property_sql_plan(store, StructureEntry)


@pytest.fixture(params=("sqlite", "duckdb"))
def long_precision_plan(request):
    database = _database_for(request)
    literal = "0.1234567890123456789"
    source = _unitcell(coordinate_precision=Fraction(literal))
    with database:
        store = SqlStore(database, entry_records={StructureEntry: UnitcellStructureRecord})
        store.save(source)
        yield stored_property_sql_plan(store, StructureEntry), literal


@pytest.fixture(params=("sqlite", "duckdb"))
def zero_site_plan(request):
    database = _database_for(request)
    source = _zero_site_unitcell()
    with database:
        store = SqlStore(database, entry_records={StructureEntry: UnitcellStructureRecord})
        store.save(source)
        yield stored_property_sql_plan(store, StructureEntry)


def _counts(plan, filter_string: str) -> list[int]:
    return [searcher.count() for searcher in plan.filter_searchers(filter_string)]


def test_plan_projects_the_same_rows_as_the_natural_structure_provider(response_plan):
    plan, source = response_plan
    actual = {row["id"]: row for row in plan.records()}
    assert set(actual) == {source.id}
    expected = dict(next(iter(StructureEntryProvider({source.id: source}).records("structures"))))
    expected["id"] = expected.pop("__id")
    expected["_httk_charge"] = None
    assert actual[source.id] == expected


def test_complete_and_incomplete_composition_filters_preserve_sql_unknown(structure_plan):
    plan, _sources = structure_plan

    # The first backing holds three unit cells (two complete and one
    # incomplete); each other backing holds one complete NaCl representation.
    assert _counts(plan, 'elements HAS "Na"') == [2, 1, 1]
    assert _counts(plan, 'chemical_formula_reduced = "ClNa"') == [2, 1, 1]
    assert _counts(plan, 'elements_ratios HAS 0.5') == [2, 1, 1]
    assert _counts(plan, "chemical_formula_reduced IS UNKNOWN") == [1, 0, 0]
    # Missing composition facts stay UNKNOWN under both inequality and NOT;
    # they never turn into accidental matches for the incomplete row.
    assert _counts(plan, 'chemical_formula_reduced != "ClNa"') == [0, 0, 0]
    assert _counts(plan, 'NOT chemical_formula_reduced = "ClNa"') == [0, 0, 0]

    ratio_searcher = plan.filter_searchers("elements_ratios HAS 0.5")[0]
    statement = ratio_searcher._base_select(
        [ratio_searcher._outputs[0].element],
        [ratio_searcher._variables[0]._alias.c["sid"]],
    )
    rendered = str(statement.compile(dialect=plan.store._database.engine.dialect))
    assert "ratio_exact" in rendered


def test_timestamp_filtering_and_sorting_use_utc_instants(structure_plan):
    plan, _sources = structure_plan
    offset_equivalent = 'last_modified = "2026-08-02T12:30:00+02:00"'
    assert _counts(plan, offset_equivalent) == [1, 0, 0]
    assert _counts(plan, 'last_modified > "2026-08-02T10:29:59Z"') == [1, 0, 0]
    searchers = plan.filter_searchers(offset_equivalent, sort=(("last_modified", False),))
    assert [searcher.count() for searcher in searchers] == [1, 0, 0]


def test_invalid_formula_and_nonintegral_count_literals_are_filter_value_errors(structure_plan):
    plan, _sources = structure_plan
    for filter_string in ('chemical_formula_reduced = "NaCl"', "nelements = 1.5"):
        with pytest.raises(FilterTranslationError) as caught:
            plan.filter_searchers(filter_string)
        # ``type-mismatch`` is the neutral filter translator's established
        # caller/filter-value category (rather than a storage error).
        assert caught.value.category == "type-mismatch"


def test_unused_disordered_species_does_not_create_a_structure_feature(structure_plan):
    plan, _sources = structure_plan
    assert _counts(plan, 'structure_features HAS "disorder"') == [0, 0, 0]


def test_structure_feature_presence_filters_execute_sql(structure_plan):
    plan, _sources = structure_plan
    assert _counts(plan, 'structure_features HAS "assemblies"') == [0, 0, 0]
    assert _counts(plan, 'structure_features HAS "site_attachments"') == [1, 0, 0]


def test_structure_features_length_beyond_the_feature_vocabulary_is_false(structure_plan):
    plan, _sources = structure_plan
    assert _counts(plan, "structure_features LENGTH 5") == [0, 0, 0]


@pytest.mark.parametrize(
    ("span", "equal_counts", "unequal_counts"),
    (
        ("unit_cell", [1, 0, 0], [1, 2, 2]),
        ("molecular_unit_cell", [1, 0, 0], [1, 2, 2]),
        ("fundamental_domain", [0, 1, 0], [2, 1, 2]),
        ("molecular_fundamental_domain", [0, 1, 0], [2, 1, 2]),
        ("asymmetric_unit", [0, 0, 1], [2, 2, 1]),
        ("molecular_asymmetric_unit", [0, 0, 1], [2, 2, 1]),
        ("molecular_entities", [0, 0, 0], [2, 2, 2]),
        ("other", [0, 0, 0], [2, 2, 2]),
    ),
)
def test_coordinate_span_filters_are_safe_across_heterogeneous_backings(span_plan, span, equal_counts, unequal_counts):
    assert _counts(span_plan, f'site_coordinate_span = "{span}"') == equal_counts
    assert _counts(span_plan, f'site_coordinate_span != "{span}"') == unequal_counts


def test_invalid_coordinate_span_literal_is_a_filter_value_error(span_plan):
    with pytest.raises(FilterTranslationError) as caught:
        span_plan.filter_searchers('site_coordinate_span = "not_a_coordinate_span"')
    assert caught.value.category == "type-mismatch"


def test_optional_reference_scalars_are_correlated_and_preserve_unknown(scoped_scalar_plan):
    plan = scoped_scalar_plan
    assert _counts(plan, "space_group_it_number = 225") == [1]
    assert _counts(plan, "space_group_it_number IS UNKNOWN") == [1]
    assert _counts(plan, "space_group_it_number IS KNOWN") == [1]
    assert _counts(plan, "NOT space_group_it_number = 225") == [0]


def test_charge_filter_is_exact_and_preserves_unknown(scoped_scalar_plan):
    assert _counts(scoped_scalar_plan, "_httk_charge = 1.5") == [1]
    assert _counts(scoped_scalar_plan, "_httk_charge IS UNKNOWN") == [1]


@pytest.mark.parametrize(
    ("property_name", "first", "second"),
    (
        ("_httk_coordinate_precision", "0.002", "0.001"),
        ("_httk_basis_precision", "0.005", "0.01"),
    ),
)
def test_nested_reference_scalar_precision_does_not_cross_parent_rows(scoped_scalar_plan, property_name, first, second):
    assert _counts(scoped_scalar_plan, f"{property_name} = {first}") == [1]
    assert _counts(scoped_scalar_plan, f"{property_name} = {second}") == [1]


def test_exact_precision_filter_retains_the_full_decimal_literal(long_precision_plan):
    plan, literal = long_precision_plan
    assert _counts(plan, f"_httk_coordinate_precision = {literal}") == [1]


@pytest.mark.parametrize("dialect", ("sqlite", "duckdb"))
def test_natural_collapsed_asu_orbit_preserves_expanded_composition(dialect):
    if dialect == "duckdb":
        pytest.importorskip("duckdb_engine")
        database = Backend.duckdb()
    else:
        database = Backend.sqlite()
    source = _collapsed_orbit_asu()
    with database:
        store = SqlStore(database, entry_records={StructureEntry: ASUStructureRecord})
        store.save(source)
        searcher = store.searcher()
        variable = searcher.variable(ASUStructureRecord)
        searcher.output(variable, "record")
        (fetched,), _names = next(iter(searcher))
        assert tuple((value.element, value.amount) for value in fetched.normalized_composition.amounts) == (
            ("Bi", Fraction(1)),
        )
        served = next(iter(StructureEntryProvider({fetched.id: fetched}).records("structures")))
        assert served["elements"] == ["Bi"]
        assert served["elements_ratios"] == [1.0]
        assert served["chemical_formula_reduced"] == "Bi"


def test_zero_site_composition_keeps_elements_known_but_formulas_unknown(zero_site_plan):
    row = next(zero_site_plan.records())
    assert row["elements"] == []
    assert row["nelements"] == 0
    assert row["elements_ratios"] == []
    assert row["chemical_formula_reduced"] is None

    assert _counts(zero_site_plan, "chemical_formula_reduced IS UNKNOWN") == [1]
    assert _counts(zero_site_plan, "chemical_formula_reduced IS KNOWN") == [0]
    assert _counts(zero_site_plan, 'chemical_formula_reduced != "ClNa"') == [0]
    assert _counts(zero_site_plan, 'NOT chemical_formula_reduced = "ClNa"') == [0]
