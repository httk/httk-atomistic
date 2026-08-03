"""Atomistic durable-property declarations without a storage implementation."""

import datetime
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import pytest
from httk.core.storage import (
    QueryContext,
    QueryExpression,
    QueryLiteralError,
    QueryScope,
    QueryValue,
    stored_property_projections,
)
from test_structure_record import _domain, _domain_record, _unitcell, _unitcell_record

from httk.atomistic import (
    ASUStructure,
    ASUStructureRecord,
    FundamentalDomainStructure,
    FundamentalDomainStructureRecord,
    Species,
    StructureEntryProvider,
    UnitcellStructure,
    UnitcellStructureRecord,
)
from httk.atomistic.entries.precision import PRECISION_PROPERTY_KEYS
from httk.atomistic.entries.symmetry import SETTING_PROPERTY_KEYS

_STANDARD = {
    "immutable_id",
    "last_modified",
    "elements",
    "nelements",
    "elements_ratios",
    "chemical_formula_descriptive",
    "chemical_formula_reduced",
    "chemical_formula_hill",
    "chemical_formula_anonymous",
    "dimension_types",
    "nperiodic_dimensions",
    "lattice_vectors",
    "space_group_symmetry_operations_xyz",
    "space_group_symbol_hall",
    "space_group_symbol_hermann_mauguin",
    "space_group_symbol_hermann_mauguin_extended",
    "space_group_it_number",
    "cartesian_site_positions",
    "fractional_site_positions",
    "site_coordinate_span",
    "site_coordinate_span_description",
    "nsites",
    "species_at_sites",
    "species",
    "assemblies",
    "wyckoff_positions",
    "structure_features",
    "optimization_type",
}
_EXPECTED = _STANDARD | set(SETTING_PROPERTY_KEYS) | set(PRECISION_PROPERTY_KEYS)


@dataclass(frozen=True)
class _Value:
    tag: str
    value: object


@dataclass(frozen=True)
class _Expression:
    tag: str
    values: tuple[object, ...]

    def __and__(self, other: QueryExpression) -> QueryExpression:
        return _Expression("and", (self, other))

    def __or__(self, other: QueryExpression) -> QueryExpression:
        return _Expression("or", (self, other))

    def __invert__(self) -> QueryExpression:
        return _Expression("not", (self,))


@dataclass(frozen=True)
class _Scope:
    path: tuple[str, ...] = ()
    predicate: QueryExpression | None = None

    def field(self, name: str) -> QueryValue:
        return cast(QueryValue, _Value("field", (*self.path, name)))

    def scope(self, name: str) -> QueryScope:
        return _Scope((*self.path, name))


class _ProbeContext(_Scope):
    def constant(self, value: object) -> QueryValue:
        return cast(QueryValue, _Value("constant", value))

    def null(self) -> QueryValue:
        return cast(QueryValue, _Value("null", None))

    def always_true(self) -> QueryExpression:
        return _Expression("true", ())

    def always_false(self) -> QueryExpression:
        return _Expression("false", ())

    def compare(self, left: QueryValue, operator: str, right: QueryValue) -> QueryExpression:
        return _Expression("compare", (left, operator, right))

    def equal(self, left: QueryValue, right: QueryValue) -> QueryExpression:
        return _Expression("equal", (left, right))

    def exact_equal(self, left: QueryValue, right: QueryValue) -> QueryExpression:
        return _Expression("exact_equal", (left, right))

    def is_null(self, value: QueryValue) -> QueryExpression:
        return _Expression("is_null", (value,))

    def exists(self, scope: QueryScope, predicate: QueryExpression) -> QueryExpression:
        return _Expression("exists", (scope, predicate))

    def filtered(self, scope: QueryScope, predicate: QueryExpression) -> QueryScope:
        assert isinstance(scope, _Scope)
        return _Scope(scope.path, predicate)

    def count(self, scope: QueryScope) -> QueryValue:
        return cast(QueryValue, _Value("count", scope))

    def distinct_count(self, scope: QueryScope, value: QueryValue) -> QueryValue:
        return cast(QueryValue, _Value("distinct_count", (scope, value)))

    def scaled_exact_equal(
        self, left: QueryValue, left_factor: QueryValue, right: QueryValue, right_factor: QueryValue
    ) -> QueryExpression:
        return _Expression("scaled_exact_equal", (left, left_factor, right, right_factor))

    def and_(self, *predicates: QueryExpression) -> QueryExpression:
        return _Expression("and", tuple(predicates))

    def or_(self, *predicates: QueryExpression) -> QueryExpression:
        return _Expression("or", tuple(predicates))

    def not_(self, predicate: QueryExpression) -> QueryExpression:
        return _Expression("not", (predicate,))

    def when_known(self, known: QueryExpression, predicate: QueryExpression) -> QueryExpression:
        return _Expression("when_known", (known, predicate))


def _walk(value: object) -> tuple[object, ...]:
    values = [value]
    if isinstance(value, _Expression):
        for item in value.values:
            values.extend(_walk(item))
    elif isinstance(value, _Value):
        values.extend(_walk(value.value))
    elif isinstance(value, _Scope) and value.predicate is not None:
        values.extend(_walk(value.predicate))
    elif isinstance(value, tuple):
        for item in value:
            values.extend(_walk(item))
    return tuple(values)


@pytest.mark.parametrize(
    "record_type",
    (UnitcellStructureRecord, FundamentalDomainStructureRecord, ASUStructureRecord),
)
def test_every_backing_declares_the_complete_structure_property_surface(record_type: type[object]) -> None:
    projections = stored_property_projections(record_type)
    assert set(projections) == _EXPECTED
    assert "id" not in projections
    assert "type" not in projections


@pytest.mark.parametrize(
    "record",
    (
        _unitcell_record(_unitcell()),
        _domain_record(_domain(FundamentalDomainStructure)),
        _domain_record(_domain(ASUStructure)),
    ),
)
def test_stored_property_responses_match_the_natural_structure_provider(record: object) -> None:
    projections = stored_property_projections(type(record))
    (natural,) = tuple(StructureEntryProvider({"entry": record}).records("structures"))
    assert {name: projection.response(record) for name, projection in projections.items()} == {
        name: natural[name] for name in projections
    }


@pytest.mark.parametrize(
    "record",
    (
        _unitcell_record(_unitcell()),
        _domain_record(_domain(FundamentalDomainStructure)),
        _domain_record(_domain(ASUStructure)),
    ),
)
def test_response_callbacks_do_not_rebuild_an_all_property_projection(
    record: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from httk.atomistic.entries import structures as structure_entries

    def forbidden_projection(*args: object, **kwargs: object) -> object:
        raise AssertionError("a stored response callback must not construct every other response property")

    monkeypatch.setattr(structure_entries, "_structure_projection", forbidden_projection)
    projections = stored_property_projections(type(record))
    for projection in projections.values():
        projection.response(record)


def test_incomplete_normalized_composition_matches_the_natural_null_projection() -> None:
    unknown_attachment = Species("C_X", ("C",), (1,), attached=("X",), nattached=(1,))
    source = UnitcellStructure([[3, 0, 0], [0, 3, 0], [0, 0, 3]], [[0, 0, 0]], [unknown_attachment], ["C_X"])
    record = _unitcell_record(source)
    projections = stored_property_projections(UnitcellStructureRecord)
    (natural,) = tuple(StructureEntryProvider({"entry": record}).records("structures"))

    assert not record.normalized_composition.complete
    for name in ("elements", "nelements", "elements_ratios", "chemical_formula_reduced", "chemical_formula_anonymous"):
        assert projections[name].response(record) is None
        assert projections[name].response(record) == natural[name]


def test_formula_and_precision_queries_construct_exact_normalized_predicates() -> None:
    projections = stored_property_projections(UnitcellStructureRecord)
    context = cast(QueryContext, _ProbeContext())

    reduced = projections["chemical_formula_reduced"].query
    assert reduced is not None
    expression = reduced(context, "=", "ClNa")
    rendered = _walk(expression)
    assert _Value("field", ("normalized_composition", "amounts", "ratio")) in rendered
    assert _Value("constant", Fraction(1, 2)) in rendered
    assert any(isinstance(value, _Expression) and value.tag == "when_known" for value in rendered)
    with pytest.raises(QueryLiteralError, match="alphabetical"):
        reduced(context, "=", "NaCl")

    anonymous = projections["chemical_formula_anonymous"].query
    assert anonymous is not None
    anonymous_expression = anonymous(context, "=", "A2B2C")
    assert any(isinstance(value, _Value) and value.value == Fraction(2, 5) for value in _walk(anonymous_expression))

    precision = projections["_httk_coordinate_precision"].query
    assert precision is not None
    precision_expression = precision(context, "=", "1/1000")
    assert _Value("constant", Fraction(1, 1000)) in _walk(precision_expression)
    assert projections["_httk_coordinate_precision"].sort is None


def test_empty_complete_composition_keeps_formulas_unknown_but_elements_known() -> None:
    source = UnitcellStructure([[3, 0, 0], [0, 3, 0], [0, 0, 3]], [], (), ())
    record = _unitcell_record(source)
    projections = stored_property_projections(UnitcellStructureRecord)
    context = cast(QueryContext, _ProbeContext())

    assert record.normalized_composition.complete
    assert record.normalized_composition.amounts == ()
    assert projections["elements"].response(record) == []
    assert projections["nelements"].response(record) == 0
    assert projections["elements_ratios"].response(record) == []
    assert projections["chemical_formula_reduced"].response(record) is None
    assert projections["chemical_formula_anonymous"].response(record) is None
    formula_query = projections["chemical_formula_reduced"].query
    assert formula_query is not None
    formula_known = formula_query(context, "IS_KNOWN", None)
    assert _Value("count", _Scope(("normalized_composition", "amounts"))) in _walk(formula_known)


def test_spacegroup_settings_are_cached_by_it_number() -> None:
    import httk.atomistic.storage.stored_properties as stored_properties
    from httk.atomistic import data

    stored_properties._settings_by_it_number.cache_clear()
    grouped = stored_properties._settings_by_it_number()
    assert sum(len(settings) for settings in grouped.values()) == len(data.spacegroup_settings())
    assert len(grouped[225]) < len(data.spacegroup_settings())
    assert stored_properties._settings_by_it_number() is grouped


@pytest.mark.parametrize(
    ("record_type", "site_name"),
    (
        (UnitcellStructureRecord, ("species_at_sites", "value")),
        (FundamentalDomainStructureRecord, ("domain_sites", "species")),
        (ASUStructureRecord, ("domain_sites", "species")),
    ),
)
def test_periodic_dimension_and_feature_queries_use_native_scopes(
    record_type: type[object], site_name: tuple[str, str]
) -> None:
    projections = stored_property_projections(record_type)
    context = cast(QueryContext, _ProbeContext())

    periodic = projections["nperiodic_dimensions"].query
    assert periodic is not None
    expression = periodic(context, "=", 3)
    assert _Value("field", ("cell", "periodicity", "value")) in _walk(expression)

    features = projections["structure_features"].query
    assert features is not None
    feature_expression = features(context, "=", ["site_attachments", "disorder"])
    assert _Value("field", ("species", "attached_present")) in _walk(feature_expression)
    assert _Value("field", site_name) in _walk(feature_expression)


@pytest.mark.parametrize(
    ("record_type", "accepted", "not_represented"),
    (
        (UnitcellStructureRecord, "unit_cell", "asymmetric_unit"),
        (FundamentalDomainStructureRecord, "fundamental_domain", "asymmetric_unit"),
        (ASUStructureRecord, "asymmetric_unit", "fundamental_domain"),
    ),
)
def test_coordinate_span_queries_are_representation_specific(
    record_type: type[object], accepted: str, not_represented: str
) -> None:
    query = stored_property_projections(record_type)["site_coordinate_span"].query
    assert query is not None
    context = cast(QueryContext, _ProbeContext())
    assert _Value("constant", False) in _walk(query(context, "=", accepted))
    assert _Expression("false", ()) in _walk(query(context, "=", not_represented))
    assert _Expression("true", ()) in _walk(query(context, "!=", not_represented))
    with pytest.raises(QueryLiteralError):
        query(context, "=", "not_a_coordinate_span")


@pytest.mark.parametrize("span", ("molecular_entities", "other"))
def test_unrepresented_standard_coordinate_spans_constant_fold(span: str) -> None:
    context = cast(QueryContext, _ProbeContext())
    for record_type in (UnitcellStructureRecord, FundamentalDomainStructureRecord, ASUStructureRecord):
        query = stored_property_projections(record_type)["site_coordinate_span"].query
        assert query is not None
        assert _Expression("false", ()) in _walk(query(context, "=", span))
        assert _Expression("true", ()) in _walk(query(context, "!=", span))


def test_last_modified_queries_parse_offset_equivalent_rfc3339_instants() -> None:
    query = stored_property_projections(UnitcellStructureRecord)["last_modified"].query
    assert query is not None
    context = cast(QueryContext, _ProbeContext())
    utc = datetime.datetime(2026, 8, 2, 10, 30, tzinfo=datetime.UTC)

    equality = query(context, "=", "2026-08-02t10:30:00z")
    ordering = query(context, ">=", "2026-08-02T12:30:00+02:00")
    assert _Value("constant", utc) in _walk(equality)
    assert _Value("constant", utc) in _walk(ordering)
    with pytest.raises(QueryLiteralError, match="timezone-aware"):
        query(context, "=", "2026-08-02T10:30:00")


def test_structure_feature_literals_are_order_independent() -> None:
    query = stored_property_projections(UnitcellStructureRecord)["structure_features"].query
    assert query is not None
    context = cast(QueryContext, _ProbeContext())
    assert query(context, "HAS_ALL", ["site_attachments", "disorder"]) == query(
        context, "HAS_ALL", ["disorder", "site_attachments"]
    )
    with pytest.raises(QueryLiteralError, match="repeat"):
        query(context, "HAS_ALL", ["disorder", "disorder"])
