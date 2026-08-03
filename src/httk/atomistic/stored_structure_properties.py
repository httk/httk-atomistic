"""Stored OPTIMADE property declarations for native structure backings.

The declarations in this module deliberately name the durable record graph,
not a storage implementation.  In particular, composition predicates use the
authoritative normalized-composition relation and exact cross multiplication;
they never compare presentation floats or persist rendered formula strings.
"""

import datetime
import functools
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping
from fractions import Fraction
from itertools import combinations
from typing import Any

from httk.core import (
    QueryContext,
    QueryExpression,
    QueryLiteralError,
    QueryScope,
    QueryValue,
    StoredPropertyProjection,
    SurdVector,
)

from ._optimade_payloads import assemblies_payload, species_payload
from .composition import anonymous_symbol
from .elements import SYMBOLS
from .precision_entries import PRECISION_PROPERTY_KEYS
from .symmetry_entries import SETTING_PROPERTY_KEYS

_ELEMENTS = frozenset(SYMBOLS)
_ELEMENT_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([1-9][0-9]*)?")
_ANONYMOUS_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]*)([1-9][0-9]*)?")
_RFC3339_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})\Z",
    re.IGNORECASE,
)

_STANDARD_PROPERTIES: tuple[str, ...] = (
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
)

_FEATURES = ("assemblies", "disorder", "implicit_atoms", "site_attachments")
_COORDINATE_SPANS = frozenset(
    {
        "unit_cell",
        "molecular_unit_cell",
        "fundamental_domain",
        "molecular_fundamental_domain",
        "asymmetric_unit",
        "molecular_asymmetric_unit",
        "molecular_entities",
        "other",
    }
)


def _normalized_composition_value(record: Any, name: str) -> object:
    """Serve composition directly from its canonical durable relation."""
    composition = record.normalized_composition
    if not composition.complete:
        return None
    if name == "elements":
        return [value.element for value in composition.amounts]
    if name == "nelements":
        return len(composition.amounts)
    if name == "elements_ratios":
        return [float(value.ratio) for value in composition.amounts]
    from .structure_record import _composition_result_from_record

    result = _composition_result_from_record(composition)
    if name == "chemical_formula_reduced":
        return result.chemical_formula_reduced
    if name == "chemical_formula_anonymous":
        return result.chemical_formula_anonymous
    raise AssertionError(f"unknown normalized composition property: {name}")


def _used_species(record: Any, backing: str) -> tuple[Any, ...]:
    names = record.species_at_sites if backing == "unitcell" else tuple(site.species for site in record.domain_sites)
    by_name = {species.name: species for species in record.species}
    return tuple(by_name[name] for name in names)


def _structure_features_value(record: Any, backing: str) -> list[str]:
    """Derive the public flags from represented, not merely stored, species."""
    features: set[str] = set()
    used = _used_species(record, backing)
    if record.assemblies is not None:
        features.add("assemblies")
    if any(len(value.chemical_symbols) > 1 or value.concentration != (Fraction(1),) for value in used):
        features.add("disorder")
    if any(value.attached for value in used):
        features.add("site_attachments")
    if record.chemical_composition is not None and record.chemical_composition.mode == "implicit":
        features.add("implicit_atoms")
    return sorted(features)


def _coordinate_span(record: Any, backing: str) -> str:
    name = {
        "unitcell": "unit_cell",
        "fundamental_domain": "fundamental_domain",
        "asymmetric_unit": "asymmetric_unit",
    }[backing]
    return f"molecular_{name}" if record.molecular else name


@functools.cache
def _settings_by_it_number() -> dict[int, tuple[dict[str, Any], ...]]:
    from . import data

    grouped: dict[int, list[dict[str, Any]]] = {}
    for setting in data.spacegroup_settings():
        grouped.setdefault(setting["it_number"], []).append(setting)
    return {it_number: tuple(settings) for it_number, settings in grouped.items()}


def _domain_setting(record: Any) -> tuple[Any, Any | None, Any]:
    """Recover the stored setting without materializing a domain structure."""
    from .spacegroup import Spacegroup

    spacegroup = Spacegroup.standard(record.spacegroup_it_number)
    from .structure_record import _setting_transform_from_record

    transform = _setting_transform_from_record(record.setting_transform)
    if transform.is_identity():
        return spacegroup, spacegroup, transform
    if transform.hall_entry is not None:
        return spacegroup, Spacegroup.for_hall_entry(transform.hall_entry), transform
    for setting_record in _settings_by_it_number().get(spacegroup.it_number, ()):
        candidate = Spacegroup(setting_record)
        if candidate.transform_from_standard == transform:
            return spacegroup, candidate, transform
    return spacegroup, None, transform


def _setting_transform_payload(transform: Any) -> dict[str, object]:
    return {
        "matrix": [[str(value) for value in row] for row in transform.matrix.to_fractions()],
        "vector": [str(value) for value in transform.vector.to_fractions()],
        "xyz": transform.operation.to_xyz(),
    }


def _unitcell_symmetry_value(record: Any, name: str) -> object:
    symmetry = record.symmetry
    if name == "space_group_symmetry_operations_xyz":
        if symmetry is not None and symmetry.space_group_symmetry_operations_xyz is not None:
            return list(symmetry.space_group_symmetry_operations_xyz or ())
        return ["x,y,z"] if any(record.cell.periodicity) else None
    if name == "wyckoff_positions":
        return (
            None if symmetry is None or symmetry.wyckoff_positions is None else list(symmetry.wyckoff_positions or ())
        )
    if name in {
        "space_group_it_number",
        "space_group_symbol_hall",
        "space_group_symbol_hermann_mauguin",
        "space_group_symbol_hermann_mauguin_extended",
    }:
        return None if symmetry is None else getattr(symmetry, name)
    if name in SETTING_PROPERTY_KEYS:
        return None
    raise AssertionError(f"unknown unit-cell symmetry property: {name}")


def _domain_symmetry_value(record: Any, name: str) -> object:
    """Serve one setting-specific field without creating an all-property projection."""
    from .spacegroup import wyckoff_letter_map

    spacegroup, setting, transform = _domain_setting(record)
    if name == "space_group_it_number":
        return spacegroup.it_number
    if name == "space_group_symbol_hall":
        return None if setting is None else setting.hall_symbol
    if name == "space_group_symbol_hermann_mauguin":
        return None if setting is None else setting.hermann_mauguin
    if name == "space_group_symbol_hermann_mauguin_extended":
        if setting is None:
            return None
        value = setting.record.get("hm_extended")
        return None if not value else " ".join(part.strip() for part in str(value).split("\n") if part.strip())
    if name == "space_group_symmetry_operations_xyz":
        operations = (
            tuple(transform.symop_to_setting(value) for value in spacegroup.symmetry_operations)
            if setting is None
            else setting.symmetry_operations
        )
        return [operation.wrapped().to_xyz() for operation in operations]
    if name == "wyckoff_positions":
        if setting is None:
            return None
        letters = wyckoff_letter_map(spacegroup, setting)
        return [setting.wyckoff_position(letters[site.wyckoff]).letter for site in record.domain_sites]
    if name == "_httk_setting_it_nc":
        return None if setting is None else setting.setting
    if name == "_httk_hall_entry":
        return None if setting is None else setting.hall_entry
    if name == "_httk_is_reference_setting":
        return False if setting is None else setting.is_standard_setting
    if name == "_httk_crystal_system":
        return spacegroup.crystal_system if setting is None else setting.crystal_system
    if name == "_httk_centring_type":
        return None if setting is None else setting.centring_type
    if name == "_httk_setting_transform":
        return _setting_transform_payload(transform)
    raise AssertionError(f"unknown domain symmetry property: {name}")


def _response_value(record: Any, name: str, backing: str) -> object:
    """Serve exactly one property; never materialize unrelated response fields."""
    if name == "immutable_id":
        return record.immutable_id
    if name == "last_modified":
        return None if record.last_modified is None else record.last_modified.isoformat()
    if name in {
        "elements",
        "nelements",
        "elements_ratios",
        "chemical_formula_reduced",
        "chemical_formula_anonymous",
    }:
        return _normalized_composition_value(record, name)
    if name in {"chemical_formula_descriptive", "chemical_formula_hill", "optimization_type"}:
        return getattr(record, name)
    if name == "dimension_types":
        return [1 if value else 0 for value in record.cell.periodicity]
    if name == "nperiodic_dimensions":
        return sum(record.cell.periodicity)
    if name == "lattice_vectors":
        from .cell_view import CellView

        return CellView(record.cell).basis.to_floats()
    if name == "fractional_site_positions":
        if backing == "unitcell":
            return record.sites.reduced_coords.to_floats()
        from .structure_record import _domain_structure_from_record

        return _domain_structure_from_record(record).fractional_site_positions
    if name == "cartesian_site_positions":
        if backing == "unitcell":
            from .cell_view import CellView
            from .sites_view import SitesView

            return (SurdVector.create(SitesView(record.sites).reduced_coords) * CellView(record.cell).basis).to_floats()
        from .structure_record import _domain_structure_from_record

        return _domain_structure_from_record(record).cartesian_site_positions
    if name == "site_coordinate_span":
        return _coordinate_span(record, backing)
    if name == "site_coordinate_span_description":
        return None
    if name == "nsites":
        return len(record.species_at_sites) if backing == "unitcell" else len(record.domain_sites)
    if name == "species_at_sites":
        return (
            list(record.species_at_sites) if backing == "unitcell" else [site.species for site in record.domain_sites]
        )
    if name == "species":
        return [species_payload(value) for value in record.species]
    if name == "assemblies":
        return assemblies_payload(record.assemblies)
    if name == "structure_features":
        return _structure_features_value(record, backing)
    if name in {
        "space_group_it_number",
        "space_group_symbol_hall",
        "space_group_symbol_hermann_mauguin",
        "space_group_symbol_hermann_mauguin_extended",
        "space_group_symmetry_operations_xyz",
        "wyckoff_positions",
        *SETTING_PROPERTY_KEYS,
    }:
        return _unitcell_symmetry_value(record, name) if backing == "unitcell" else _domain_symmetry_value(record, name)
    if name == "_httk_coordinate_precision":
        value = record.sites.precision if backing == "unitcell" else record.coordinate_precision
        return None if value is None else float(value)
    if name == "_httk_basis_precision":
        return None if record.cell.precision is None else float(record.cell.precision)
    raise AssertionError(f"unknown stored structure property: {name}")


def _response(name: str, backing: str) -> Callable[[object], object]:
    return lambda record: _response_value(record, name, backing)


def _value(context: QueryContext, path: tuple[str, ...]) -> QueryValue:
    scope: QueryScope = context
    for name in path[:-1]:
        scope = scope.scope(name)
    return scope.field(path[-1])


def _comparison(
    context: QueryContext,
    value: QueryValue,
    operator: str,
    literal: object,
    *,
    exact: bool = False,
) -> QueryExpression:
    if operator == "IS_UNKNOWN":
        return context.is_null(value)
    if operator == "IS_KNOWN":
        return context.not_(context.is_null(value))
    right = context.null() if literal is None else context.constant(literal)
    if operator == "=":
        return context.exact_equal(value, right) if exact else context.equal(value, right)
    if operator == "!=":
        equals = context.exact_equal(value, right) if exact else context.equal(value, right)
        return context.not_(equals)
    if operator not in {"<", "<=", ">", ">=", "CONTAINS", "STARTS", "ENDS"}:
        raise QueryLiteralError(f"unsupported structure-property operator: {operator}")
    return context.compare(value, operator, right)


def _string_query(path: tuple[str, ...]) -> Callable[[QueryContext, str, object], QueryExpression]:
    def query(context: QueryContext, operator: str, literal: object) -> QueryExpression:
        if literal is not None and not isinstance(literal, str):
            raise QueryLiteralError("structure string property requires a string or null literal")
        return _comparison(context, _value(context, path), operator, literal)

    return query


def _string_sort(path: tuple[str, ...]) -> Callable[[QueryContext], QueryValue]:
    return lambda context: _value(context, path)


def _timestamp_literal(literal: object) -> object:
    """Parse a complete timezone-aware RFC 3339 literal into its instant."""
    if literal is None:
        return None
    if not isinstance(literal, str) or _RFC3339_TIMESTAMP.fullmatch(literal) is None:
        raise QueryLiteralError("last_modified requires a timezone-aware RFC 3339 timestamp or null")
    try:
        value = datetime.datetime.fromisoformat(literal[:-1] + "+00:00" if literal[-1] in {"Z", "z"} else literal)
    except ValueError as error:
        raise QueryLiteralError("last_modified requires a valid RFC 3339 timestamp") from error
    if value.tzinfo is None or value.utcoffset() is None:  # defensive: the regex already requires an offset
        raise QueryLiteralError("last_modified requires a timezone-aware RFC 3339 timestamp")
    return value.astimezone(datetime.UTC)


def _timestamp_query(context: QueryContext, operator: str, literal: object) -> QueryExpression:
    """Filter timestamp instants, never their rendered offset-specific strings."""
    value = context.field("last_modified")
    if operator in {"IS_UNKNOWN", "IS_KNOWN"}:
        return _comparison(context, value, operator, literal)
    if operator not in {"=", "!=", "<", "<=", ">", ">="}:
        raise QueryLiteralError("last_modified supports equality, ordering, and unknown operators")
    return _comparison(context, value, operator, _timestamp_literal(literal))


def _fraction_literal(literal: object, *, property_name: str) -> Fraction:
    if isinstance(literal, bool):
        raise QueryLiteralError(f"{property_name} requires a finite rational literal")
    if isinstance(literal, Fraction):
        return literal
    if isinstance(literal, int):
        return Fraction(literal)
    if isinstance(literal, float):
        if not math.isfinite(literal):
            raise QueryLiteralError(f"{property_name} requires a finite rational literal")
        return Fraction(str(literal))
    if isinstance(literal, str):
        try:
            return Fraction(literal)
        except (ValueError, ZeroDivisionError) as error:
            raise QueryLiteralError(f"{property_name} requires a rational literal") from error
    raise QueryLiteralError(f"{property_name} requires a rational literal")


def _fraction_query(
    path: tuple[str, ...], property_name: str
) -> Callable[[QueryContext, str, object], QueryExpression]:
    def query(context: QueryContext, operator: str, literal: object) -> QueryExpression:
        value = _value(context, path)
        if operator in {"IS_UNKNOWN", "IS_KNOWN"}:
            return _comparison(context, value, operator, literal)
        return _comparison(
            context, value, operator, _fraction_literal(literal, property_name=property_name), exact=True
        )

    return query


def _composition_scope(context: QueryContext) -> QueryScope:
    return context.scope("normalized_composition")


def _complete_composition(context: QueryContext, predicate: QueryExpression) -> QueryExpression:
    return context.when_known(_composition_known(context), predicate)


def _composition_known(context: QueryContext) -> QueryExpression:
    composition = _composition_scope(context)
    return context.exists(composition, context.equal(composition.field("complete"), context.constant(True)))


def _formula_known(context: QueryContext) -> QueryExpression:
    """A formula exists only for a complete non-empty elemental composition."""
    amounts = _composition_scope(context).scope("amounts")
    return context.and_(
        _composition_known(context),
        context.compare(context.count(amounts), ">", context.constant(0)),
    )


def _formula_tokens(literal: object, *, anonymous: bool) -> tuple[tuple[str, int], ...]:
    if not isinstance(literal, str) or not literal:
        raise QueryLiteralError("chemical formula equality requires a non-empty formula string")
    token = _ANONYMOUS_FORMULA_TOKEN if anonymous else _ELEMENT_FORMULA_TOKEN
    position = 0
    values: list[tuple[str, int]] = []
    while position < len(literal):
        match = token.match(literal, position)
        if match is None:
            raise QueryLiteralError("chemical formula has invalid syntax")
        symbol = match.group(1)
        coefficient = int(match.group(2) or 1)
        values.append((symbol, coefficient))
        position = match.end()
    if anonymous:
        expected = tuple(anonymous_symbol(index) for index in range(len(values)))
        if tuple(symbol for symbol, _ in values) != expected:
            raise QueryLiteralError("anonymous chemical formula must use consecutive anonymous symbols")
        coefficients = tuple(value for _, value in values)
        if tuple(sorted(coefficients, reverse=True)) != coefficients:
            raise QueryLiteralError("anonymous chemical formula coefficients must be sorted descending")
    else:
        elements = tuple(symbol for symbol, _ in values)
        if any(element not in _ELEMENTS for element in elements):
            raise QueryLiteralError("chemical formula contains an unknown element")
        if elements != tuple(sorted(elements)) or len(elements) != len(set(elements)):
            raise QueryLiteralError("reduced chemical formula elements must be unique and alphabetical")
    return tuple(values)


def _formula_query(*, anonymous: bool) -> Callable[[QueryContext, str, object], QueryExpression]:
    property_name = "chemical_formula_anonymous" if anonymous else "chemical_formula_reduced"

    def query(context: QueryContext, operator: str, literal: object) -> QueryExpression:
        if operator == "IS_KNOWN":
            return _formula_known(context)
        if operator == "IS_UNKNOWN":
            return context.not_(_formula_known(context))
        if operator not in {"=", "!="}:
            raise QueryLiteralError(f"{property_name} supports exact equality only")
        tokens = _formula_tokens(literal, anonymous=anonymous)
        composition = _composition_scope(context)
        result = context.when_known(
            _formula_known(context),
            _anonymous_formula_predicate(context, composition, tuple(value for _, value in tokens))
            if anonymous
            else _reduced_formula_predicate(context, composition, tokens),
        )
        return result if operator == "=" else context.not_(result)

    return query


def _reduced_formula_predicate(
    context: QueryContext, composition: QueryScope, tokens: tuple[tuple[str, int], ...]
) -> QueryExpression:
    """Match named formula coefficients against canonical exact ratios."""
    amounts = composition.scope("amounts")
    predicates: list[QueryExpression] = [
        context.compare(context.count(amounts), "=", context.constant(len(tokens))),
    ]
    total = sum(value for _, value in tokens)
    for element, coefficient in tokens:
        candidate = composition.scope("amounts")
        matching = context.filtered(
            candidate,
            context.and_(
                context.exact_equal(candidate.field("element"), context.constant(element)),
                context.exact_equal(candidate.field("ratio"), context.constant(Fraction(coefficient, total))),
            ),
        )
        predicates.append(context.compare(context.count(matching), "=", context.constant(1)))
    return context.and_(*predicates)


def _anonymous_formula_predicate(
    context: QueryContext, composition: QueryScope, coefficients: tuple[int, ...]
) -> QueryExpression:
    """Match an anonymous coefficient multiset against canonical exact ratios."""
    amounts = composition.scope("amounts")
    predicates: list[QueryExpression] = [
        context.compare(context.count(amounts), "=", context.constant(len(coefficients))),
    ]
    total = sum(coefficients)
    for coefficient, multiplicity in Counter(coefficients).items():
        peer = composition.scope("amounts")
        matching = context.filtered(
            peer,
            context.exact_equal(peer.field("ratio"), context.constant(Fraction(coefficient, total))),
        )
        predicates.append(context.compare(context.count(matching), "=", context.constant(multiplicity)))
    return context.and_(*predicates)


def _element_literals(literal: object) -> tuple[str, ...]:
    if not isinstance(literal, tuple | list) or not all(
        isinstance(value, str) and value in _ELEMENTS for value in literal
    ):
        raise QueryLiteralError("elements requires a list of valid element symbols")
    values = tuple(literal)
    if len(values) != len(set(values)):
        raise QueryLiteralError("elements literal must not repeat an element")
    return values


def _elements_query(context: QueryContext, operator: str, literal: object) -> QueryExpression:
    if operator == "IS_KNOWN":
        return _composition_known(context)
    if operator == "IS_UNKNOWN":
        return context.not_(_composition_known(context))
    composition = _composition_scope(context)
    amounts = composition.scope("amounts")
    if operator.startswith("LENGTH "):
        size = literal
        if not isinstance(size, int) or isinstance(size, bool):
            raise QueryLiteralError("elements LENGTH requires an integer literal")
        return _complete_composition(
            context,
            context.compare(context.count(amounts), operator.removeprefix("LENGTH "), context.constant(size)),
        )
    values = _element_literals(literal)
    matching = [
        context.exists(amounts, context.exact_equal(amounts.field("element"), context.constant(value)))
        for value in values
    ]
    if operator == "HAS_ANY":
        predicate = context.or_(*matching)
    elif operator == "HAS_ALL":
        predicate = context.and_(*matching)
    elif operator == "HAS_ONLY":
        allowed = context.or_(
            *[context.exact_equal(amounts.field("element"), context.constant(value)) for value in values]
        )
        predicate = context.compare(
            context.count(context.filtered(amounts, context.not_(allowed))), "=", context.constant(0)
        )
    else:
        raise QueryLiteralError(
            "elements supports HAS, LENGTH, and unknown operators; ordered-list equality is unavailable"
        )
    return _complete_composition(context, predicate)


def _nelements_query(context: QueryContext, operator: str, literal: object) -> QueryExpression:
    if operator == "IS_KNOWN":
        return _composition_known(context)
    if operator == "IS_UNKNOWN":
        return context.not_(_composition_known(context))
    if not isinstance(literal, int) or isinstance(literal, bool):
        raise QueryLiteralError("nelements requires an integer literal")
    amounts = _composition_scope(context).scope("amounts")
    return _complete_composition(context, context.compare(context.count(amounts), operator, context.constant(literal)))


def _ratio_literals(literal: object) -> tuple[Fraction, ...]:
    if not isinstance(literal, tuple | list):
        raise QueryLiteralError("elements_ratios requires a list of rational literals")
    values = tuple(_fraction_literal(value, property_name="elements_ratios") for value in literal)
    if any(not 0 < value <= 1 for value in values):
        raise QueryLiteralError("elements_ratios values must be in (0, 1]")
    return values


def _elements_ratios_query(context: QueryContext, operator: str, literal: object) -> QueryExpression:
    if operator == "IS_KNOWN":
        return _composition_known(context)
    if operator == "IS_UNKNOWN":
        return context.not_(_composition_known(context))
    amounts = _composition_scope(context).scope("amounts")
    if operator.startswith("LENGTH "):
        if not isinstance(literal, int) or isinstance(literal, bool):
            raise QueryLiteralError("elements_ratios LENGTH requires an integer literal")
        return _complete_composition(
            context,
            context.compare(context.count(amounts), operator.removeprefix("LENGTH "), context.constant(literal)),
        )
    values = _ratio_literals(literal)

    def matching(value: Fraction) -> QueryScope:
        return context.filtered(
            amounts,
            context.exact_equal(amounts.field("ratio"), context.constant(value)),
        )

    if operator == "HAS_ANY":
        predicate = context.or_(*[context.exists(matching(value), context.always_true()) for value in values])
    elif operator == "HAS_ALL":
        predicate = context.and_(*[context.exists(matching(value), context.always_true()) for value in values])
    elif operator == "HAS_ONLY":
        allowed = context.or_(
            *[context.exact_equal(amounts.field("ratio"), context.constant(value)) for value in values]
        )
        predicate = context.compare(
            context.count(context.filtered(amounts, context.not_(allowed))), "=", context.constant(0)
        )
    else:
        raise QueryLiteralError(
            "elements_ratios supports HAS, LENGTH, and unknown operators; ordered-list equality is unavailable"
        )
    return _complete_composition(context, predicate)


def _span_query(spans: Mapping[str, bool]) -> Callable[[QueryContext, str, object], QueryExpression]:

    def query(context: QueryContext, operator: str, literal: object) -> QueryExpression:
        if operator == "IS_KNOWN":
            return context.always_true()
        if operator == "IS_UNKNOWN":
            return context.always_false()
        if operator not in {"=", "!="}:
            raise QueryLiteralError("site_coordinate_span supports equality and unknown operators")
        if not isinstance(literal, str) or literal not in _COORDINATE_SPANS:
            raise QueryLiteralError("site_coordinate_span requires a standard coordinate-span literal")
        if literal not in spans:
            return context.always_false() if operator == "=" else context.always_true()
        result = context.equal(context.field("molecular"), context.constant(spans[literal]))
        return result if operator == "=" else context.not_(result)

    return query


def _used_species_scope(context: QueryContext, site_scope: QueryScope, site_name: str) -> QueryScope:
    """Return only species actually referenced by this backing's represented sites."""
    species = context.scope("species")
    return context.filtered(
        species,
        context.exists(
            site_scope,
            context.exact_equal(site_scope.field(site_name), species.field("name")),
        ),
    )


def _feature_predicate(context: QueryContext, name: str, species: QueryScope) -> QueryExpression:
    if name == "assemblies":
        return context.equal(context.field("assemblies_present"), context.constant(True))
    if name == "implicit_atoms":
        composition = context.scope("chemical_composition")
        return context.exists(composition, context.equal(composition.field("mode"), context.constant("implicit")))
    if name == "site_attachments":
        return context.exists(species, context.equal(species.field("attached_present"), context.constant(True)))
    if name == "disorder":
        symbols = species.scope("chemical_symbols")
        concentration = species.scope("concentration")
        partial = context.exists(
            concentration,
            context.not_(context.exact_equal(concentration.field("value"), context.constant(Fraction(1)))),
        )
        return context.exists(
            species,
            context.or_(context.compare(context.count(symbols), ">", context.constant(1)), partial),
        )
    raise AssertionError(f"unknown structure feature: {name}")


def _feature_literals(literal: object) -> tuple[str, ...]:
    if not isinstance(literal, tuple | list) or not all(
        isinstance(value, str) and value in _FEATURES for value in literal
    ):
        raise QueryLiteralError("structure_features requires a list of standard feature strings")
    values = tuple(literal)
    if len(values) != len(set(values)):
        raise QueryLiteralError("structure_features literal must not repeat a feature")
    return tuple(sorted(values))


def _structure_features_query(
    used_species: Callable[[QueryContext], QueryScope],
) -> Callable[[QueryContext, str, object], QueryExpression]:
    def query(context: QueryContext, operator: str, literal: object) -> QueryExpression:
        if operator == "IS_KNOWN":
            return context.always_true()
        if operator == "IS_UNKNOWN":
            return context.always_false()
        species = used_species(context)
        if operator.startswith("LENGTH "):
            if not isinstance(literal, int) or isinstance(literal, bool) or literal < 0:
                raise QueryLiteralError("structure_features LENGTH requires a non-negative integer literal")
            alternatives: list[QueryExpression] = []
            for selected in combinations(_FEATURES, literal):
                selected_set = frozenset(selected)
                alternatives.append(
                    context.and_(
                        *[
                            _feature_predicate(context, name, species)
                            if name in selected_set
                            else context.not_(_feature_predicate(context, name, species))
                            for name in _FEATURES
                        ]
                    )
                )
            return context.or_(*alternatives)
        values = _feature_literals(literal)
        matches = [_feature_predicate(context, value, species) for value in values]
        if operator == "HAS_ANY":
            return context.or_(*matches)
        if operator == "HAS_ALL":
            return context.and_(*matches)
        if operator == "HAS_ONLY":
            allowed = frozenset(values)
            return context.and_(
                *[
                    _feature_predicate(context, name, species)
                    if name in allowed
                    else context.not_(_feature_predicate(context, name, species))
                    for name in _FEATURES
                ]
            )
        if operator in {"=", "!="}:
            allowed = frozenset(values)
            result = context.and_(
                *[
                    _feature_predicate(context, name, species)
                    if name in allowed
                    else context.not_(_feature_predicate(context, name, species))
                    for name in _FEATURES
                ]
            )
            return result if operator == "=" else context.not_(result)
        raise QueryLiteralError("structure_features supports HAS, equality, LENGTH, and unknown operators")

    return query


def _count_query(
    scope_path: tuple[str, ...], property_name: str
) -> Callable[[QueryContext, str, object], QueryExpression]:
    def query(context: QueryContext, operator: str, literal: object) -> QueryExpression:
        scope: QueryScope = context
        for name in scope_path:
            scope = scope.scope(name)
        if operator == "IS_KNOWN":
            return context.always_true()
        if operator == "IS_UNKNOWN":
            return context.always_false()
        if not isinstance(literal, int) or isinstance(literal, bool):
            raise QueryLiteralError(f"{property_name} requires an integer literal")
        return context.compare(context.count(scope), operator, context.constant(literal))

    return query


def _nperiodic_dimensions_query(context: QueryContext, operator: str, literal: object) -> QueryExpression:
    """Count the persisted boolean periodicity rows without deriving a cache column."""
    if operator == "IS_KNOWN":
        return context.always_true()
    if operator == "IS_UNKNOWN":
        return context.always_false()
    if not isinstance(literal, int) or isinstance(literal, bool) or not 0 <= literal <= 3:
        raise QueryLiteralError("nperiodic_dimensions requires an integer literal in [0, 3]")
    periodicity = context.scope("cell").scope("periodicity")
    periodic_rows = context.filtered(
        periodicity,
        context.exact_equal(periodicity.field("value"), context.constant(True)),
    )
    return context.compare(context.count(periodic_rows), operator, context.constant(literal))


def _symmetry_query(path: tuple[str, ...]) -> Callable[[QueryContext, str, object], QueryExpression]:
    def query(context: QueryContext, operator: str, literal: object) -> QueryExpression:
        scope: QueryScope = context
        for name in path[:-1]:
            scope = scope.scope(name)
        value = scope.field(path[-1])
        if operator in {"IS_KNOWN", "IS_UNKNOWN"}:
            return _comparison(context, value, operator, literal)
        if not isinstance(literal, int) or isinstance(literal, bool):
            raise QueryLiteralError("space_group_it_number requires an integer literal")
        return _comparison(context, value, operator, literal)

    return query


def _base_projections(backing: str) -> dict[str, StoredPropertyProjection]:
    return {name: StoredPropertyProjection(response=_response(name, backing)) for name in _STANDARD_PROPERTIES}


def _common_queries(
    projections: dict[str, StoredPropertyProjection],
    *,
    backing: str,
    coordinate_path: tuple[str, ...],
    used_species: Callable[[QueryContext], QueryScope],
) -> None:
    for name in ("immutable_id", "chemical_formula_descriptive", "chemical_formula_hill", "optimization_type"):
        projections[name] = StoredPropertyProjection(
            response=_response(name, backing), query=_string_query((name,)), sort=_string_sort((name,))
        )
    projections["last_modified"] = StoredPropertyProjection(
        response=_response("last_modified", backing), query=_timestamp_query, sort=_string_sort(("last_modified",))
    )
    projections["elements"] = StoredPropertyProjection(response=_response("elements", backing), query=_elements_query)
    projections["nelements"] = StoredPropertyProjection(
        response=_response("nelements", backing), query=_nelements_query
    )
    projections["elements_ratios"] = StoredPropertyProjection(
        response=_response("elements_ratios", backing), query=_elements_ratios_query
    )
    projections["chemical_formula_reduced"] = StoredPropertyProjection(
        response=_response("chemical_formula_reduced", backing), query=_formula_query(anonymous=False)
    )
    projections["chemical_formula_anonymous"] = StoredPropertyProjection(
        response=_response("chemical_formula_anonymous", backing), query=_formula_query(anonymous=True)
    )
    projections["structure_features"] = StoredPropertyProjection(
        response=_response("structure_features", backing), query=_structure_features_query(used_species)
    )
    projections["nperiodic_dimensions"] = StoredPropertyProjection(
        response=_response("nperiodic_dimensions", backing), query=_nperiodic_dimensions_query
    )
    # ``dimension_types`` is a positional list. The neutral scope protocol has
    # no child-position selector, so only its response projection is exact.
    projections["_httk_coordinate_precision"] = StoredPropertyProjection(
        response=_response("_httk_coordinate_precision", backing),
        query=_fraction_query(coordinate_path, "_httk_coordinate_precision"),
    )
    projections["_httk_basis_precision"] = StoredPropertyProjection(
        response=_response("_httk_basis_precision", backing),
        query=_fraction_query(("cell", "precision"), "_httk_basis_precision"),
    )
    for name in (*SETTING_PROPERTY_KEYS, *PRECISION_PROPERTY_KEYS):
        projections.setdefault(name, StoredPropertyProjection(response=_response(name, backing)))


def unitcell_structure_properties() -> Mapping[str, StoredPropertyProjection]:
    """Return the declaration map for :class:`httk.atomistic.UnitcellStructureRecord`."""
    projections = _base_projections("unitcell")
    _common_queries(
        projections,
        backing="unitcell",
        coordinate_path=("sites", "precision"),
        used_species=lambda context: _used_species_scope(context, context.scope("species_at_sites"), "value"),
    )
    projections["site_coordinate_span"] = StoredPropertyProjection(
        response=_response("site_coordinate_span", "unitcell"),
        query=_span_query({"molecular_unit_cell": True, "unit_cell": False}),
    )
    projections["nsites"] = StoredPropertyProjection(
        response=_response("nsites", "unitcell"), query=_count_query(("sites", "reduced_coords"), "nsites")
    )
    projections["space_group_it_number"] = StoredPropertyProjection(
        response=_response("space_group_it_number", "unitcell"),
        query=_symmetry_query(("symmetry", "space_group_it_number")),
    )
    return projections


def _domain_structure_properties(*, asymmetric_unit: bool) -> Mapping[str, StoredPropertyProjection]:
    """Return the declaration map for one non-expanded symmetry backing."""
    backing = "asymmetric_unit" if asymmetric_unit else "fundamental_domain"
    projections = _base_projections(backing)
    _common_queries(
        projections,
        backing=backing,
        coordinate_path=("coordinate_precision",),
        used_species=lambda context: _used_species_scope(context, context.scope("domain_sites"), "species"),
    )
    projections["site_coordinate_span"] = StoredPropertyProjection(
        response=_response("site_coordinate_span", backing),
        query=_span_query(
            {"molecular_asymmetric_unit": True, "asymmetric_unit": False}
            if asymmetric_unit
            else {"molecular_fundamental_domain": True, "fundamental_domain": False}
        ),
    )
    projections["nsites"] = StoredPropertyProjection(
        response=_response("nsites", backing), query=_count_query(("domain_sites",), "nsites")
    )
    projections["space_group_it_number"] = StoredPropertyProjection(
        response=_response("space_group_it_number", backing), query=_symmetry_query(("spacegroup_it_number",))
    )
    return projections


def fundamental_domain_structure_properties() -> Mapping[str, StoredPropertyProjection]:
    """Return the declaration map for fundamental-domain backings."""
    return _domain_structure_properties(asymmetric_unit=False)


def asymmetric_unit_structure_properties() -> Mapping[str, StoredPropertyProjection]:
    """Return the declaration map for asymmetric-unit backings."""
    return _domain_structure_properties(asymmetric_unit=True)


def attach_structure_property_projections(
    unitcell: type[Any], fundamental_domain: type[Any], asymmetric_unit: type[Any]
) -> None:
    """Attach exact-class declaration maps without importing a storage capability module."""
    unitcell.__httk_stored_properties__ = unitcell_structure_properties()
    fundamental_domain.__httk_stored_properties__ = fundamental_domain_structure_properties()
    asymmetric_unit.__httk_stored_properties__ = asymmetric_unit_structure_properties()
