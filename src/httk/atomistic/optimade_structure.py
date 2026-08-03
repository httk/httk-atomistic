"""Lazy, exact OPTIMADE-backed structure representation.

The transport spelling of an OPTIMADE property is deliberately never part of
the conversion contract.  The accompanying ``/info/structures`` snapshot maps
each spelling to a property-definition IRI; only that IRI selects a local
meaning.
"""

import datetime
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import cached_property
from types import MappingProxyType, SimpleNamespace
from typing import Any, ClassVar, cast
from urllib.parse import urlsplit

from httk.core import (
    EntryTypeDefinition,
    SurdVector,
    combined_precision,
    decimal_precision,
    load_entry_type_definition,
)
from httk.core.optimade import (
    IncompleteOptimadeResourceError,
    OptimadeResource,
    decode_optimade_value,
    optimade_document_root,
)
from httk.core.storage import stored_property

from ._composition_values import normalization
from .cell import Cell
from .composition import Assembly, CompositionResult, anonymous_symbol, project_composition, validate_assemblies
from .precision_entries import precision_definitions
from .sites import Sites
from .spacegroup import Spacegroup
from .species import Species
from .structure_backend import StructureBackend
from .structure_semantics import (
    _ELEMENTS,
    _FORMULA_TOKEN,
    _OPTIMIZATION_TYPES,
    _SYMOP_COORDINATE,
    StructureSymmetry,
    validate_descriptive_formula,
    validate_hill_formula,
)

_STRUCTURES_DEFINITION_ID = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"
_MISSING = object()
_COORDINATE_SPANS = frozenset(
    {
        "fundamental_domain",
        "asymmetric_unit",
        "molecular_fundamental_domain",
        "molecular_asymmetric_unit",
        "unit_cell",
        "molecular_unit_cell",
        "molecular_entities",
        "other",
    }
)
_UNIT_CELL_SPANS = frozenset({"unit_cell", "molecular_unit_cell"})
_STRUCTURE_FEATURES = frozenset({"assemblies", "disorder", "implicit_atoms", "site_attachments"})
_ANONYMOUS_TOKEN = re.compile(r"([A-Z][a-z]*)([1-9][0-9]*)?")
_WYCKOFF = frozenset("abcdefghijklmnopqrstuvwxyzα")


def _is_definition_iri(value: object) -> bool:
    """Return whether *value* is a minimally well-formed absolute IRI."""

    if not isinstance(value, str) or not value or value != value.strip():
        return False
    try:
        return bool(urlsplit(value).scheme)
    except ValueError:
        return False


@dataclass(frozen=True, init=False)
class OptimadeStructure(StructureBackend):
    """A structure backend whose exact source is an :class:`~httk.core.optimade.OptimadeResource`.

    Construction merely retains the resource.  The canonical structure quartet
    is decoded one component at a time, so an incomplete remote resource is
    still storable, inspectable, and round-trippable.
    """

    resource: OptimadeResource

    kind: ClassVar[str] = "optimade"
    entry_type_definition_id: ClassVar[str] = _STRUCTURES_DEFINITION_ID

    def __new__(cls, obj: Any = None, **hints: Any) -> Any:
        resource = hints.get("resource", obj)
        if not isinstance(resource, OptimadeResource):
            return None
        if hints and hints.get("kind", cls.kind) != cls.kind:
            return None
        return super().__new__(cls)

    def __init__(self, obj: OptimadeResource | None = None, **hints: Any) -> None:
        resource = hints.get("resource", obj)
        if not isinstance(resource, OptimadeResource):
            raise TypeError("OptimadeStructure requires an OptimadeResource")
        if hints and hints.get("kind", self.kind) != self.kind:
            raise TypeError("OptimadeStructure kind must be 'optimade'")
        object.__setattr__(self, "resource", resource)

    def unwrap(self) -> OptimadeResource:
        """Return the exact authoritative source resource by identity."""

        return self.resource

    @property
    def raw(self) -> Mapping[str, object]:
        """The immutable JSON API resource envelope, decoded only on access."""

        return cast(Mapping[str, object], self.resource.unwrap())

    @cached_property
    def _local_schema(self) -> EntryTypeDefinition:
        return load_entry_type_definition(self.entry_type_definition_id)

    @cached_property
    def _remote_names_by_definition_id(self) -> Mapping[str, str]:
        root = optimade_document_root(self.resource.schema.info_document)
        data = root.get("data")
        if not isinstance(data, Mapping):
            raise IncompleteOptimadeResourceError("OPTIMADE structure schema has no object 'data' member")
        properties = data.get("properties")
        if not isinstance(properties, Mapping):
            raise IncompleteOptimadeResourceError("OPTIMADE structure schema has no object 'data.properties' member")
        names: dict[str, str] = {}
        for remote_name, document in properties.items():
            if not isinstance(remote_name, str) or not isinstance(document, Mapping):
                continue
            definition_id = document.get("$id")
            if not _is_definition_iri(definition_id):
                # A remote label is not a semantic identity.  Invalid or absent
                # IDs are unknown rather than inferred from their spelling.
                continue
            definition_id = cast(str, definition_id)
            previous = names.get(definition_id)
            if previous is not None and previous != remote_name:
                raise IncompleteOptimadeResourceError(
                    f"OPTIMADE structure schema assigns {definition_id!r} to both {previous!r} and {remote_name!r}"
                )
            names[definition_id] = remote_name
        return MappingProxyType(names)

    def _value(self, property_name: str, *, component: str, optional: bool = False) -> object:
        definition = self._local_schema.properties[property_name]
        definition_id = definition.definition_id
        remote_name = self._remote_names_by_definition_id.get(definition_id)
        if remote_name is None:
            if optional:
                return _MISSING
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE {component} requires semantic property {property_name!r} "
                f"({definition_id}), but the schema does not identify it"
            )
        attributes = self.raw.get("attributes")
        if not isinstance(attributes, Mapping):
            if optional and attributes is None:
                return _MISSING
            raise IncompleteOptimadeResourceError(f"OPTIMADE {component} requires an object 'attributes' member")
        value = attributes.get(remote_name, _MISSING)
        if value is _MISSING or value is None:
            if optional:
                return _MISSING
            state = "null" if value is None else "missing"
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE {component} has {state} semantic property {property_name!r}"
            )
        return value

    def _raw_optional(self, property_name: str) -> object:
        """Return a supplied semantic value, preserving missing/null separately."""

        definition = self._local_schema.properties[property_name]
        remote_name = self._remote_names_by_definition_id.get(definition.definition_id)
        if remote_name is None:
            return _MISSING
        attributes = self.raw.get("attributes")
        if not isinstance(attributes, Mapping):
            return _MISSING
        return attributes.get(remote_name, _MISSING)

    def _decoded_optional(self, property_name: str) -> object | None:
        """Decode one optional source property without borrowing its transport name."""

        raw = self._raw_optional(property_name)
        if raw is _MISSING or raw is None:
            return None
        definition = self._local_schema.properties[property_name]
        try:
            return decode_optimade_value(definition, raw)
        except (TypeError, ValueError) as exc:
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE semantic property {property_name!r} is invalid: {exc}"
            ) from exc

    @staticmethod
    def _number(value: object, *, property_name: str) -> Fraction:
        if not isinstance(value, Decimal | int) or isinstance(value, bool):
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE semantic property {property_name!r} must contain only JSON numbers"
            )
        return Fraction(value)

    @classmethod
    def _numeric_matrix(
        cls,
        value: object,
        *,
        property_name: str,
        rows: int | None = None,
        allow_null_rows: bool = False,
    ) -> tuple[tuple[Fraction, Fraction, Fraction] | None, ...]:
        if not isinstance(value, tuple | list) or (rows is not None and len(value) != rows):
            expected = f"exactly {rows}" if rows is not None else "any number of"
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE semantic property {property_name!r} must contain {expected} three-vectors"
            )
        result: list[tuple[Fraction, Fraction, Fraction] | None] = []
        for row in value:
            if allow_null_rows and row is None:
                result.append(None)
                continue
            if not isinstance(row, tuple | list) or len(row) != 3:
                raise IncompleteOptimadeResourceError(
                    f"OPTIMADE semantic property {property_name!r} must contain three-vectors"
                )
            if any(item is None for item in row):
                if allow_null_rows and all(item is None for item in row):
                    result.append(None)
                    continue
                raise IncompleteOptimadeResourceError(
                    f"OPTIMADE semantic property {property_name!r} has a partly null vector"
                )
            converted = tuple(cls._number(item, property_name=property_name) for item in row)
            result.append(cast(tuple[Fraction, Fraction, Fraction], converted))
        return tuple(result)

    def _require_unit_cell_projection(self, component: str) -> None:
        span = self.site_coordinate_span
        if span not in _UNIT_CELL_SPANS:
            description = f": {self.site_coordinate_span_description}" if self.site_coordinate_span_description else ""
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE {component} cannot project site_coordinate_span={span!r} as a native unit cell{description}"
            )

    def _portable_value(self, property_name: str) -> object | None:
        """Decode an optional portable-query field by its exact definition IRI."""

        definition = self._local_schema.properties[property_name]
        remote_name = self._remote_names_by_definition_id.get(definition.definition_id)
        if remote_name is None:
            return None
        attributes = self.raw.get("attributes")
        if not isinstance(attributes, Mapping):
            return None
        raw = attributes.get(remote_name, _MISSING)
        if raw is _MISSING or raw is None:
            return None
        try:
            return decode_optimade_value(definition, raw)
        except (TypeError, ValueError) as exc:
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE portable property {property_name!r} is invalid: {exc}"
            ) from exc

    @cached_property
    def _composition_from_sites(self) -> CompositionResult | None:
        """Project a complete supplied unit-cell site model without requiring a view."""

        raw_species = self._raw_optional("species")
        raw_names = self._raw_optional("species_at_sites")
        raw_span = self._raw_optional("site_coordinate_span")
        if (
            not isinstance(raw_species, tuple | list)
            or not isinstance(raw_names, tuple | list)
            or (raw_span not in (_MISSING, None) and raw_span not in _UNIT_CELL_SPANS)
        ):
            return None
        if not all(isinstance(name, str) for name in raw_names):
            return None
        proxy = SimpleNamespace(
            species=self._decoded_species,
            species_at_sites=tuple(raw_names),
            assemblies=self.assemblies,
            chemical_composition=None,
        )
        try:
            return project_composition(proxy)
        except (TypeError, ValueError) as exc:
            raise IncompleteOptimadeResourceError(f"OPTIMADE supplied site composition is invalid: {exc}") from exc

    @cached_property
    def composition(self) -> CompositionResult:
        """The source-backed composition, retaining implicit/source-only element ratios."""

        site_result = self._composition_from_sites
        features = self.structure_features or ()
        if site_result is not None and "implicit_atoms" not in features:
            return site_result
        elements = self.elements
        ratios = self.elements_ratios
        if elements is None or ratios is None or len(elements) != len(ratios):
            if site_result is not None:
                return site_result
            raise IncompleteOptimadeResourceError(
                "OPTIMADE composition requires consistent 'elements' and 'elements_ratios'"
            )
        raw_ratios = self._raw_optional("elements_ratios")
        raw_values = raw_ratios if isinstance(raw_ratios, tuple | list) else ()
        uncertainties = tuple(
            (
                element,
                decimal_precision(raw_values[index])
                if index < len(raw_values) and isinstance(raw_values[index], Decimal)
                else None,
            )
            for index, element in enumerate(elements)
        )
        exact = all(width is None for _, width in uncertainties)
        normalized, status, _, _ = normalization(ratios, tuple(width for _, width in uncertainties))
        return CompositionResult(
            tuple(zip(elements, ratios)),
            uncertainties,
            True,
            exact,
            normalized,
            status,
        )

    @stored_property
    def formula(self) -> str | None:
        """Source-declared reduced formula through the native convenience name."""

        return self.chemical_formula_reduced

    def _precision(self, name: str, *, component: str) -> Decimal | int | object:
        definition_id = precision_definitions()[name].definition_id
        remote_name = self._remote_names_by_definition_id.get(definition_id)
        if remote_name is None:
            return _MISSING
        attributes = self.raw.get("attributes")
        if not isinstance(attributes, Mapping):
            return _MISSING
        value = attributes.get(remote_name, _MISSING)
        if value is _MISSING or value is None:
            return _MISSING
        if not isinstance(value, Decimal | int) or isinstance(value, bool):
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE {component} has invalid semantic property {name!r}: expected a JSON number"
            )
        return value

    @staticmethod
    def _decimal_precision(value: object) -> Fraction | None:
        """Return the coarsest precision stated by Decimal leaves in *value*.

        JSON integers are exact protocol values here, not decimal measurement
        spellings.  Only :class:`Decimal` leaves retain a source decimal token
        (including its trailing zeros), so only they contribute a fallback.
        """

        values: list[Fraction] = []

        def visit(item: object) -> None:
            if isinstance(item, Decimal):
                precision = decimal_precision(item)
                if precision is not None:
                    values.append(precision)
            elif isinstance(item, Mapping):
                for nested in item.values():
                    visit(nested)
            elif isinstance(item, tuple | list):
                for nested in item:
                    visit(nested)

        visit(value)
        return combined_precision(values)

    def _fractional_cartesian_precision(self, cartesian: object) -> Fraction | None:
        """Convert Cartesian decimal-token precision to a conservative reduced bound.

        As for POSCAR Cartesian coordinates, divide by the shortest cell edge:
        it produces the largest fractional uncertainty and is conservative for
        the coordinate frame.  The existing atomistic boundary uses the same
        deterministic float rendering for non-rational edge lengths.
        """

        absolute = self._decimal_precision(cartesian)
        if absolute is None:
            return None
        shortest = min(length.to_float() for length in self.cell.lengths)
        if shortest <= 0:
            return None
        return absolute / Fraction(str(shortest)).limit_denominator(10**12)

    @stored_property
    def id(self) -> str:
        """The JSON API resource identifier (not inferred from a remote label)."""

        return self.resource.id

    @stored_property
    def type(self) -> str:
        """The JSON API resource type identifier (not inferred from a remote label)."""

        return self.resource.type

    @stored_property
    def immutable_id(self) -> str | None:
        return cast(str | None, self._portable_value("immutable_id"))

    @stored_property
    def last_modified(self) -> datetime.datetime | None:
        return cast(datetime.datetime | None, self._portable_value("last_modified"))

    @stored_property
    def elements(self) -> tuple[str, ...] | None:
        value = self._portable_value("elements")
        if value is None:
            return None
        if (
            not isinstance(value, tuple)
            or any(not isinstance(element, str) or element not in _ELEMENTS for element in value)
            or tuple(sorted(set(value))) != value
        ):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'elements' must contain unique real element symbols in alphabetical order"
            )
        ratios = self._portable_value("elements_ratios")
        if ratios is not None and (not isinstance(ratios, tuple) or len(ratios) != len(value)):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic properties 'elements' and 'elements_ratios' have different lengths"
            )
        nelements = self._portable_value("nelements")
        if nelements is not None and nelements != len(value):
            raise IncompleteOptimadeResourceError("OPTIMADE semantic property 'nelements' disagrees with 'elements'")
        projected = self._composition_from_sites
        features = self._portable_value("structure_features")
        if (
            projected is not None
            and projected.complete
            and (not isinstance(features, tuple) or "implicit_atoms" not in features)
            and projected.elements != value
        ):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'elements' disagrees with the supplied site composition"
            )
        return value

    @stored_property
    def nelements(self) -> int | None:
        value = self._portable_value("nelements")
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise IncompleteOptimadeResourceError("OPTIMADE semantic property 'nelements' must be non-negative")
        elements = self._portable_value("elements")
        ratios = self._portable_value("elements_ratios")
        if isinstance(elements, tuple) and len(elements) != value:
            raise IncompleteOptimadeResourceError("OPTIMADE semantic property 'nelements' disagrees with 'elements'")
        if isinstance(ratios, tuple) and len(ratios) != value:
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'nelements' disagrees with 'elements_ratios'"
            )
        return value

    @stored_property
    def elements_ratios(self) -> tuple[Fraction, ...] | None:
        value = self._portable_value("elements_ratios")
        if value is None:
            return None
        if not isinstance(value, tuple):
            raise IncompleteOptimadeResourceError("OPTIMADE portable property 'elements_ratios' is not a list")
        try:
            ratios = tuple(Fraction(ratio) for ratio in value)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE portable property 'elements_ratios' is invalid: {exc}"
            ) from exc
        width = Fraction()
        for ratio in value:
            width += decimal_precision(ratio) or Fraction()
        if any(ratio < 0 for ratio in ratios) or abs(sum(ratios, Fraction()) - 1) > width:
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'elements_ratios' must contain non-negative values summing to one"
            )
        elements = self._portable_value("elements")
        nelements = self._portable_value("nelements")
        if isinstance(elements, tuple) and len(elements) != len(ratios):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic properties 'elements_ratios' and 'elements' have different lengths"
            )
        if isinstance(nelements, int) and nelements != len(ratios):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'elements_ratios' disagrees with 'nelements'"
            )
        projected = self._composition_from_sites
        features = self._portable_value("structure_features")
        if projected is not None and (not isinstance(features, tuple) or "implicit_atoms" not in features):
            expected = projected.elements_ratios
            if len(expected) == len(ratios) and any(abs(left - right) > width for left, right in zip(expected, ratios)):
                raise IncompleteOptimadeResourceError(
                    "OPTIMADE semantic property 'elements_ratios' disagrees with the supplied site composition"
                )
        return ratios

    @staticmethod
    def _formula_tokens(value: str, property_name: str) -> tuple[tuple[str, int], ...]:
        if not value:
            raise IncompleteOptimadeResourceError(f"OPTIMADE semantic property {property_name!r} is empty")
        position = 0
        result: list[tuple[str, int]] = []
        while position < len(value):
            match = _FORMULA_TOKEN.match(value, position)
            if match is None or match.group(1) not in _ELEMENTS:
                raise IncompleteOptimadeResourceError(
                    f"OPTIMADE semantic property {property_name!r} has invalid chemical-formula syntax"
                )
            result.append((match.group(1), int(match.group(2) or 1)))
            position = match.end()
        return tuple(result)

    def _formula(self, property_name: str) -> str | None:
        value = self._portable_value(property_name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise IncompleteOptimadeResourceError(f"OPTIMADE semantic property {property_name!r} must be a string")
        if property_name == "chemical_formula_anonymous":
            if not value or "".join(match.group(0) for match in _ANONYMOUS_TOKEN.finditer(value)) != value:
                raise IncompleteOptimadeResourceError(
                    "OPTIMADE semantic property 'chemical_formula_anonymous' has invalid syntax"
                )
            anonymous_matches = tuple(_ANONYMOUS_TOKEN.finditer(value))
            labels = tuple(match.group(1) for match in anonymous_matches)
            coefficients = tuple(int(match.group(2) or 1) for match in anonymous_matches)
            if labels != tuple(anonymous_symbol(index) for index in range(len(labels))) or coefficients != tuple(
                sorted(coefficients, reverse=True)
            ):
                raise IncompleteOptimadeResourceError(
                    "OPTIMADE semantic property 'chemical_formula_anonymous' has invalid symbol or coefficient order"
                )
            ratios = self.elements_ratios
            if ratios is not None:
                total = sum(coefficients)
                formula_ratios = sorted((Fraction(item, total) for item in coefficients), reverse=True)
                stated_ratios = sorted(ratios, reverse=True)
                width = self._elements_ratio_width()
                if len(formula_ratios) != len(stated_ratios) or any(
                    abs(left - right) > width for left, right in zip(formula_ratios, stated_ratios)
                ):
                    raise IncompleteOptimadeResourceError(
                        "OPTIMADE semantic property 'chemical_formula_anonymous' disagrees with 'elements_ratios'"
                    )
            projected = self._composition_from_sites
            features = self._portable_value("structure_features")
            if (
                projected is not None
                and (not isinstance(features, tuple) or "implicit_atoms" not in features)
                and projected.chemical_formula_anonymous is not None
                and value != projected.chemical_formula_anonymous
            ):
                raise IncompleteOptimadeResourceError(
                    "OPTIMADE semantic property 'chemical_formula_anonymous' disagrees with the supplied site composition"
                )
            return value
        tokens = self._formula_tokens(value, property_name)
        if len({element for element, _ in tokens}) != len(tokens):
            raise IncompleteOptimadeResourceError(f"OPTIMADE semantic property {property_name!r} repeats an element")
        if property_name == "chemical_formula_hill":
            try:
                validate_hill_formula(value, None)
            except ValueError as exc:
                raise IncompleteOptimadeResourceError(
                    "OPTIMADE semantic property 'chemical_formula_hill' is not in Hill order"
                ) from exc
        if property_name == "chemical_formula_reduced" and tuple(element for element, _ in tokens) != tuple(
            sorted(element for element, _ in tokens)
        ):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'chemical_formula_reduced' is not in alphabetical order"
            )
        declared_elements = self.elements
        if declared_elements is not None and set(declared_elements) != {element for element, _ in tokens}:
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE semantic property {property_name!r} disagrees with 'elements'"
            )
        declared_ratios = self.elements_ratios
        if declared_elements is not None and declared_ratios is not None:
            counts = dict(tokens)
            total = sum(counts.values())
            named_formula_ratios = tuple(Fraction(counts[element], total) for element in declared_elements)
            width = self._elements_ratio_width()
            if any(abs(left - right) > width for left, right in zip(named_formula_ratios, declared_ratios)):
                raise IncompleteOptimadeResourceError(
                    f"OPTIMADE semantic property {property_name!r} disagrees with 'elements_ratios'"
                )
        projected = self._composition_from_sites
        features = self._portable_value("structure_features")
        if projected is not None and (not isinstance(features, tuple) or "implicit_atoms" not in features):
            expected_formula = (
                projected.chemical_formula_anonymous
                if property_name == "chemical_formula_anonymous"
                else projected.chemical_formula_reduced
            )
            if property_name in {"chemical_formula_reduced", "chemical_formula_anonymous"} and (
                expected_formula is not None and value != expected_formula
            ):
                raise IncompleteOptimadeResourceError(
                    f"OPTIMADE semantic property {property_name!r} disagrees with the supplied site composition"
                )
            if property_name == "chemical_formula_hill":
                try:
                    validate_hill_formula(value, projected)
                except ValueError as exc:
                    raise IncompleteOptimadeResourceError(
                        "OPTIMADE semantic property 'chemical_formula_hill' disagrees with the supplied site composition"
                    ) from exc
        return value

    def _elements_ratio_width(self) -> Fraction:
        raw = self._raw_optional("elements_ratios")
        if not isinstance(raw, tuple | list):
            return Fraction()
        return sum((decimal_precision(value) or Fraction() for value in raw), Fraction())

    @stored_property
    def chemical_formula_descriptive(self) -> str | None:
        value = self._portable_value("chemical_formula_descriptive")
        try:
            return validate_descriptive_formula(cast(str | None, value))
        except (TypeError, ValueError) as exc:
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE semantic property 'chemical_formula_descriptive' is invalid: {exc}"
            ) from exc

    @stored_property
    def chemical_formula_reduced(self) -> str | None:
        return self._formula("chemical_formula_reduced")

    @stored_property
    def chemical_formula_hill(self) -> str | None:
        return self._formula("chemical_formula_hill")

    @stored_property
    def chemical_formula_anonymous(self) -> str | None:
        return self._formula("chemical_formula_anonymous")

    @stored_property
    def dimension_types(self) -> tuple[int, ...] | None:
        value = self._portable_value("dimension_types")
        if value is None:
            return None
        if (
            not isinstance(value, tuple)
            or len(value) != 3
            or any(not isinstance(item, int) or isinstance(item, bool) or item not in (0, 1) for item in value)
        ):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'dimension_types' must contain exactly three 0/1 integers"
            )
        stated = self._portable_value("nperiodic_dimensions")
        if stated is not None and stated != sum(value):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'nperiodic_dimensions' disagrees with 'dimension_types'"
            )
        return cast(tuple[int, int, int], value)

    @stored_property
    def nperiodic_dimensions(self) -> int | None:
        value = self._portable_value("nperiodic_dimensions")
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 3:
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'nperiodic_dimensions' must be an integer in [0, 3]"
            )
        dimensions = self._portable_value("dimension_types")
        if isinstance(dimensions, tuple) and value != sum(cast(tuple[int, ...], dimensions)):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'nperiodic_dimensions' disagrees with 'dimension_types'"
            )
        return value

    @stored_property
    def nsites(self) -> int | None:
        value = self._portable_value("nsites")
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise IncompleteOptimadeResourceError("OPTIMADE semantic property 'nsites' must be non-negative")
        for property_name in ("species_at_sites", "cartesian_site_positions", "fractional_site_positions"):
            supplied = self._raw_optional(property_name)
            if (
                supplied is not _MISSING
                and supplied is not None
                and (not isinstance(supplied, tuple | list) or len(supplied) != value)
            ):
                raise IncompleteOptimadeResourceError(
                    f"OPTIMADE semantic property 'nsites' disagrees with {property_name!r}"
                )
        return value

    @stored_property
    def structure_features(self) -> tuple[str, ...] | None:
        value = self._portable_value("structure_features")
        if value is None:
            return None
        if (
            not isinstance(value, tuple)
            or any(not isinstance(item, str) or item not in _STRUCTURE_FEATURES for item in value)
            or tuple(sorted(set(value))) != value
        ):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'structure_features' must be a unique alphabetical list of standard flags"
            )
        raw_species = self._raw_optional("species")
        raw_assemblies = self._raw_optional("assemblies")
        derived: set[str] = set()
        if raw_assemblies is not _MISSING and raw_assemblies is not None:
            derived.add("assemblies")
        if isinstance(raw_species, tuple | list):
            for species in self._decoded_species:
                if len(species.chemical_symbols) > 1 or species.concentration != (Fraction(1),):
                    derived.add("disorder")
                if species.attached is not None:
                    derived.add("site_attachments")
        for flag in ("assemblies", "disorder", "site_attachments"):
            if (flag in value) != (flag in derived):
                raise IncompleteOptimadeResourceError(
                    f"OPTIMADE semantic property 'structure_features' has an inconsistent {flag!r} flag"
                )
        return value

    @cached_property
    def lattice_vectors(self) -> tuple[tuple[Fraction, Fraction, Fraction] | None, ...] | None:
        value = self._decoded_optional("lattice_vectors")
        if value is None:
            return None
        vectors = self._numeric_matrix(value, property_name="lattice_vectors", rows=3, allow_null_rows=True)
        dimensions = self.dimension_types
        if dimensions is not None:
            for index, vector in enumerate(vectors):
                if vector is None and dimensions[index] != 0:
                    raise IncompleteOptimadeResourceError(
                        "OPTIMADE semantic property 'lattice_vectors' nulls a periodic lattice vector"
                    )
        return vectors

    @cached_property
    def _coordinate_arrays(
        self,
    ) -> tuple[
        tuple[tuple[Fraction, Fraction, Fraction], ...] | None,
        tuple[tuple[Fraction, Fraction, Fraction], ...] | None,
    ]:
        raw_fractional = self._decoded_optional("fractional_site_positions")
        raw_cartesian = self._decoded_optional("cartesian_site_positions")
        fractional = (
            None
            if raw_fractional is None
            else cast(
                tuple[tuple[Fraction, Fraction, Fraction], ...],
                self._numeric_matrix(raw_fractional, property_name="fractional_site_positions"),
            )
        )
        cartesian = (
            None
            if raw_cartesian is None
            else cast(
                tuple[tuple[Fraction, Fraction, Fraction], ...],
                self._numeric_matrix(raw_cartesian, property_name="cartesian_site_positions"),
            )
        )
        if fractional is not None and cartesian is not None:
            if len(fractional) != len(cartesian):
                raise IncompleteOptimadeResourceError(
                    "OPTIMADE fractional and Cartesian site-position arrays have different lengths"
                )
            lattice = self.lattice_vectors
            if lattice is not None and all(vector is not None for vector in lattice):
                expected = SurdVector.create(fractional) * SurdVector.create(lattice)
                actual = SurdVector.create(cartesian)
                difference = (expected - actual).to_fractions_approx(Fraction(1, 10**24))
                cartesian_width = self._decimal_precision(self._raw_optional("cartesian_site_positions")) or Fraction()
                fractional_width = (
                    self._decimal_precision(self._raw_optional("fractional_site_positions")) or Fraction()
                )
                lattice_width = self._decimal_precision(self._raw_optional("lattice_vectors")) or Fraction()
                scale = max(abs(value) for row in cast(tuple[tuple[Fraction, ...], ...], lattice) for value in row)
                tolerance = cartesian_width + 3 * (fractional_width * scale + lattice_width)
                if any(abs(value) > tolerance for row in difference for value in row):
                    raise IncompleteOptimadeResourceError(
                        "OPTIMADE fractional_site_positions and cartesian_site_positions disagree with lattice_vectors"
                    )
        expected_nsites = self._portable_value("nsites")
        species_names = self._raw_optional("species_at_sites")
        for name, positions in (
            ("fractional_site_positions", fractional),
            ("cartesian_site_positions", cartesian),
        ):
            if positions is None:
                continue
            if isinstance(expected_nsites, int) and len(positions) != expected_nsites:
                raise IncompleteOptimadeResourceError(f"OPTIMADE semantic property {name!r} disagrees with 'nsites'")
            if isinstance(species_names, tuple | list) and len(positions) != len(species_names):
                raise IncompleteOptimadeResourceError(
                    f"OPTIMADE semantic property {name!r} disagrees with 'species_at_sites'"
                )
        return fractional, cartesian

    @cached_property
    def fractional_site_positions(self) -> tuple[tuple[Fraction, Fraction, Fraction], ...] | None:
        return self._coordinate_arrays[0]

    @cached_property
    def cartesian_site_positions(self) -> tuple[tuple[Fraction, Fraction, Fraction], ...] | None:
        return self._coordinate_arrays[1]

    @stored_property
    def site_coordinate_span(self) -> str:
        value = self._decoded_optional("site_coordinate_span")
        if value is None:
            return "unit_cell"
        if not isinstance(value, str) or value not in _COORDINATE_SPANS:
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'site_coordinate_span' has an unrecognized representation"
            )
        if (
            value
            in {
                "fundamental_domain",
                "asymmetric_unit",
                "molecular_fundamental_domain",
                "molecular_asymmetric_unit",
            }
            and not self.space_group_symmetry_operations_xyz
        ):
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE site_coordinate_span={value!r} requires space-group symmetry operations"
            )
        return value

    @stored_property
    def site_coordinate_span_description(self) -> str | None:
        value = self._decoded_optional("site_coordinate_span_description")
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'site_coordinate_span_description' must be a non-empty string"
            )
        if self.site_coordinate_span != "other":
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'site_coordinate_span_description' is only valid for span 'other'"
            )
        return value

    @stored_property
    def molecular(self) -> bool:
        """Whether the native unit-cell projection carries molecular placement."""

        return self.site_coordinate_span == "molecular_unit_cell"

    @stored_property
    def coordinate_precision(self) -> Fraction | None:
        value = self._precision("_httk_coordinate_precision", component="sites")
        if value is not _MISSING:
            return Fraction(cast(Decimal | int, value))
        fractional = self._raw_optional("fractional_site_positions")
        return None if fractional is _MISSING else self._decimal_precision(fractional)

    @stored_property
    def basis_precision(self) -> Fraction | None:
        value = self._precision("_httk_basis_precision", component="cell")
        if value is not _MISSING:
            return Fraction(cast(Decimal | int, value))
        lattice = self._raw_optional("lattice_vectors")
        return None if lattice is _MISSING else self._decimal_precision(lattice)

    @cached_property
    def symmetry(self) -> StructureSymmetry:
        """Typed source symmetry metadata used by the common unit-cell view layer."""

        try:
            return StructureSymmetry(
                space_group_it_number=self._space_group_it_number_value(),
                space_group_symbol_hall=self._symmetry_string_value("space_group_symbol_hall"),
                space_group_symbol_hermann_mauguin=self._symmetry_string_value("space_group_symbol_hermann_mauguin"),
                space_group_symbol_hermann_mauguin_extended=self._symmetry_string_value(
                    "space_group_symbol_hermann_mauguin_extended"
                ),
                space_group_symmetry_operations_xyz=self._space_group_operations_value(),
                wyckoff_positions=self._wyckoff_positions_value(),
            )
        except ValueError as exc:
            message = str(exc)
            if message == "supplied space-group number and symbols are inconsistent":
                message = "OPTIMADE supplied space-group number and symbols are mutually inconsistent"
            elif message == "supplied space-group operations disagree with its number or symbols":
                message = "OPTIMADE supplied space-group operations disagree with its number or symbols"
            elif message == "supplied Wyckoff positions disagree with the space-group setting":
                message = "OPTIMADE semantic property 'wyckoff_positions' disagrees with the supplied space group"
            elif message.startswith("space-group symmetry operations "):
                message = f"OPTIMADE supplied space-group operations do not form a valid group: {message}"
            raise IncompleteOptimadeResourceError(message) from exc

    @stored_property
    def optimization_type(self) -> str | None:
        value = self._decoded_optional("optimization_type")
        if value is None:
            return None
        if not isinstance(value, str):
            raise IncompleteOptimadeResourceError("OPTIMADE semantic property 'optimization_type' must be a string")
        if value in _OPTIMIZATION_TYPES:
            return value
        # The specification tells clients to interpret an unrecognized string as
        # "other".  Keep the source spelling available through ``raw``.
        return "other"

    @cached_property
    def assemblies(self) -> tuple[Assembly, ...] | None:
        raw = self._raw_optional("assemblies")
        if raw is _MISSING or raw is None:
            return None
        if not isinstance(raw, tuple | list) or not all(isinstance(value, Mapping) for value in raw):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'assemblies' must be a list of dictionaries"
            )
        values: list[Assembly] = []
        try:
            for item in raw:
                item = cast(Mapping[str, object], item)
                if set(item) != {"sites_in_groups", "group_probabilities"} and any(
                    not str(name).startswith("_") for name in set(item) - {"sites_in_groups", "group_probabilities"}
                ):
                    raise ValueError("assembly dictionaries contain an unknown non-namespaced key")
                groups = item.get("sites_in_groups")
                probabilities = item.get("group_probabilities")
                if not isinstance(groups, tuple | list) or not isinstance(probabilities, tuple | list):
                    raise ValueError("assembly dictionaries require sites_in_groups and group_probabilities lists")
                values.append(Assembly(tuple(tuple(group) for group in groups), tuple(probabilities)))
            nsites = self._portable_value("nsites")
            if nsites is None:
                species_names = self._raw_optional("species_at_sites")
                nsites = len(species_names) if isinstance(species_names, tuple | list) else None
            result = validate_assemblies(values, cast(int | None, nsites))
        except (TypeError, ValueError) as exc:
            raise IncompleteOptimadeResourceError(f"OPTIMADE semantic property 'assemblies' is invalid: {exc}") from exc
        features = self._portable_value("structure_features")
        if isinstance(features, tuple) and "assemblies" not in features:
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'assemblies' is present without its structure_features flag"
            )
        return result

    @cached_property
    def _declared_spacegroup_candidates(self) -> tuple[Spacegroup, ...]:
        return tuple(Spacegroup(record) for record in self.symmetry.matched_settings)

    def _declared_periodic_dimensions(self) -> int | None:
        periodic = self._portable_value("nperiodic_dimensions")
        if isinstance(periodic, int) and not isinstance(periodic, bool):
            return periodic
        dimensions = self._portable_value("dimension_types")
        if isinstance(dimensions, tuple) and len(dimensions) == 3 and all(value in (0, 1) for value in dimensions):
            return sum(cast(tuple[int, int, int], dimensions))
        return None

    def _symmetry_string_value(self, property_name: str) -> str | None:
        value = self._decoded_optional(property_name)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE semantic property {property_name!r} must be a non-empty string"
            )
        periodic = self._declared_periodic_dimensions()
        if isinstance(periodic, int) and periodic != 3:
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE semantic property {property_name!r} must be null unless nperiodic_dimensions is 3"
            )
        return value

    def _symmetry_string(self, property_name: str) -> str | None:
        value = self._symmetry_string_value(property_name)
        _ = self.symmetry
        return value

    def _space_group_it_number_value(self) -> int | None:
        value = self._decoded_optional("space_group_it_number")
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 230:
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'space_group_it_number' must be an integer in [1, 230]"
            )
        periodic = self._declared_periodic_dimensions()
        if isinstance(periodic, int) and periodic != 3:
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'space_group_it_number' must be null unless nperiodic_dimensions is 3"
            )
        return value

    def _space_group_operations_value(self) -> tuple[str, ...] | None:
        value = self._decoded_optional("space_group_symmetry_operations_xyz")
        if value is None:
            return None
        if (
            not isinstance(value, tuple)
            or not value
            or any(
                not isinstance(operation, str)
                or len(operation.split(",")) != 3
                or not all(_SYMOP_COORDINATE.fullmatch(part) for part in operation.split(","))
                for operation in value
            )
        ):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'space_group_symmetry_operations_xyz' must contain three-coordinate strings"
            )
        normalized = tuple(operation.replace(" ", "") for operation in value)
        if "x,y,z" not in normalized:
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'space_group_symmetry_operations_xyz' lacks the identity operation"
            )
        periodic = self._declared_periodic_dimensions()
        if periodic == 0:
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'space_group_symmetry_operations_xyz' must be null for a nonperiodic structure"
            )
        return value

    def _wyckoff_positions_value(self) -> tuple[str, ...] | None:
        value = self._decoded_optional("wyckoff_positions")
        if value is None:
            return None
        if not isinstance(value, tuple) or any(item not in _WYCKOFF for item in value):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'wyckoff_positions' must contain single valid Wyckoff letters"
            )
        for property_name in ("species_at_sites", "fractional_site_positions", "cartesian_site_positions"):
            supplied = self._raw_optional(property_name)
            if isinstance(supplied, tuple | list) and len(supplied) != len(value):
                raise IncompleteOptimadeResourceError(
                    f"OPTIMADE semantic property 'wyckoff_positions' disagrees with {property_name!r}"
                )
        nsites = self._portable_value("nsites")
        if isinstance(nsites, int) and nsites != len(value):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'wyckoff_positions' disagrees with 'nsites'"
            )
        return value

    @stored_property
    def space_group_symbol_hall(self) -> str | None:
        return self._symmetry_string("space_group_symbol_hall")

    @stored_property
    def space_group_symbol_hermann_mauguin(self) -> str | None:
        return self._symmetry_string("space_group_symbol_hermann_mauguin")

    @stored_property
    def space_group_symbol_hermann_mauguin_extended(self) -> str | None:
        return self._symmetry_string("space_group_symbol_hermann_mauguin_extended")

    @stored_property
    def space_group_it_number(self) -> int | None:
        value = self._space_group_it_number_value()
        _ = self.symmetry
        return value

    @stored_property
    def space_group_symmetry_operations_xyz(self) -> tuple[str, ...] | None:
        value = self._space_group_operations_value()
        _ = self.symmetry
        return value

    @stored_property
    def wyckoff_positions(self) -> tuple[str, ...] | None:
        value = self._wyckoff_positions_value()
        _ = self.symmetry
        return value

    @cached_property
    def _cell(self) -> Cell:
        self._require_unit_cell_projection("cell")
        lattice = self._value("lattice_vectors", component="cell")
        dimensions = self.dimension_types
        if dimensions is None:
            raise IncompleteOptimadeResourceError("OPTIMADE cell has missing or null 'dimension_types'")
        vectors = self.lattice_vectors
        if vectors is None or any(vector is None for vector in vectors):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE cell cannot project null non-periodic lattice vectors into the native coordinate frame"
            )
        precision = self._precision("_httk_basis_precision", component="cell")
        if precision is _MISSING:
            precision = self._decimal_precision(lattice)
        try:
            return Cell(
                cast(Any, lattice),
                precision=None if precision is _MISSING else precision,
                periodicity=tuple(bool(value) for value in dimensions),
            )
        except (TypeError, ValueError) as exc:
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE cell has invalid semantic property 'lattice_vectors': {exc}"
            ) from exc

    @property
    def cell(self) -> Cell:
        return self._cell

    @cached_property
    def _sites(self) -> Sites:
        self._require_unit_cell_projection("sites")
        fractional, cartesian = self._coordinate_arrays
        precision = self._precision("_httk_coordinate_precision", component="sites")
        if fractional is not None:
            if precision is _MISSING:
                precision = self._decimal_precision(self._raw_optional("fractional_site_positions"))
            try:
                return Sites(cast(Any, fractional), precision=None if precision is _MISSING else precision)
            except (TypeError, ValueError) as exc:
                raise IncompleteOptimadeResourceError(
                    f"OPTIMADE sites has invalid semantic property 'fractional_site_positions': {exc}"
                ) from exc

        if cartesian is None:
            raise IncompleteOptimadeResourceError(
                "OPTIMADE sites requires 'fractional_site_positions' or 'cartesian_site_positions'"
            )
        if precision is _MISSING:
            precision = self._fractional_cartesian_precision(self._raw_optional("cartesian_site_positions"))
        try:
            reduced = SurdVector.create(cast(Any, cartesian)) * self.cell.basis.inv()
            return Sites(reduced, precision=None if precision is _MISSING else precision)
        except (TypeError, ValueError) as exc:
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE sites has invalid semantic property 'cartesian_site_positions': {exc}"
            ) from exc

    @property
    def sites(self) -> Sites:
        return self._sites

    @cached_property
    def _decoded_species(self) -> tuple[Species, ...]:
        raw_species = self._value("species", component="species")
        if not isinstance(raw_species, tuple | list):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE species has invalid semantic property 'species': expected a JSON array"
            )
        if not all(isinstance(value, Mapping) for value in raw_species):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE species has invalid semantic property 'species': expected objects"
            )
        try:
            values = tuple(Species.create(dict(value)) for value in raw_species)
        except (TypeError, ValueError, KeyError) as exc:
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE species has invalid semantic property 'species': {exc}"
            ) from exc
        names = tuple(value.name for value in values)
        if len(set(names)) != len(names):
            raise IncompleteOptimadeResourceError("OPTIMADE semantic property 'species' has duplicate names")
        raw_names = self._raw_optional("species_at_sites")
        if isinstance(raw_names, tuple | list):
            unknown = sorted(set(raw_names) - set(names))
            if unknown:
                raise IncompleteOptimadeResourceError(
                    f"OPTIMADE semantic property 'species_at_sites' references undefined species {unknown!r}"
                )
        return values

    @property
    def species(self) -> tuple[Species, ...]:
        return self._decoded_species

    @cached_property
    def _species_at_sites(self) -> tuple[str, ...]:
        raw_names = self._value("species_at_sites", component="species_at_sites")
        if not isinstance(raw_names, tuple | list) or not all(isinstance(name, str) for name in raw_names):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE species_at_sites has invalid semantic property 'species_at_sites': expected strings"
            )
        names = tuple(raw_names)
        for property_name in ("fractional_site_positions", "cartesian_site_positions"):
            positions = self._raw_optional(property_name)
            if isinstance(positions, tuple | list) and len(names) != len(positions):
                raise IncompleteOptimadeResourceError(
                    f"OPTIMADE semantic property 'species_at_sites' disagrees with {property_name!r}"
                )
        stated_nsites = self._portable_value("nsites")
        if isinstance(stated_nsites, int) and len(names) != stated_nsites:
            raise IncompleteOptimadeResourceError(
                "OPTIMADE semantic property 'species_at_sites' disagrees with 'nsites'"
            )
        defined = {value.name for value in self._decoded_species}
        unknown = sorted(set(names) - defined)
        if unknown:
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE semantic property 'species_at_sites' references undefined species {unknown!r}"
            )
        return names

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return self._species_at_sites
