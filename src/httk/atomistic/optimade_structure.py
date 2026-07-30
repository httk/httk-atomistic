"""Lazy, exact OPTIMADE-backed structure representation.

The transport spelling of an OPTIMADE property is deliberately never part of
the conversion contract.  The accompanying ``/info/structures`` snapshot maps
each spelling to a property-definition IRI; only that IRI selects a local
meaning.
"""

import datetime
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import cached_property
from types import MappingProxyType
from typing import Any, ClassVar, cast
from urllib.parse import urlsplit

from httk.core import (
    EntryTypeDefinition,
    IncompleteOptimadeResourceError,
    OptimadeResource,
    SurdVector,
    combined_precision,
    decimal_precision,
    decode_optimade_value,
    load_entry_type_schema,
    optimade_document_root,
    stored_property,
)

from .cell import Cell
from .precision_entries import precision_definitions
from .sites import Sites
from .species import Species
from .structure_backend import StructureBackend

_STRUCTURES_DEFINITION_ID = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"
_MISSING = object()


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
    """A structure backend whose exact source is an :class:`~httk.core.OptimadeResource`.

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
        return load_entry_type_schema(self.entry_type_definition_id)

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
        return cast(tuple[str, ...] | None, self._portable_value("elements"))

    @stored_property
    def nelements(self) -> int | None:
        return cast(int | None, self._portable_value("nelements"))

    @stored_property
    def elements_ratios(self) -> tuple[Fraction, ...] | None:
        value = self._portable_value("elements_ratios")
        if value is None:
            return None
        if not isinstance(value, tuple):
            raise IncompleteOptimadeResourceError("OPTIMADE portable property 'elements_ratios' is not a list")
        try:
            return tuple(Fraction(ratio) for ratio in value)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE portable property 'elements_ratios' is invalid: {exc}"
            ) from exc

    @stored_property
    def chemical_formula_descriptive(self) -> str | None:
        return cast(str | None, self._portable_value("chemical_formula_descriptive"))

    @stored_property
    def chemical_formula_reduced(self) -> str | None:
        return cast(str | None, self._portable_value("chemical_formula_reduced"))

    @stored_property
    def chemical_formula_anonymous(self) -> str | None:
        return cast(str | None, self._portable_value("chemical_formula_anonymous"))

    @stored_property
    def nperiodic_dimensions(self) -> int | None:
        return cast(int | None, self._portable_value("nperiodic_dimensions"))

    @stored_property
    def nsites(self) -> int | None:
        return cast(int | None, self._portable_value("nsites"))

    @stored_property
    def structure_features(self) -> tuple[str, ...] | None:
        return cast(tuple[str, ...] | None, self._portable_value("structure_features"))

    @cached_property
    def _cell(self) -> Cell:
        lattice = self._value("lattice_vectors", component="cell")
        dimensions = self._value("dimension_types", component="cell")
        if not isinstance(dimensions, tuple | list) or len(dimensions) != 3:
            raise IncompleteOptimadeResourceError(
                "OPTIMADE cell has invalid semantic property 'dimension_types': expected three 0/1 values"
            )
        if any(not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1) for value in dimensions):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE cell has invalid semantic property 'dimension_types': expected three 0/1 values"
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
        fractional = self._value("fractional_site_positions", component="sites", optional=True)
        precision = self._precision("_httk_coordinate_precision", component="sites")
        if fractional is not _MISSING:
            if precision is _MISSING:
                precision = self._decimal_precision(fractional)
            try:
                return Sites(cast(Any, fractional), precision=None if precision is _MISSING else precision)
            except (TypeError, ValueError) as exc:
                raise IncompleteOptimadeResourceError(
                    f"OPTIMADE sites has invalid semantic property 'fractional_site_positions': {exc}"
                ) from exc

        cartesian = self._value("cartesian_site_positions", component="sites")
        if precision is _MISSING:
            precision = self._fractional_cartesian_precision(cartesian)
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
    def _species(self) -> tuple[Species, ...]:
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
            return tuple(Species.create(dict(value)) for value in raw_species)
        except (TypeError, ValueError, KeyError) as exc:
            raise IncompleteOptimadeResourceError(
                f"OPTIMADE species has invalid semantic property 'species': {exc}"
            ) from exc

    @property
    def species(self) -> tuple[Species, ...]:
        return self._species

    @cached_property
    def _species_at_sites(self) -> tuple[str, ...]:
        raw_names = self._value("species_at_sites", component="species_at_sites")
        if not isinstance(raw_names, tuple | list) or not all(isinstance(name, str) for name in raw_names):
            raise IncompleteOptimadeResourceError(
                "OPTIMADE species_at_sites has invalid semantic property 'species_at_sites': expected strings"
            )
        return tuple(raw_names)

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return self._species_at_sites
