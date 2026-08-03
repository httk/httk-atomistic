"""Project atomistic structures onto the neutral OPTIMADE entry-provider contract."""

import io
import os
import urllib.request
from collections.abc import Iterable, Mapping
from typing import Any, Self

import httk.core
from httk.core import (
    EntryProvider,
    EntryTypeDefinition,
    PropertyDefinition,
    load_entry_type_definition,
)
from httk.core.datastream import BytestreamBackend, BytestreamView, TextstreamBackend, TextstreamView
from httk.core.optimade import IncompleteOptimadeResourceError

from ._optimade_payloads import assemblies_payload, species_payload
from .asu_structure import FundamentalDomainStructure
from .precision_entries import PRECISION_PROPERTY_KEYS, precision_definitions, precision_properties
from .structure_like import StructureLike
from .symmetry_entries import (
    SETTING_PROPERTY_KEYS,
    SYMMETRY_PROPERTY_KEYS,
    setting_definitions,
    symmetry_properties,
)
from .unitcell_structure import UnitcellStructure

__all__ = ["StructureEntry", "StructureEntryProvider"]


def _structures_definition() -> EntryTypeDefinition:
    """Return the vendored OPTIMADE v1.3 ``structures`` definition."""
    return load_entry_type_definition("https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures")


# All four common entry properties and all 26 structure-specific properties in
# OPTIMADE v1.3 section 8.2. Keeping this explicit makes omissions at the provider
# boundary visible in review and tests.
_STANDARD_PROPERTY_NAMES: tuple[str, ...] = (
    "id",
    "type",
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
_STANDARD_PROPERTY_SET = frozenset(_STANDARD_PROPERTY_NAMES)
_STANDARD_STRUCTURE_NAMES = _STANDARD_PROPERTY_NAMES[4:]


class StructureEntry:
    """Non-instantiable logical family for OPTIMADE structure entries."""

    type = "structures"
    definition_id = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        raise TypeError("StructureEntry is a logical entry family; store a structure representation directly")

    @classmethod
    def entry_type_definition(cls) -> EntryTypeDefinition:
        """Return the standard structure definition extended by atomistic properties."""
        return _structures_definition().extended(setting_definitions()).extended(precision_definitions())


def _as_structure(obj: StructureLike) -> Any:
    """Normalize accepted convenience inputs to the common structure property layer."""
    from .structure_backend import StructureBackend
    from .structure_record import (
        ASUStructureRecord,
        FundamentalDomainStructureRecord,
        UnitcellStructureRecord,
        _domain_structure_from_record,
    )
    from .unitcell_structure_view import UnitcellStructureView

    datastreamish = isinstance(
        obj,
        (
            httk.core.DatastreamURL,
            str,
            os.PathLike,
            urllib.request.Request,
            bytes,
            bytearray,
            io.IOBase,
            TextstreamBackend,
            TextstreamView,
            BytestreamBackend,
            BytestreamView,
        ),
    ) or callable(getattr(obj, "resolve", None))
    if datastreamish:
        backend = obj if isinstance(obj, StructureBackend) else StructureBackend.create(obj)
        resolver = getattr(backend, "resolve", None)
        return resolver() if callable(resolver) else backend

    if isinstance(obj, UnitcellStructureRecord):
        return UnitcellStructureView(obj)
    if isinstance(obj, (FundamentalDomainStructureRecord, ASUStructureRecord)):
        obj = _domain_structure_from_record(obj)
    if isinstance(obj, FundamentalDomainStructure):
        # Representation is semantic: do not expand a fundamental domain merely because
        # it is being served. Its composition already accounts for orbit multiplicities.
        return obj
    if isinstance(obj, (tuple, list)):
        return UnitcellStructure(*tuple(obj))
    return obj


def _structure_projection(structure: Any) -> dict[str, Any]:
    """Return every standard structure property as a JSON-compatible value."""

    def numeric_matrix(value: Any) -> Any:
        if value is None:
            return None
        return [None if row is None else [float(item) for item in row] for row in value]

    composition = structure.composition
    composition_values = (
        {
            "elements": list(composition.elements),
            "nelements": composition.nelements,
            # Preserve a source backend's accepted central values at the response
            # boundary.  ``composition.elements_ratios`` is normalized for formula
            # derivation and exact query comparison, but normalizing an imprecise
            # remote OPTIMADE value here would silently rewrite its measurement.
            "elements_ratios": [float(value) for value in structure.elements_ratios],
            "chemical_formula_reduced": composition.chemical_formula_reduced,
            "chemical_formula_anonymous": composition.chemical_formula_anonymous,
        }
        if composition.complete
        else {
            "elements": None,
            "nelements": None,
            "elements_ratios": None,
            "chemical_formula_reduced": None,
            "chemical_formula_anonymous": None,
        }
    )

    values: dict[str, Any] = {
        **composition_values,
        "chemical_formula_descriptive": structure.chemical_formula_descriptive,
        "chemical_formula_hill": structure.chemical_formula_hill,
        "dimension_types": None if structure.dimension_types is None else list(structure.dimension_types),
        "nperiodic_dimensions": structure.nperiodic_dimensions,
        "lattice_vectors": numeric_matrix(structure.lattice_vectors),
        "cartesian_site_positions": numeric_matrix(structure.cartesian_site_positions),
        "site_coordinate_span": structure.site_coordinate_span,
        "site_coordinate_span_description": structure.site_coordinate_span_description,
        "nsites": len(structure.domain_sites)
        if isinstance(structure, FundamentalDomainStructure)
        else structure.nsites,
        "species_at_sites": (
            list(structure.domain_species_at_sites)
            if isinstance(structure, FundamentalDomainStructure)
            else list(structure.species_at_sites)
        ),
        "species": [species_payload(value) for value in structure.species],
        "assemblies": assemblies_payload(structure.assemblies),
        "structure_features": list(structure.structure_features),
        "optimization_type": structure.optimization_type,
    }

    # Standard symmetry is part of the common property layer. In particular, direct
    # projection preserves fundamental-domain/ASU site counts, coordinates and Wyckoff
    # positions instead of silently expanding them to a unit cell.
    for name in SYMMETRY_PROPERTY_KEYS:
        value = getattr(structure, name)
        if name == "fractional_site_positions":
            value = numeric_matrix(value)
        values[name] = list(value) if isinstance(value, tuple) else value

    # ``fractional_site_positions`` is included by the symmetry projection above. Assert
    # completeness here so adding a standard property cannot silently create a sparse row.
    missing: set[str] = set(_STANDARD_STRUCTURE_NAMES)
    missing.difference_update(values)
    if missing:  # pragma: no cover - a maintenance assertion
        raise AssertionError(f"incomplete OPTIMADE structure projection: {sorted(missing)!r}")
    return values


class StructureEntryProvider(EntryProvider):
    """Serve complete OPTIMADE v1.3 structure records from atomistic structures.

    A mapping keeps the convenient ``{"example": structure}`` form and its explicit
    served ids. An iterable of natural structures uses each representation's structural
    content id. Entry metadata lives on the structures themselves.

    Custom properties may extend the schema, but standard OPTIMADE fields are a pure
    projection of the entry and structure and cannot be replaced by custom values.
    """

    def __init__(
        self,
        entries: Mapping[str, StructureLike | None] | Iterable[StructureLike],
        *,
        extra_definitions: Mapping[str, PropertyDefinition] | None = None,
        properties: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        normalized: dict[str, StructureLike | None] = {}
        if isinstance(entries, Mapping):
            for raw_id, value in entries.items():
                entry_id = str(raw_id)
                if not entry_id:
                    raise ValueError("StructureEntryProvider ids must be non-empty strings")
                normalized[entry_id] = value
        else:
            for structure in entries:
                candidate_id = getattr(structure, "id", None)
                if not isinstance(candidate_id, str) or not candidate_id:
                    raise TypeError("iterable StructureEntryProvider input must contain structures with an id")
                entry_id = candidate_id
                if entry_id in normalized:
                    raise ValueError(f"duplicate structure id: {entry_id!r}")
                normalized[entry_id] = structure
        self._entries = normalized

        self._extra_definitions = dict(extra_definitions or {})
        definition_clashes = sorted(_STANDARD_PROPERTY_SET.intersection(self._extra_definitions))
        if definition_clashes:
            raise ValueError(
                "custom definitions may not override standard OPTIMADE structure properties: "
                + ", ".join(definition_clashes)
            )

        self._properties = {str(entry_id): dict(values) for entry_id, values in (properties or {}).items()}
        used_names = sorted({name for values in self._properties.values() for name in values})
        value_clashes = sorted(_STANDARD_PROPERTY_SET.intersection(used_names))
        if value_clashes:
            raise ValueError(
                "custom values may not override standard OPTIMADE structure properties: " + ", ".join(value_clashes)
            )
        described = self._definition().properties
        offenders = [name for name in used_names if name not in described]
        if offenders:
            raise ValueError(
                "StructureEntryProvider was given properties not described by its (extended) definition: "
                + ", ".join(offenders)
                + ". Add them via extra_definitions (custom names need a registered prefix)."
            )
        self._property_names = used_names

    def _definition(self) -> EntryTypeDefinition:
        definition = StructureEntry.entry_type_definition()
        if self._extra_definitions:
            definition = definition.extended(self._extra_definitions)
        return definition

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {"structures": self._definition()}

    def property_keys(self, entry_type: str) -> Mapping[str, str]:
        if entry_type != "structures":
            raise KeyError("StructureEntryProvider serves only the 'structures' entry type.")
        property_keys = {name: ("__id" if name == "id" else name) for name in _STANDARD_PROPERTY_NAMES}
        property_keys.update({name: name for name in SETTING_PROPERTY_KEYS})
        property_keys.update({name: name for name in PRECISION_PROPERTY_KEYS})
        property_keys.update({name: name for name in self._property_names})
        return property_keys

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        if entry_type != "structures":
            raise KeyError("StructureEntryProvider serves only the 'structures' entry type.")
        records: list[dict[str, Any]] = []
        for entry_id, value in self._entries.items():
            structure = None if value is None else _as_structure(value)
            record: dict[str, Any] = {
                "__id": entry_id,
                "type": StructureEntry.type,
                "immutable_id": None if structure is None else structure.immutable_id,
                "last_modified": (
                    None
                    if structure is None or structure.last_modified is None
                    else structure.last_modified.isoformat()
                ),
            }
            if structure is None:
                record.update({name: None for name in _STANDARD_STRUCTURE_NAMES})
                record.update(symmetry_properties(None))
                record.update(precision_properties(None))
            else:
                record.update(_structure_projection(structure))
                # Provider-specific setting/precision fields remain extensions to the
                # standard projection and preserve their existing null behavior.
                record.update({name: value for name, value in symmetry_properties(structure).items() if name[0] == "_"})
                try:
                    record.update(precision_properties(structure))
                except IncompleteOptimadeResourceError:
                    if structure.site_coordinate_span in {"unit_cell", "molecular_unit_cell"}:
                        raise
                    record.update({name: None for name in PRECISION_PROPERTY_KEYS})
            entry_properties = self._properties.get(entry_id, {})
            for name in self._property_names:
                record[name] = entry_properties.get(name)
            records.append(record)
        return records
