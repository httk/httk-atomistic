"""An :class:`~httk.core.EntryProvider` serving OPTIMADE ``structures`` from httk-atomistic.

:class:`StructureEntryProvider` maps :class:`~httk.atomistic.Structure` objects to
the neutral httk-core entry-provider contract, so a serving module (such as
*httk-optimade*) can expose them as an OPTIMADE ``structures`` endpoint without
this module depending on the serving module. The served entry type is described
by the vendored OPTIMADE standard ``structures`` definition (loaded via
:func:`httk.core.load_entry_type_definition`).

Beyond the core structural fields, the provider auto-derives the standard
composition fields (``nperiodic_dimensions``, ``dimension_types``,
``elements_ratios``, ``chemical_formula_reduced`` / ``_anonymous`` /
``_descriptive``) for ordered structures, may serve custom database-specific
properties layered on via an
:meth:`~httk.core.EntryTypeDefinition.extended` definition, and may map an entry
id to ``None`` (a known entry with no structure — structural columns serve
null).
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from functools import reduce
from math import gcd
from typing import Any

from httk.core import (
    EntryProvider,
    EntryTypeDefinition,
    PropertyDefinition,
    load_entry_type_definition,
)

from .elements import SYMBOLS
from .species_primitive_view import SpeciesPrimitiveView
from .structure import Structure
from .structure_like import StructureLike

_ELEMENTS: frozenset[str] = frozenset(SYMBOLS)


def _structures_definition() -> EntryTypeDefinition:
    """The vendored OPTIMADE ``structures`` entry-type definition (cached)."""
    return load_entry_type_definition("httk.atomistic", "structures")


# Served property name -> record column key. simple_property_handlers on the
# consumer side treats 'id' (against '__id') and 'type' (constant) specially.
_STRUCTURES_COLUMNS: dict[str, str] = {
    'id': '__id',
    'type': 'type',
    'elements': 'elements',
    'nelements': 'nelements',
    'nsites': 'nsites',
    'lattice_vectors': 'lattice_vectors',
    'cartesian_site_positions': 'cartesian_site_positions',
    'species_at_sites': 'species_at_sites',
    'species': 'species',
    'structure_features': 'structure_features',
}

# Structural columns that serve null for a None (structure-less) entry.
_STRUCTURAL_NULL_COLUMNS: tuple[str, ...] = (
    'elements',
    'nelements',
    'nsites',
    'lattice_vectors',
    'cartesian_site_positions',
    'species_at_sites',
    'species',
    'structure_features',
)

# Standard composition properties auto-derived from each Structure (None-safe).
_AUTO_DERIVED_COLUMNS: tuple[str, ...] = (
    'nperiodic_dimensions',
    'dimension_types',
    'elements_ratios',
    'chemical_formula_reduced',
    'chemical_formula_anonymous',
    'chemical_formula_descriptive',
)


def _as_structure(obj: StructureLike) -> Any:
    """Return something exposing the ``cell``/``sites``/``species``/``species_at_sites`` quartet."""
    if isinstance(obj, (tuple, list)):
        args: tuple[Any, ...] = tuple(obj)
        return Structure(*args)  # a (cell, sites, species, species_at_sites) tuple/list
    return obj


def _anonymous_symbol(index: int) -> str:
    """The OPTIMADE anonymous element symbol for position ``index`` (A, B, ..., Z, Aa, Ba, ...)."""
    first = chr(ord("A") + index % 26)
    tail = index // 26
    return first if tail == 0 else first + chr(ord("a") + tail - 1)


def _element_counts(structure: Any) -> Counter[str] | None:
    """Integer per-element site counts, or ``None`` unless every species is a single unattached element."""
    species_by_name = {species.name: species for species in structure.species}
    if not all(species.is_single_element for species in structure.species):
        return None
    counts: Counter[str] = Counter()
    for site_name in structure.species_at_sites:
        counts[species_by_name[site_name].chemical_symbols[0]] += 1
    return counts if counts else None


def _derived_properties(structure: Any) -> dict[str, Any]:
    """The auto-derived composition properties for ``structure`` (all ``None`` when not well-defined)."""
    counts = _element_counts(structure)
    if counts is None:
        return {name: None for name in _AUTO_DERIVED_COLUMNS}
    total = sum(counts.values())
    elements_sorted = sorted(counts)
    common = reduce(gcd, counts.values())
    reduced = {element: counts[element] // common for element in elements_sorted}
    reduced_formula = "".join(element + (str(amount) if amount > 1 else "") for element, amount in reduced.items())
    anonymous_amounts = sorted(reduced.values(), reverse=True)
    anonymous_formula = "".join(
        _anonymous_symbol(position) + (str(amount) if amount > 1 else "")
        for position, amount in enumerate(anonymous_amounts)
    )
    return {
        'nperiodic_dimensions': 3,
        'dimension_types': [1, 1, 1],
        'elements_ratios': [counts[element] / total for element in elements_sorted],
        'chemical_formula_reduced': reduced_formula,
        'chemical_formula_anonymous': anonymous_formula,
        'chemical_formula_descriptive': reduced_formula,
    }


class StructureEntryProvider(EntryProvider):
    """Serves OPTIMADE ``structures`` from a mapping of id to structure.

    ``entries`` maps each entry id to a :class:`~httk.atomistic.Structure` (or
    any structure exposing the ``cell``/``sites``/``species``/
    ``species_at_sites`` quartet, or a ``(cell, sites, species,
    species_at_sites)`` tuple), or to ``None`` for a known entry that has no
    structure (its structural columns then serve null).

    The always-served structural fields are ``id``, ``type``, ``nsites``,
    ``elements``, ``nelements``, ``species`` (as OPTIMADE species dicts),
    ``species_at_sites``, ``lattice_vectors`` (the cell basis rows),
    ``cartesian_site_positions`` (reduced coordinates times the cell basis), and
    ``structure_features`` (``disorder`` when any species mixes several chemical
    symbols; ``site_attachments`` when any species has attached atoms). The
    standard composition fields ``nperiodic_dimensions``, ``dimension_types``,
    ``elements_ratios``, ``chemical_formula_reduced`` / ``_anonymous`` /
    ``_descriptive`` are auto-derived for a fully ordered structure (every
    species a single, unattached element), else served as null.

    ``extra_definitions`` extends the served entry-type definition with custom
    database-specific properties (each carrying a registered prefix), and
    ``properties`` supplies their per-entry values as ``{entry_id: {name:
    value}}``; every property named there MUST be described by the (extended)
    definition (a :class:`ValueError` at construction names any offender), and a
    value absent for an entry is served as null.
    """

    def __init__(
        self,
        entries: Mapping[str, StructureLike | None],
        *,
        extra_definitions: Mapping[str, PropertyDefinition] | None = None,
        properties: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._structures: dict[str, Any] = {
            str(key): (None if value is None else _as_structure(value)) for key, value in entries.items()
        }
        self._extra_definitions: dict[str, PropertyDefinition] = dict(extra_definitions or {})
        self._properties: dict[str, dict[str, Any]] = {
            str(entry_id): dict(values) for entry_id, values in (properties or {}).items()
        }

        described = self._definition().properties
        used_names = sorted({name for values in self._properties.values() for name in values})
        offenders = [name for name in used_names if name not in described]
        if offenders:
            raise ValueError(
                "StructureEntryProvider was given properties not described by its (extended) definition: "
                + ", ".join(offenders)
                + ". Add them via extra_definitions (custom names need a registered prefix)."
            )
        self._property_names: list[str] = used_names

    def _definition(self) -> EntryTypeDefinition:
        definition = _structures_definition()
        if self._extra_definitions:
            definition = definition.extended(self._extra_definitions)
        return definition

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {'structures': self._definition()}

    def columns(self, entry_type: str) -> Mapping[str, str]:
        if entry_type != 'structures':
            raise KeyError("StructureEntryProvider serves only the 'structures' entry type.")
        columns = dict(_STRUCTURES_COLUMNS)
        for name in _AUTO_DERIVED_COLUMNS:
            columns[name] = name
        for name in self._property_names:
            columns[name] = name
        return columns

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        if entry_type != 'structures':
            raise KeyError("StructureEntryProvider serves only the 'structures' entry type.")
        records: list[dict[str, Any]] = []
        for entry_id, structure in self._structures.items():
            record: dict[str, Any] = {'__id': entry_id, 'type': 'structures'}
            if structure is None:
                for column in _STRUCTURAL_NULL_COLUMNS:
                    record[column] = None
                for name in _AUTO_DERIVED_COLUMNS:
                    record[name] = None
            else:
                species_dicts = [dict(SpeciesPrimitiveView(species)) for species in structure.species]
                elements = sorted(
                    {
                        symbol
                        for species in species_dicts
                        for symbol in species['chemical_symbols']
                        if symbol in _ELEMENTS
                    }
                )
                features: list[str] = []
                if any(len(species['chemical_symbols']) > 1 for species in species_dicts):
                    features.append('disorder')
                if any(species.get('attached') for species in species_dicts):
                    features.append('site_attachments')
                record.update(
                    {
                        'elements': elements,
                        'nelements': len(elements),
                        'nsites': len(structure.species_at_sites),
                        'lattice_vectors': structure.cell.basis.to_floats(),
                        'cartesian_site_positions': structure.cartesian_sites().to_floats(),
                        'species_at_sites': list(structure.species_at_sites),
                        'species': species_dicts,
                        'structure_features': features,
                    }
                )
                record.update(_derived_properties(structure))
            entry_properties = self._properties.get(entry_id, {})
            for name in self._property_names:
                record[name] = entry_properties.get(name)
            records.append(record)
        return records
