"""An :class:`~httk.core.EntryProvider` serving OPTIMADE ``structures`` from httk-atomistic.

:class:`StructureEntryProvider` maps :class:`~httk.atomistic.Structure` objects to
the neutral httk-core entry-provider contract, so a serving module (such as
*httk-optimade*) can expose them as an OPTIMADE ``structures`` endpoint without
this module depending on the serving module. The served entry type is described
by the vendored OPTIMADE standard ``structures`` definition (loaded via
:func:`httk.core.load_entry_type_definition`); the provider serves a subset of
its properties (chemical-formula fields are intentionally out of scope this
round, as they are ill-defined for disordered species).
"""

from collections.abc import Iterable, Mapping
from typing import Any

from httk.core import EntryProvider, EntryTypeDefinition, load_entry_type_definition

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


def _as_structure(obj: StructureLike) -> Any:
    """Return something exposing the ``cell``/``sites``/``species``/``species_at_sites`` quartet."""
    if isinstance(obj, (tuple, list)):
        args: tuple[Any, ...] = tuple(obj)
        return Structure(*args)  # a (cell, sites, species, species_at_sites) tuple/list
    return obj


class StructureEntryProvider(EntryProvider):
    """Serves OPTIMADE ``structures`` from a mapping of id to structure.

    ``structures`` maps each entry id to a :class:`~httk.atomistic.Structure`
    (or any structure exposing the ``cell``/``sites``/``species``/
    ``species_at_sites`` quartet, or a ``(cell, sites, species,
    species_at_sites)`` tuple). The derived fields are ``id``, ``type``,
    ``nsites``, ``elements``, ``nelements``, ``species`` (as OPTIMADE species
    dicts), ``species_at_sites``, ``lattice_vectors`` (the cell basis rows),
    ``cartesian_site_positions`` (reduced coordinates times the cell basis),
    and ``structure_features`` (``disorder`` when any species mixes several
    chemical symbols; ``site_attachments`` when any species has attached atoms).
    """

    def __init__(self, structures: Mapping[str, StructureLike]) -> None:
        self._structures = {str(key): _as_structure(value) for key, value in structures.items()}

    def entry_types(self) -> Mapping[str, EntryTypeDefinition]:
        return {'structures': _structures_definition()}

    def columns(self, entry_type: str) -> Mapping[str, str]:
        if entry_type != 'structures':
            raise KeyError("StructureEntryProvider serves only the 'structures' entry type.")
        return _STRUCTURES_COLUMNS

    def records(self, entry_type: str) -> Iterable[Mapping[str, Any]]:
        if entry_type != 'structures':
            raise KeyError("StructureEntryProvider serves only the 'structures' entry type.")
        records: list[dict[str, Any]] = []
        for entry_id, structure in self._structures.items():
            species_dicts = [dict(SpeciesPrimitiveView(species)) for species in structure.species]
            elements = sorted(
                {symbol for species in species_dicts for symbol in species['chemical_symbols'] if symbol in _ELEMENTS}
            )
            features: list[str] = []
            if any(len(species['chemical_symbols']) > 1 for species in species_dicts):
                features.append('disorder')
            if any(species.get('attached') for species in species_dicts):
                features.append('site_attachments')
            records.append(
                {
                    '__id': entry_id,
                    'type': 'structures',
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
        return records
