"""An :class:`~httk.core.EntryProvider` serving OPTIMADE ``structures`` from httk-atomistic.

:class:`StructureEntryProvider` maps :class:`~httk.atomistic.Structure` objects to
the neutral httk-core entry-provider contract, so a serving module (such as
*httk-optimade*) can expose them as an OPTIMADE ``structures`` endpoint without
this module depending on the serving module. The property descriptions below are
a self-contained reformatting of the OPTIMADE specification (see the CC0
dedication note) and require no import from any serving module.
"""

# The property-description dictionaries in this module are a mere reformatting of
# information in the OPTIMADE specification [https://www.optimade.org/].
# Formally, the author makes a Public Domain Dedication according to CC0 1.0
# Universal (CC0 1.0): https://creativecommons.org/publicdomain/zero/1.0/
# (Note: this applies only to those literal description dictionaries.)

from collections.abc import Iterable, Mapping
from typing import Any

from httk.core import EntryProvider

from .elements import SYMBOLS
from .species_primitive_view import SpeciesPrimitiveView
from .structure import Structure
from .structure_like import StructureLike

_ELEMENTS: frozenset[str] = frozenset(SYMBOLS)

# The served ``structures`` entry type, described in the OPTIMADE
# property-definition dialect. Chemical-formula fields are intentionally out of
# scope this round (they are ill-defined for disordered species).
_STRUCTURES_ENTRY_INFO: dict[str, Any] = {
    'description': 'A structures entry.',
    'properties': {
        'id': {
            'description': "The ID for the structures entry.",
            'type': 'string',
            'fulltype': 'string',
            'required_support': True,
            'should_support': True,
            'required_query': True,
            'required_response': True,
            'default_response': True,
        },
        'type': {
            'description': "The name of the type of this entry, always 'structures'.",
            'type': 'string',
            'fulltype': 'string',
            'required_support': True,
            'should_support': True,
            'required_query': True,
            'required_response': True,
            'default_response': True,
        },
        'elements': {
            'description': "Names of the different elements present in the structure.",
            'type': 'list',
            'fulltype': 'list of string',
            'required_support': False,
            'should_support': True,
            'required_query': True,
            'required_response': False,
            'default_response': True,
        },
        'nelements': {
            'description': "The number of elements found in a structure.",
            'type': 'integer',
            'fulltype': 'integer',
            'required_support': False,
            'should_support': True,
            'required_query': True,
            'required_response': False,
            'default_response': True,
        },
        'nsites': {
            'description': "An integer specifying the length of the cartesian_site_positions property.",
            'type': 'integer',
            'fulltype': 'integer',
            'required_support': False,
            'should_support': True,
            'required_query': False,
            'required_response': False,
            'default_response': True,
        },
        'lattice_vectors': {
            'description': "The three lattice vectors in Cartesian coordinates, in ångström (Å).",
            'type': 'list',
            'fulltype': 'list of list of float',
            'unit': 'angstrom',
            'required_support': False,
            'should_support': True,
            'required_query': False,
            'required_response': False,
            'default_response': True,
            'dimensions': {'names': ['dim_lattice', 'dim_spatial'], 'sizes': [3, 3]},
        },
        'cartesian_site_positions': {
            'description': (
                "Cartesian positions of each site in the structure. A site is usually used to describe "
                "positions of atoms; what atoms can be encountered at a given site is conveyed by the "
                "species_at_sites property, and the species themselves are described in the species property."
            ),
            'type': 'list',
            'fulltype': 'list of list of float',
            'unit': 'angstrom',
            'required_support': False,
            'should_support': True,
            'required_query': False,
            'required_response': False,
            'default_response': True,
            'dimensions': {'names': ['dim_sites', 'dim_spatial'], 'sizes': [None, 3]},
        },
        'species_at_sites': {
            'description': (
                "Name of the species at each site (where values for sites are specified with the same order "
                "of the property cartesian_site_positions). The properties of the species are found in the "
                "property species."
            ),
            'type': 'list',
            'fulltype': 'list of string',
            'required_support': False,
            'should_support': True,
            'required_query': False,
            'required_response': False,
            'default_response': True,
        },
        'species': {
            'description': (
                "A list describing the species of the sites of this structure. Species can represent pure "
                "chemical elements, virtual-crystal atoms representing a statistical occupation of a given "
                "site by multiple chemical elements, and/or a location to which there are attached atoms."
            ),
            'type': 'list',
            'fulltype': 'list of dict',
            'required_support': False,
            'should_support': True,
            'required_query': False,
            'required_response': False,
            'default_response': True,
        },
        'structure_features': {
            'description': "A list of strings that flag which special features are used by the structure.",
            'type': 'list',
            'fulltype': 'list of string',
            'required_support': True,
            'should_support': True,
            'required_query': True,
            'required_response': False,
            'default_response': True,
        },
    },
}

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
    dicts), ``species_at_sites``, ``lattice_vectors`` (the cell matrix rows),
    ``cartesian_site_positions`` (reduced coordinates times the cell matrix),
    and ``structure_features`` (``disorder`` when any species mixes several
    chemical symbols; ``site_attachments`` when any species has attached atoms).
    """

    def __init__(self, structures: Mapping[str, StructureLike]) -> None:
        self._structures = {str(key): _as_structure(value) for key, value in structures.items()}

    def entry_types(self) -> Mapping[str, dict[str, Any]]:
        return {'structures': _STRUCTURES_ENTRY_INFO}

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
            cartesian = structure.cartesian_sites_floats()
            records.append(
                {
                    '__id': entry_id,
                    'type': 'structures',
                    'elements': elements,
                    'nelements': len(elements),
                    'nsites': len(structure.species_at_sites),
                    'lattice_vectors': [list(row) for row in structure.cell.matrix_floats()],
                    'cartesian_site_positions': [list(row) for row in cartesian],
                    'species_at_sites': list(structure.species_at_sites),
                    'species': species_dicts,
                    'structure_features': features,
                }
            )
        return records
