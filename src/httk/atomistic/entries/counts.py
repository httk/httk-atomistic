"""Serving the atom and formula-unit reference counts of an ordered structure."""

import re
from collections.abc import Iterable
from functools import cache
from typing import Any

from httk.core import PropertyDefinition, load_property_definition
from httk.core.definition_ids import NATOMS, NFORMULA_UNITS

from httk.atomistic.models.structure.asu import FundamentalDomainStructure

__all__ = [
    "COUNT_PROPERTY_KEYS",
    "count_definitions",
    "count_properties",
    "formula_unit_counts",
    "is_ordered",
]

COUNT_PROPERTY_KEYS: dict[str, str] = {"_httk_natoms": NATOMS, "_httk_nformula_units": NFORMULA_UNITS}

_UNORDERED_FEATURES = frozenset({"assemblies", "disorder", "implicit_atoms", "site_attachments"})
_FORMULA_COUNT = re.compile(r"[A-Z][a-z]*([0-9]*)")


@cache
def count_definitions() -> dict[str, PropertyDefinition]:
    """Load the core atom and formula-unit count definitions.

    :return: The definitions keyed by their served names.
    """
    definitions = (
        load_property_definition(definition_id).served_form() for definition_id in COUNT_PROPERTY_KEYS.values()
    )
    return {definition.name: definition for definition in definitions}


def is_ordered(structure_features: Iterable[str] | None, species: Iterable[Any] | None) -> bool:
    """Decide whether every site holds exactly one fully occupied chemical species.

    Attached atoms count as unordered: they are atoms without a site of their own. Unknown
    features or species (a remote document omitting them) do not establish order.

    :param structure_features: The structure's OPTIMADE feature flags, or ``None`` if unknown.
    :param species: The structure's species; without ``implicit_atoms`` all of them are used.
    :return: Whether the structure is ordered.
    """
    if structure_features is None or species is None:
        return False
    return _UNORDERED_FEATURES.isdisjoint(structure_features) and all(
        tuple(value.chemical_symbols) != ("vacancy",) for value in species
    )


def formula_unit_counts(natoms: int | None, chemical_formula_reduced: str | None) -> dict[str, int | None]:
    """Relate a full-cell atom count to one formula unit of the reduced formula.

    :param natoms: The full-cell atom count, or ``None`` when it is unavailable.
    :param chemical_formula_reduced: The served reduced formula, or ``None``.
    :return: ``_httk_natoms`` and ``_httk_nformula_units``; the latter is ``None`` when the
        count does not divide exactly (a declared composition that contradicts the sites).
    """
    per_unit = sum(int(count or 1) for count in _FORMULA_COUNT.findall(chemical_formula_reduced or ""))
    units = None if natoms is None or not per_unit or natoms % per_unit else natoms // per_unit
    return {"_httk_natoms": natoms, "_httk_nformula_units": units}


def count_properties(structure: Any) -> dict[str, int | None]:
    """Project the reference counts of a structure.

    :param structure: The structure to project, or ``None`` for an empty entry.
    :return: ``_httk_natoms`` and ``_httk_nformula_units``, null unless the structure is ordered.
    """
    natoms = None
    if structure is not None and is_ordered(structure.structure_features, structure.species):
        if isinstance(structure, FundamentalDomainStructure):
            natoms = sum(structure.multiplicities())
        elif structure.site_coordinate_span in {"unit_cell", "molecular_unit_cell"}:
            natoms = structure.nsites
    return formula_unit_counts(natoms, None if structure is None else structure.chemical_formula_reduced)
