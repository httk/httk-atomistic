"""Shared JSON-compatible OPTIMADE payload projections."""

from collections.abc import Iterable
from typing import Any


def species_payload(species_like: Any) -> dict[str, object]:
    """Render a Species or SpeciesRecord with the OPTIMADE species fields."""
    payload: dict[str, object] = {
        "name": species_like.name,
        "chemical_symbols": list(species_like.chemical_symbols),
        "concentration": [float(value) for value in species_like.concentration],
    }
    if species_like.mass is not None:
        payload["mass"] = list(species_like.mass)
    if species_like.original_name is not None:
        payload["original_name"] = species_like.original_name
    if species_like.attached is not None:
        payload["attached"] = list(species_like.attached)
        payload["nattached"] = list(species_like.nattached or ())
    return payload


def assemblies_payload(assemblies: Iterable[Any] | None) -> list[dict[str, object]] | None:
    """Render Assembly or AssemblyRecord values without changing null semantics."""
    if assemblies is None:
        return None
    return [
        {
            "sites_in_groups": [list(group) for group in assembly.sites_in_groups],
            "group_probabilities": [float(value) for value in assembly.group_probabilities],
        }
        for assembly in assemblies
    ]
