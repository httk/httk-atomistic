"""Shared JSON-compatible OPTIMADE payload projections."""

import numbers
from collections.abc import Iterable
from typing import Any


def concentration_payload(species_like: Any) -> list[int | float]:
    """Render species concentrations as JSON numbers, keeping an exact integer exact.

    An exact integral concentration (0 or 1) with no measured precision is written as the
    JSON integer, which reads back exact; a float ``1.0`` would read back as a measured value
    good to 1/10. Everything else is written as a float.

    :param species_like: A species object exposing ``concentration`` and, optionally,
        ``concentration_precision``.
    :return: One JSON number per constituent.
    """
    precisions = getattr(species_like, "concentration_precision", None) or ()
    return [
        int(value.numerator)
        if isinstance(value, numbers.Rational)
        and value.denominator == 1
        and (index >= len(precisions) or precisions[index] is None)
        else float(value)
        for index, value in enumerate(species_like.concentration)
    ]


def species_payload(species_like: Any) -> dict[str, object]:
    """Render species fields for an OPTIMADE structure entry.

    :param species_like: A species object exposing the stored species fields.
    :return: A JSON-compatible species payload.
    """
    payload: dict[str, object] = {
        "name": species_like.name,
        "chemical_symbols": list(species_like.chemical_symbols),
        "concentration": concentration_payload(species_like),
    }
    if species_like.mass is not None:
        payload["mass"] = list(species_like.mass)
    if species_like.original_name is not None:
        payload["original_name"] = species_like.original_name
    if species_like.attached is not None:
        payload["attached"] = list(species_like.attached)
        payload["nattached"] = list(species_like.nattached or ())
    charges = getattr(species_like, "charges", None)
    if charges is not None:
        # _httk_charges/_httk_spins are JSON-number floats; exact values live on Species.
        payload["_httk_charges"] = [None if value is None else float(value) for value in charges]
    spins = getattr(species_like, "spins", None)
    if spins is not None:
        payload["_httk_spins"] = [None if value is None else float(value) for value in spins]
    labels = getattr(species_like, "labels", None)
    if labels is not None:
        payload["_httk_labels"] = list(labels)
    return payload


def assemblies_payload(assemblies: Iterable[Any] | None) -> list[dict[str, object]] | None:
    """Render assembly fields without changing null semantics.

    :param assemblies: The assemblies to render, or ``None`` when unstated.
    :return: JSON-compatible assembly payloads, or ``None`` when unstated.
    """
    if assemblies is None:
        return None
    return [
        {
            "sites_in_groups": [list(group) for group in assembly.sites_in_groups],
            "group_probabilities": [float(value) for value in assembly.group_probabilities],
        }
        for assembly in assemblies
    ]
