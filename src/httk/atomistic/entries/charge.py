"""Serving an explicitly assigned structure charge."""

from functools import cache
from typing import Any

from httk.core import PropertyDefinition

from httk.atomistic.entries.definitions import load_httk_definitions

__all__ = ["CHARGE_PROPERTY_KEYS", "charge_definitions", "charge_properties"]

CHARGE_PROPERTY_KEYS: dict[str, str] = {"_httk_charge": "structure_charge"}


@cache
def charge_definitions() -> dict[str, PropertyDefinition]:
    """Load the vendored structure-charge property definition.

    :return: The definition keyed by ``_httk_charge``.
    """
    return load_httk_definitions(CHARGE_PROPERTY_KEYS)


def charge_properties(structure: Any) -> dict[str, float | None]:
    """Project an explicitly assigned structure charge.

    :param structure: The structure to project, or ``None`` for an empty entry.
    :return: The top-level ``_httk_charge`` value, preserving an unstated charge as ``None``.
    """
    charge = None if structure is None else structure.charge
    return {"_httk_charge": None if charge is None else float(charge)}
