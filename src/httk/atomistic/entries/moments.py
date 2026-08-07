"""Serve structure site moments as Cartesian JSON values."""

from functools import cache
from typing import Any

from httk.core import PropertyDefinition

from httk.atomistic.entries.definitions import load_httk_definitions
from httk.atomistic.models.moments.cartesian_view import CartesianSiteMomentsView

__all__ = ["MOMENT_PROPERTY_KEYS", "moment_definitions", "moment_properties"]

MOMENT_PROPERTY_KEYS: dict[str, str] = {"_httk_site_moments": "site_moments"}


@cache
def moment_definitions() -> dict[str, PropertyDefinition]:
    """Load the vendored site-moment property definition.

    :return: The definition keyed by ``_httk_site_moments``.
    """
    return load_httk_definitions(MOMENT_PROPERTY_KEYS)


def moment_properties(structure: Any) -> dict[str, Any]:
    """Project non-collinear site moments as Cartesian JSON values.

    :param structure: The structure to project, or ``None`` for an empty entry.
    :return: The ``_httk_site_moments`` value, or ``None`` when unavailable or collinear.
    """
    values: dict[str, Any] = {name: None for name in MOMENT_PROPERTY_KEYS}
    if structure is None:
        return values
    moments = structure.site_moments
    if moments is not None and getattr(moments, "kind", None) != "collinear":
        values["_httk_site_moments"] = CartesianSiteMomentsView(moments).cartesian_moments.to_floats()
    return values
