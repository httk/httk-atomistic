"""Serving how precisely a structure's numbers were stated.

OPTIMADE standardises nothing about data precision — the only ``standard_uncertainty`` in
its structures definition is CODATA metadata inside a unit definition — so these are
database-specific properties described by the published definitions in
:mod:`~httk.atomistic.anyterial_properties`.

Serving them matters because a consumer choosing a matching tolerance or a symmetry
``symprec`` otherwise has to guess. With these two values it can derive one, exactly as
:func:`~httk.atomistic.structure_tolerance` does locally.
"""

from functools import cache
from typing import Any

from httk.core import PropertyDefinition

from .anyterial_properties import load_anyterial_definitions

__all__ = ["PRECISION_PROPERTY_KEYS", "precision_definitions", "precision_properties"]

#: Served name to vendored anyterial definition. The coordinate precision is dimensionless
#: (a fraction of a cell edge) and the basis precision is a length, which is why they are
#: two definitions rather than one.
PRECISION_PROPERTY_KEYS: dict[str, str] = {
    "_httk_coordinate_precision": "fractional_coordinate_precision",
    "_httk_basis_precision": "length_precision",
}


@cache
def precision_definitions() -> dict[str, PropertyDefinition]:
    """The vendored precision definitions, keyed by their served name."""
    return load_anyterial_definitions(PRECISION_PROPERTY_KEYS)


def precision_properties(structure: Any) -> dict[str, Any]:
    """How precisely this structure's coordinates and cell were stated.

    Both are ``null`` for a structure that does not say — one built by hand, or read from a
    source that does not write its numbers to a definite number of digits. That is the
    honest answer, and it is distinguishable from a claim of exactness.

    Rendered as floats, since that is what a JSON consumer will use them as; they are held
    exactly on the structure itself.
    """
    values: dict[str, Any] = {name: None for name in PRECISION_PROPERTY_KEYS}
    if structure is None:
        return values

    coordinate = structure.sites.precision
    basis = structure.cell.precision
    if coordinate is not None:
        values["_httk_coordinate_precision"] = float(coordinate)
    if basis is not None:
        values["_httk_basis_precision"] = float(basis)
    return values
