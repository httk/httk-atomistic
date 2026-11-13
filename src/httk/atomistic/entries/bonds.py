"""Serve a structure's bond network as the OPTIMADE property ``_httk_bonds``.

The value states the network in terms of the served sites: a unit cell serves its coordinates as
stored, so each bond's lattice translation is served as is. A fundamental domain serves
representative positions in its own setting, which can differ from the stored reference atoms
by a lattice translation, another orbit member, the setting transform or (for a position stated
to limited precision) a little more, so each bond's operation is corrected per site.
"""

import logging
from collections.abc import Iterable
from fractions import Fraction
from functools import cache
from typing import Any

from httk.core import PropertyDefinition

from httk.atomistic.entries.definitions import load_httk_definitions
from httk.atomistic.models.structure.asu import (
    FundamentalDomainStructure,
    _affine,
    _apply,
    _group,
    _inv,
    _mul,
    _Op,
    _op,
    _reference,
)
from httk.atomistic.models.structure.network import BondNetwork, BondSource
from httk.atomistic.symmetry.affine_operation import AffineOperation

__all__ = ["BOND_PROPERTY_KEYS", "bond_definitions", "bond_properties"]

_logger = logging.getLogger(__name__)

BOND_PROPERTY_KEYS: dict[str, str] = {"_httk_bonds": "bonds"}

_IDENTITY = _op(AffineOperation.identity())


@cache
def bond_definitions() -> dict[str, PropertyDefinition]:
    """Load the vendored bond-network property definition.

    The definition sets the response level ``should not``: the value is large and seldom needed,
    so it is served only when asked for with ``response_fields``.

    :return: The definition keyed by ``_httk_bonds``.
    """
    return load_httk_definitions(BOND_PROPERTY_KEYS)


def bond_properties(structure: Any, *, entry: str | None = None) -> dict[str, Any]:
    """Project a structure's bond network as the ``_httk_bonds`` JSON value.

    A network that cannot be served (an internal inconsistency) gives ``None`` and a warning; the
    rest of the entry is served as usual.

    :param structure: The structure to project, or ``None`` for an empty entry.
    :param entry: The entry id named in a warning.
    :return: The ``_httk_bonds`` value, or ``None`` when no network is known.
    """
    values: dict[str, Any] = {name: None for name in BOND_PROPERTY_KEYS}
    if structure is None:
        return values
    try:
        network = getattr(structure, "bonds", None)
        if network is None:
            return values
        operations = (
            _served_operations(structure, network)
            if isinstance(structure, FundamentalDomainStructure)
            else [bond.operation.to_xyz() for bond in network.pairs]
        )
    except ValueError as error:
        _logger.warning(
            "the bond network of entry %s is served as null: %s",
            entry or "<unnamed>",
            error,
            extra={"context": "serve"},
        )
        return values
    values["_httk_bonds"] = _network_payload(
        ((bond.site, bond.partner, xyz) for bond, xyz in zip(network.pairs, operations, strict=True)),
        network.shapes,
        network.source,
        network.shapes_source,
    )
    return values


def _source_payload(source: BondSource | None) -> dict[str, str | None] | None:
    if source is None:
        return None
    return {
        "description": source.description,
        "software": source.software,
        "version": source.version,
        "uri": source.uri,
    }


def _network_payload(
    pairs: Iterable[tuple[int, int, str]],
    shapes: tuple[str | None, ...] | None,
    source: BondSource | None,
    shapes_source: BondSource | None,
) -> dict[str, Any]:
    """Return the ``_httk_bonds`` dictionary of served pairs ``(site, partner, operation_xyz)``."""
    return {
        "pairs": [{"site": site, "partner": partner, "operation_xyz": xyz} for site, partner, xyz in pairs],
        "shapes": None if shapes is None else list(shapes),
        "source": _source_payload(source),
        "shapes_source": _source_payload(shapes_source),
    }


def _served_operations(structure: FundamentalDomainStructure, network: BondNetwork) -> list[str]:
    """Return each bond's operation on the served positions of a fundamental domain, as ``xyz``.

    Per site, find the map ``A = t o T o g`` (``g`` a stored group element, ``T`` the setting
    transform, ``t`` a lattice coset of ``T`` plus a lattice translation) that takes the site's
    reference atom onto its served position: exactly, or within three times the coordinate
    precision for a position stated to limited precision. A bond ``(i, j, b)`` is then served as
    ``A_i o b o A_j^-1``, which takes the served position of ``j`` to the atom bonded to the
    served position of ``i``.

    :param structure: The fundamental domain.
    :param network: Its network (normalized pairs).
    :return: One ``xyz`` operation per pair.
    :raises ValueError: If a served position is not (within tolerance) an atom of its site.
    """
    transform = _op(structure.transform.operation)
    served = structure._representative_sites().reduced_coords.to_fractions()
    tolerance = 3 * (structure.coordinate_precision or Fraction(0))
    cosets = [tuple(vector.to_fractions()) for vector in structure.transform.lattice_cosets()]
    group = _group(structure.spacegroup)[0]
    maps: list[_Op] = []
    for number, (site, point) in enumerate(zip(structure.wyckoff_sites, served, strict=True)):
        reference = _reference(structure.spacegroup, site)
        candidates = []
        for g in group:
            moved = _mul(transform, g)
            image = _apply(moved, reference)
            for coset in cosets:
                d = [p - a - c for p, a, c in zip(point, image, coset, strict=True)]
                shift = [Fraction(round(x)) for x in d]
                error = max(abs(x - n) for x, n in zip(d, shift, strict=True))
                candidates.append((error, moved, tuple(c + n for c, n in zip(coset, shift, strict=True))))
        error, moved, translation = min(candidates, key=lambda candidate: candidate[0])
        if error > tolerance:
            raise ValueError(f"served position {number} is {float(error):g} from an atom of its site")
        maps.append(_mul((_IDENTITY[0], translation), moved))
    return [
        _affine(_mul(_mul(maps[bond.site], _op(bond.operation)), _inv(maps[bond.partner]))).to_xyz()
        for bond in network.pairs
    ]
