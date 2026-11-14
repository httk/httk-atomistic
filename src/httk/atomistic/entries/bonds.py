"""Serve a structure's bond network as the OPTIMADE property ``_httk_bonds``.

The value states the network in terms of the served sites and the served
``space_group_symmetry_operations_xyz`` of the same entry. Each bond is a row of six integers
``[site, partner, op, t1, t2, t3]``: the atom at the served position of ``site`` is bonded to the
served position of ``partner`` moved by operation ``op`` of the served list and then by the lattice
translation ``(t1, t2, t3)``. Without a served list ``op`` is 0 and means the identity. A unit cell
serves its coordinates as stored, so each bond is the identity followed by its translation. A
fundamental domain serves representative positions in its own setting, which can differ from the
stored reference atoms by a lattice translation, another orbit member, the setting transform or
(for a position stated to limited precision) a little more, so each bond's operation is corrected
per site before it is looked up in the served list. The served list holds every operation of the
group for the served cell, so the lookup always succeeds; if it ever does not, the network is
served as null with a warning. Shapes are served sparsely, as aligned lists of the sites with an
assigned shape and their symbols.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence
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
from httk.atomistic.models.structure.network import BondSource
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.shapes import _check_shape, _is_known, _unknown

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

    A network that cannot be served (an internal inconsistency, or a bond whose operation is not
    one of the served symmetry operations followed by a lattice translation) gives ``None`` and a
    warning; the rest of the entry is served as usual.

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
        served = structure.space_group_symmetry_operations_xyz  # the list the entry serves
        positions = (
            structure._representative_sites().reduced_coords.to_fractions()
            if isinstance(structure, FundamentalDomainStructure)
            else structure.sites.reduced_coords.to_fractions()
        )
        codes = _bond_codes(structure, positions, None if served is None else [AffineOperation(xyz) for xyz in served])
    except ValueError as error:
        _logger.warning(
            "the bond network of entry %s is served as null: %s",
            entry or "<unnamed>",
            error,
            extra={"context": "serve"},
        )
        return values
    values["_httk_bonds"] = _network_payload(
        ((bond.site, bond.partner, op, t) for bond, (op, t) in zip(network.pairs, codes, strict=True)),
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
    pairs: Iterable[tuple[int, int, int, Sequence[int]]],
    shapes: Sequence[str | None] | None,
    source: BondSource | None,
    shapes_source: BondSource | None,
) -> dict[str, Any]:
    """Return the ``_httk_bonds`` dictionary of served pairs ``(site, partner, op, translation)`` and per-site shapes."""
    assigned = [(site, symbol) for site, symbol in enumerate(shapes or ()) if symbol is not None]
    return {
        "pairs": [[site, partner, op, *t] for site, partner, op, t in pairs],
        "shapes": {"sites": [site for site, _ in assigned], "symbols": [symbol for _, symbol in assigned]}
        if assigned
        else None,
        "source": _source_payload(source),
        "shapes_source": _source_payload(shapes_source),
    }


def _listed(operations: Sequence[AffineOperation] | None) -> dict[_Op, tuple[int, tuple[Fraction, ...]]]:
    """Index listed operations by their exact value modulo lattice translations, the first of equals winning.

    :param operations: The listed operations, or ``None`` when none are listed (then only the
        identity, at index 0).
    :return: The wrapped operation mapped to its 0-based index and the translation of the listed entry.
    """
    listed: dict[_Op, tuple[int, tuple[Fraction, ...]]] = {}
    for index, operation in enumerate([AffineOperation.identity()] if operations is None else operations):
        W, w = _op(operation)
        listed.setdefault((W, tuple(x % 1 for x in w)), (index, w))
    return listed


def _bond_codes(
    structure: Any, positions: Sequence[Sequence[Fraction]], operations: Sequence[AffineOperation] | None
) -> list[tuple[int, tuple[int, int, int]]]:
    """Return each bond's operation on the given positions as an index into ``operations`` and a lattice translation.

    Per site, find the map ``A`` that takes the site's stored atom onto its given position:
    exactly, or within three times the coordinate precision for a position stated to limited
    precision. On a fundamental domain ``A = t o T o g`` (``g`` a stored group element, ``T`` the
    setting transform, ``t`` a lattice coset of ``T`` plus a lattice translation) and the stored
    atom is the reference atom; on a unit cell ``A`` is a lattice translation of the stored
    coordinates. A bond ``(i, j, b)`` then has the operation ``A_i o b o A_j^-1`` on the given
    positions, which takes the position of ``j`` to the atom bonded to the position of ``i``.
    That operation is the first entry of ``operations`` equal to it modulo a lattice translation,
    followed by that translation.

    :param structure: The structure, with a bond network.
    :param positions: One exact fractional position per site (per Wyckoff site of a domain).
    :param operations: The listed operations the indices refer to, or ``None`` when none are
        listed, which makes the identity index 0.
    :return: Per bond, the 0-based index of its operation and the lattice translation.
    :raises ValueError: If the structure has no network, a position is not (within tolerance) an
        atom of its site, or a bond operation is not a listed operation followed by a lattice
        translation.
    """
    if isinstance(structure, FundamentalDomainStructure):
        transform = _op(structure.transform.operation)
        moves = [_mul(transform, g) for g in _group(structure.spacegroup)[0]]
        cosets = [tuple(vector.to_fractions()) for vector in structure.transform.lattice_cosets()]
        references = [_reference(structure.spacegroup, site) for site in structure.wyckoff_sites]
    else:
        moves, cosets = [_IDENTITY], [_IDENTITY[1]]
        references = structure.sites.reduced_coords.to_fractions()
    tolerance = 3 * (getattr(structure, "coordinate_precision", None) or Fraction(0))
    maps: list[_Op] = []
    for number, (reference, point) in enumerate(zip(references, positions, strict=True)):
        candidates = []
        for moved in moves:
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
    network = structure.bonds
    if network is None:
        raise ValueError("the structure has no bond network")
    listed = _listed(operations)
    codes = []
    for bond in network.pairs:
        W, w = _mul(_mul(maps[bond.site], _op(bond.operation)), _inv(maps[bond.partner]))
        found = listed.get((W, tuple(x % 1 for x in w)))
        if found is None:
            raise ValueError(
                f"the operation {_affine((W, w)).to_xyz()} of bond {bond.site}-{bond.partner} is not a served "
                "symmetry operation followed by a lattice translation"
            )
        index, start = found
        t1, t2, t3 = (int(x - y) for x, y in zip(w, start, strict=True))
        codes.append((index, (t1, t2, t3)))
    return codes


def _integers(value: object, name: str) -> list[int]:
    if not isinstance(value, list | tuple) or not all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        raise TypeError(f"{name} must be a list of integers")
    return list(value)


def _check_bonds_value(
    value: Mapping[str, Any], nsites: int, operations: Sequence[AffineOperation] | None, *, cell: bool = False
) -> None:
    """Check the rules of an ``_httk_bonds`` value that its JSON schema does not express.

    Rows have six integers, name sites in range and an operation in range (0 when none are
    listed); the shape lists are aligned, name sites in range and known symbols
    (``_SHAPE_SYMBOLS``); each ``pair_rows`` row names its site; a repeated site has disjoint row
    lists. For a unit-cell span (``cell``) every row also names the identity ``x,y,z``, and each
    named symbol's number equals the atoms its shape covers, at least 3 for ``I``: a row
    ``(i, j, t)`` adds one atom at each end, ``(i, i, t)`` two at ``i``.

    :param value: The decoded value, a dictionary.
    :param nsites: The number of served sites.
    :param operations: The served operations the rows index, or ``None`` when none are served.
    :param cell: Whether the value is of a unit-cell span.
    :raises TypeError: If a value has the wrong JSON type.
    :raises ValueError: If a rule is violated.
    """
    count = 1 if operations is None else len(operations)
    # a stated list holds the literal identity x,y,z once (StructureSymmetry refuses anything else)
    identity = 0 if operations is None else next((i for i, g in enumerate(operations) if g.is_identity()), None)
    pairs = value.get("pairs")
    if not isinstance(pairs, list | tuple):
        raise TypeError("'pairs' must be a list")
    rows = [_integers(row, f"pairs[{number}]") for number, row in enumerate(pairs)]
    for number, row in enumerate(rows):
        if len(row) != 6:
            raise ValueError(f"pairs[{number}] must have six integers, not {len(row)}")
        if not (0 <= row[0] < nsites and 0 <= row[1] < nsites):
            raise ValueError(f"pairs[{number}] names a site outside 0..{nsites - 1}")
        if not 0 <= row[2] < count:
            raise ValueError(f"pairs[{number}] names operation {row[2]}, outside 0..{count - 1}")
        if cell and row[2] != identity:
            raise ValueError(f"pairs[{number}] names operation {row[2]}, which is not the identity of a unit cell")
    shapes = value.get("shapes")
    if shapes is None:
        return
    if not isinstance(shapes, Mapping):
        raise TypeError("'shapes' must be a dictionary or null")
    if unknown := set(shapes) - {"sites", "symbols", "pair_rows"}:
        raise ValueError(f"'shapes' has unknown keys {sorted(map(str, unknown))!r}")
    sites = _integers(shapes.get("sites"), "'shapes' 'sites'")
    symbols = shapes.get("symbols")
    if not isinstance(symbols, list | tuple) or not all(isinstance(symbol, str) for symbol in symbols):
        raise TypeError("'shapes' 'symbols' must be a list of strings")
    covers = shapes.get("pair_rows")
    if covers is None:
        covers = [None] * len(sites)
    if not isinstance(covers, list | tuple):
        raise TypeError("'shapes' 'pair_rows' must be a list or null")
    if not len(sites) == len(symbols) == len(covers):
        raise ValueError("the lists of 'shapes' must have the same length")
    if not sites:
        raise ValueError("'shapes' must be null when no shape is assigned")
    taken: dict[int, list[set[int] | None]] = {}
    for number, (site, symbol, cover) in enumerate(zip(sites, symbols, covers, strict=True)):
        if not 0 <= site < nsites:
            raise ValueError(f"shape {number} names site {site}, outside 0..{nsites - 1}")
        if not _is_known(symbol):
            raise ValueError(f"shape {number}: {_unknown(symbol)}")
        if cover is None:
            covered = {r for r, row in enumerate(rows) if site in row[:2]}
        else:
            covered = set(_integers(cover, f"'shapes' 'pair_rows'[{number}]"))
            if outside := sorted(r for r in covered if not (0 <= r < len(rows) and site in rows[r][:2])):
                raise ValueError(f"shape {number} of site {site} lists rows {outside} that do not name the site")
        others = taken.setdefault(site, [])
        if others and (cover is None or any(other is None or other & covered for other in others)):
            raise ValueError(f"site {site} has several shapes whose row lists are not disjoint lists")
        others.append(None if cover is None else covered)
        if cell:
            atoms = set()
            for i, j, _op, *t in (rows[r] for r in covered):
                if i == site:
                    atoms.add((j, *t))
                if j == site:
                    atoms.add((i, *(-x for x in t)))
            try:
                _check_shape(symbol, len(atoms))
            except ValueError as error:
                raise ValueError(
                    f"shape {number} {symbol!r} of site {site} covers {len(atoms)} bonded atoms: {error}"
                ) from None
