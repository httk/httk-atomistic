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
assigned shape and their symbols. ``dimensions`` is the structure's
:attr:`~httk.atomistic.models.structure.StructureAPI.bonds_dimensions`, as a list. Torsion
statements are served as aligned lists: per statement 20 integers, ``[site, op, t1, t2, t3]`` for
each of its four atoms, found as a bond's partner is, and its symbol, angle and uncertainty. Each
atom keeps its own code; no atom is moved onto the identity, which could apply an improper served
operation and negate the angle.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from functools import cache
from typing import Any

from httk.core import PropertyDefinition

from httk.atomistic.bonding import _bonds_dimensions
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
from httk.atomistic.models.structure.network import _TORSION_SYMBOLS, BondSource, SiteTorsion
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
        listed = None if served is None else [AffineOperation(xyz) for xyz in served]
        codes = _bond_codes(structure, positions, listed)
        atoms = _torsion_codes(structure, positions, listed)
        dimensions = _bonds_dimensions(structure)
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
        dimensions,
        network.torsions,
        atoms,
        network.torsions_source,
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
    dimensions: Sequence[int] | None,
    torsions: Sequence[SiteTorsion] | None,
    atoms: Sequence[Sequence[int]],
    torsions_source: BondSource | None,
) -> dict[str, Any]:
    """Return the ``_httk_bonds`` dictionary of served pairs ``(site, partner, op, translation)``, per-site shapes,
    dimensions and torsions (the served ``atoms`` row of each statement; a column is null when no statement has a value).
    """
    assigned = [(site, symbol) for site, symbol in enumerate(shapes or ()) if symbol is not None]

    def column(values: list[Any]) -> list[Any] | None:
        return None if all(value is None for value in values) else values

    return {
        "pairs": [[site, partner, op, *t] for site, partner, op, t in pairs],
        "shapes": {"sites": [site for site, _ in assigned], "symbols": [symbol for _, symbol in assigned]}
        if assigned
        else None,
        "source": _source_payload(source),
        "shapes_source": _source_payload(shapes_source),
        "dimensions": None if dimensions is None else list(dimensions),
        "torsions": None
        if not torsions
        else {
            "atoms": [list(row) for row in atoms],
            "symbols": column([row.symbol for row in torsions]),
            "angles": column([row.angle for row in torsions]),
            "angle_uncertainties": column([row.uncertainty for row in torsions]),
        },
        "torsions_source": _source_payload(torsions_source),
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

    Per site, find the map ``A`` that takes the site's stored atom onto its given position
    (:func:`_site_maps`). A bond ``(i, j, b)`` then has the operation ``A_i o b o A_j^-1`` on the
    given positions, which takes the position of ``j`` to the atom bonded to the position of ``i``.
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
    maps = _site_maps(structure, positions)
    network = structure.bonds
    if network is None:
        raise ValueError("the structure has no bond network")
    listed = _listed(operations)
    return [
        _code(
            listed,
            _mul(_mul(maps[bond.site], _op(bond.operation)), _inv(maps[bond.partner])),
            f"bond {bond.site}-{bond.partner}",
        )
        for bond in network.pairs
    ]


def _torsion_codes(
    structure: Any, positions: Sequence[Sequence[Fraction]], operations: Sequence[AffineOperation] | None
) -> list[list[int]]:
    """Return each torsion statement's four atoms on the given positions as a row ``[site, op, t1, t2, t3]`` per atom.

    Each atom is served on its own (no atom is moved to the identity): atom ``(s, g)`` is the stored
    atom ``g(x_s)``, written in the served coordinates as ``T(g(x_s))`` (``T`` the setting
    transform of a domain, the identity on a unit cell), which is ``T o g o A_s^-1`` applied to
    the given position of ``s`` (``A_s`` from :func:`_site_maps`). That operation is looked up as
    in :func:`_bond_codes`. The served atoms are the stored atoms under the same lattice vectors,
    so the signs of the angles are unchanged.

    :param structure: The structure, with a bond network.
    :param positions: One exact fractional position per site (per Wyckoff site of a domain).
    :param operations: The listed operations the indices refer to, or ``None`` when none are listed.
    :return: Per statement, the 20 integers of its four atoms; none when the network has no torsions.
    :raises ValueError: As :func:`_bond_codes`.
    """
    network = structure.bonds
    if network is None or network.torsions is None:
        return []
    maps = _site_maps(structure, positions)
    frame = _op(structure.transform.operation) if isinstance(structure, FundamentalDomainStructure) else _IDENTITY
    listed = _listed(operations)
    rows = []
    for row in network.torsions:
        atoms: list[int] = []
        for site, operation in zip(row.sites, row.operations, strict=True):
            index, t = _code(
                listed, _mul(_mul(frame, _op(operation)), _inv(maps[site])), f"torsion atom of site {site}"
            )
            atoms.extend((site, index, *t))
        rows.append(atoms)
    return rows


def _site_maps(structure: Any, positions: Sequence[Sequence[Fraction]]) -> list[_Op]:
    """Return per site the map ``A`` that takes the site's stored atom onto its given position.

    The map is exact, or within three times the coordinate precision for a position stated to
    limited precision. On a fundamental domain ``A = t o T o g`` (``g`` a stored group element,
    ``T`` the setting transform, ``t`` a lattice coset of ``T`` plus a lattice translation) and
    the stored atom is the reference atom; on a unit cell ``A`` is a lattice translation of the
    stored coordinates.

    :param structure: The structure.
    :param positions: One exact fractional position per site (per Wyckoff site of a domain).
    :return: One map per site.
    :raises ValueError: If a position is not (within tolerance) an atom of its site.
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
    return maps


def _code(
    listed: dict[_Op, tuple[int, tuple[Fraction, ...]]], operation: _Op, what: str
) -> tuple[int, tuple[int, int, int]]:
    """Return an operation as the index of the first listed entry equal to it modulo a lattice translation, and that translation.

    :param listed: The listed operations, as :func:`_listed` indexes them.
    :param operation: The operation on the served positions.
    :param what: What the operation belongs to, for the error message.
    :return: The 0-based index and the lattice translation.
    :raises ValueError: If no listed entry is equal to it modulo a lattice translation.
    """
    W, w = operation
    found = listed.get((W, tuple(x % 1 for x in w)))
    if found is None:
        raise ValueError(
            f"the operation {_affine(operation).to_xyz()} of {what} is not a served symmetry operation followed by "
            "a lattice translation"
        )
    index, start = found
    t1, t2, t3 = (int(x - y) for x, y in zip(w, start, strict=True))
    return index, (t1, t2, t3)


def _check_torsions(torsions: object, nsites: int, count: int, identity: int | None, *, cell: bool) -> None:
    """Check the JSON shape of an ``_httk_bonds`` ``torsions`` dictionary.

    Its keys are ``atoms``, ``symbols``, ``angles`` and ``angle_uncertainties``. ``atoms`` is a
    non-empty list of rows of 20 integers ``[site, op, t1, t2, t3]`` per atom with sites in range
    and operations in range (``identity`` for a unit-cell span); each other
    key is null or a list aligned with ``atoms`` of a symbol (``_TORSION_SYMBOLS``), an angle in
    [-180, 180], or a positive uncertainty of a row with an angle, or null.

    :param torsions: The decoded ``torsions`` value, not null.
    :param nsites: The number of served sites.
    :param count: The number of served operations (1 when none are served).
    :param identity: The index of the served identity, or ``None`` when none is served.
    :param cell: Whether the value is of a unit-cell span.
    :raises TypeError: If a value has the wrong JSON type.
    :raises ValueError: If a rule is violated.
    """
    if not isinstance(torsions, Mapping):
        raise TypeError("'torsions' must be a dictionary or null")
    keys = {"atoms", "symbols", "angles", "angle_uncertainties"}
    if set(torsions) != keys:
        raise ValueError(f"'torsions' must have the keys {sorted(keys)}, not {sorted(map(str, torsions))}")
    atoms = torsions["atoms"]
    if not isinstance(atoms, list | tuple):
        raise TypeError("'torsions' 'atoms' must be a list")
    if not atoms:
        raise ValueError("'torsions' must be null when there are no statements")
    for number, row in enumerate(atoms):
        row = _integers(row, f"'torsions' 'atoms'[{number}]")
        if len(row) != 20:
            raise ValueError(f"'torsions' 'atoms'[{number}] must have 20 integers, not {len(row)}")
        for site, op in zip(row[0::5], row[1::5], strict=True):
            if not 0 <= site < nsites:
                raise ValueError(f"'torsions' 'atoms'[{number}] names a site outside 0..{nsites - 1}")
            if not 0 <= op < count:
                raise ValueError(f"'torsions' 'atoms'[{number}] names operation {op}, outside 0..{count - 1}")
            if cell and op != identity:
                raise ValueError(
                    f"'torsions' 'atoms'[{number}] names operation {op}, which is not the identity of a unit cell"
                )
    columns = {}
    for key in ("symbols", "angles", "angle_uncertainties"):
        column = torsions[key]
        if column is not None and not isinstance(column, list | tuple):
            raise TypeError(f"'torsions' {key!r} must be a list or null")
        if column is not None and len(column) != len(atoms):
            raise ValueError("the lists of 'torsions' must have the same length")
        columns[key] = [None] * len(atoms) if column is None else column
    for number, (symbol, angle, uncertainty) in enumerate(
        zip(columns["symbols"], columns["angles"], columns["angle_uncertainties"], strict=True)
    ):
        if symbol is not None and symbol not in _TORSION_SYMBOLS:
            raise ValueError(f"torsion {number}: unknown symbol {symbol!r}")
        for name, x in (("angle", angle), ("uncertainty", uncertainty)):
            if x is not None and (isinstance(x, bool) or not isinstance(x, int | float | Decimal)):
                raise TypeError(f"torsion {number}: the {name} must be a number or null")
        if angle is not None and not -180 <= angle <= 180:
            raise ValueError(f"torsion {number}: the angle {angle!r} is outside [-180, 180]")
        if uncertainty is not None and (angle is None or not uncertainty > 0):
            raise ValueError(f"torsion {number}: an uncertainty must be positive and needs an angle")


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
    lists; ``dimensions``, when given, is null or distinct integers from 0 to 3 in increasing order;
    ``torsions`` is null or as :func:`_check_torsions` checks it.
    For a unit-cell span (``cell``) every row also names the identity ``x,y,z``, and each
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
    if value.get("torsions") is not None:
        _check_torsions(value["torsions"], nsites, count, identity, cell=cell)
    if value.get("dimensions") is not None:
        dimensions = _integers(value["dimensions"], "'dimensions'")
        if dimensions != sorted(set(dimensions)) or not all(0 <= x <= 3 for x in dimensions):
            raise ValueError("'dimensions' must list distinct integers from 0 to 3 in increasing order")
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
