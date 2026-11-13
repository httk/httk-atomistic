"""Carrying a fundamental domain's bond network through re-descriptions (private).

Every operation that returns a :class:`~httk.atomistic.FundamentalDomainStructure` from another
and only re-describes the same sites carries the network with ``_carried``: given the exact
affine map from the source's stored-setting coordinates to the target's, both atoms of every
bond are mapped, located exactly in the target's orbits and folded back into a
:class:`~httk.atomistic.SiteBond` of the target. ``_matched_bonds`` does the same for a
target whose coordinates were snapped by recognition, matching each mapped atom to the nearest
target atom. Where the target's group is not the source's, every image of every bond is
carried, and the bond set must be invariant under the target's group, which is checked exactly.

Assigned shapes (``bonds.shapes``) travel with the network: every image of a site with a symbol
hands it to the target site it lands on (``_mapped_shapes``). A site without a symbol gives way to
one; two different symbols on one target site are a ``_ShapeConflict``, which means "not carried",
as a bond set that is not invariant does. The sources of the network are copied unchanged.

Unit cells reuse the same algebra through a bridge: ``_cell_domain`` presents a unit cell as the
P1 fundamental domain of its sites, whose normalized pairs are the unit cell's own. The bridge
never expands that domain, so coincident sites never meet its coincidence check; a carry that
cannot tell coincident atoms apart raises ``_CoincidentAtoms``, and every bridge path drops the
network instead. ``_expanded_network`` gives a fundamental domain's network on its unit cell.
Every drop is logged at debug level with the context ``bonds``.
"""

import logging
import math
from collections.abc import Callable
from fractions import Fraction
from functools import lru_cache

from httk.atomistic.models.structure.api import StructureAPI
from httk.atomistic.models.structure.asu import (
    FundamentalDomainStructure,
    WyckoffSite,
    _affine,
    _apply,
    _group,
    _in_group,
    _inv,
    _mul,
    _normalized_bonds,
    _Op,
    _op,
    _Point,
    _reference,
)
from httk.atomistic.models.structure.network import BondNetwork, SiteBond
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.symmetry._lattice import finite_translation_cosets
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.limits import _checkpoint
from httk.atomistic.symmetry.spacegroup import Spacegroup

__all__: list[str] = []

_logger = logging.getLogger(__name__)


def _dropped(reason: str) -> None:
    """Log at debug level that a carry dropped a bond network, and why."""
    _logger.debug("bond network dropped: %s", reason, extra={"context": "bonds"})


# Per wrapped point, one ``(site, operation, unwrapped image, species)`` per site with an atom there:
# coincident sites (two species at one point) stay two atoms, told apart by their species (its repr,
# so a renamed or changed species never passes for the original).
type _Index = dict[_Point, list[tuple[int, _Op, _Point, str]]]


def _orbit_index(structure: FundamentalDomainStructure) -> _Index:
    """Map every wrapped orbit point to the sites with an atom there."""
    index: _Index = {}
    species = {value.name: repr(value) for value in structure.species}
    for number, site in enumerate(structure.wyckoff_sites):
        reference = _reference(structure.spacegroup, site)
        for op in _group(structure.spacegroup)[0]:
            image = _apply(op, reference)
            entries = index.setdefault(_mod1(image), [])
            if not entries or entries[-1][0] != number:  # sites are indexed in turn
                entries.append((number, op, image, species[site.species]))
    return index


def _tags(structure: FundamentalDomainStructure) -> tuple[str, ...]:
    """Per site, its species (repr) where it shares a point with another site, else ``""``.

    Only coincident sites need a species to tell them apart, so carries that rename species (the
    anonymous frames) still work for every other site.
    """
    tags = [""] * len(structure.wyckoff_sites)
    for entries in _orbit_index(structure).values():
        if len(entries) > 1:
            for number, _op_, _image, species in entries:
                tags[number] = species
    return tuple(tags)


class _CoincidentAtoms(ValueError):
    """Raise when coincident sites cannot be told apart: two of one species, or renamed on the way."""


def _locate(index: _Index, point: _Point, tag: str = "") -> tuple[int, _Op]:
    """Return the site with an atom at an exact point (the one of species ``tag`` where several are) and its element.

    Where several sites share the point, exactly one must have the species ``tag``: a carry that
    renames coincident sites, or two sites of one species at one point, raises
    ``_CoincidentAtoms``. A point with no site at all is an internal error.
    """
    entries = index.get(_mod1(point), [])
    if not entries:
        raise ValueError(f"internal error: bond atom {point} is not an atom of the re-described structure")
    matches = entries if len(entries) == 1 else [entry for entry in entries if entry[3] == tag]
    if len(matches) != 1:
        raise _CoincidentAtoms(
            f"bond atom {point} is not one atom of the re-described structure "
            f"({len(entries)} coincident site(s) there, {len(matches)} of the atom's species)"
        )
    number, (W, w), image, _species = matches[0]
    return number, (W, tuple(a + p - q for a, p, q in zip(w, point, image, strict=True)))


def _fold(index: _Index, first: _Point, second: _Point, tags: tuple[str, str] = ("", "")) -> SiteBond:
    """Return the bond joining two exact atoms of a structure, in terms of its reference atoms."""
    site, element = _locate(index, first, tags[0])
    partner, operation = _locate(index, _apply(_inv(element), second), tags[1])
    return SiteBond(site, partner, _affine(operation))


def _bond_atoms(
    structure: FundamentalDomainStructure, bonds: tuple[SiteBond, ...] | None = None
) -> list[tuple[_Point, _Point, tuple[str, str]]]:
    """Return both atoms of every bond (the stored ones by default), in the stored setting, with their tags."""
    if bonds is None:
        assert structure._network is not None
        bonds = structure._network.pairs
    sites, group, tags = structure.wyckoff_sites, structure.spacegroup, _tags(structure)
    return [
        (
            _reference(group, sites[bond.site]),
            _apply(_op(bond.operation), _reference(group, sites[bond.partner])),
            (tags[bond.site], tags[bond.partner]),
        )
        for bond in bonds
    ]


@lru_cache(maxsize=4096)
def _contains(outer: Spacegroup, inner: Spacegroup, phi: _Op) -> bool:
    """Return whether ``phi``, from ``inner``'s coordinates to ``outer``'s, carries ``inner``'s group into ``outer``'s.

    The inner group's lattice translations are checked too: a smaller outer lattice does not contain them.
    """
    phi_inverse = _inv(phi)
    W = phi[0]
    one, zero = Fraction(1), Fraction(0)
    identity = (one, zero, zero, zero, one, zero, zero, zero, one)
    translations = [(identity, tuple(W[3 * r + c] for r in range(3))) for c in range(3)]
    return all(_in_group(outer, translation) for translation in translations) and all(
        _in_group(outer, _mul(_mul(phi, g), phi_inverse)) for g in _group(inner)[0]
    )


class _AmbiguousMatch(ValueError):
    """Raise when the atoms of a structure do not match a recognized structure's one to one."""


class _ShapeConflict(ValueError):
    """Raise when sites with different assigned shapes land on one site of the target."""


def _rebuilt(network: BondNetwork, pairs: tuple[SiteBond, ...], shapes: tuple[str | None, ...] | None) -> BondNetwork:
    """Return ``network`` with carried pairs and shapes, its sources copied (the shapes' only with shapes)."""
    return BondNetwork(pairs, shapes, network.source, None if shapes is None else network.shapes_source)


def _mapped_shapes(
    source: FundamentalDomainStructure,
    target: FundamentalDomainStructure,
    image: Callable[[_Point], _Point],
    cosets: list[_Point],
) -> tuple[str | None, ...] | None:
    """Hand the source's assigned shapes to the target sites its atoms land on.

    :param source: The structure whose shapes are carried.
    :param target: The re-described structure.
    :param image: Maps a source atom to the target's coordinates.
    :param cosets: The target-lattice shifts that complete the images where the lattices differ.
    :return: The target's shapes, or ``None`` when no target site gets a symbol.
    :raises _ShapeConflict: If two different symbols land on one target site.
    """
    if source._network is None or source._network.shapes is None:
        return None
    index = _orbit_index(target)
    shapes: list[str | None] = [None] * len(target.wyckoff_sites)
    for site, symbol, tag in zip(source.wyckoff_sites, source._network.shapes, _tags(source), strict=True):
        if symbol is None:
            continue
        reference = _reference(source.spacegroup, site)
        for g in _group(source.spacegroup)[0]:
            point = image(_apply(g, reference))
            for c in cosets:
                _checkpoint()
                number, _element = _locate(index, _shifted(point, c), tag)
                if shapes[number] not in (None, symbol):
                    raise _ShapeConflict(f"sites with the shapes {shapes[number]!r} and {symbol!r} merge into one")
                shapes[number] = symbol
    return None if all(symbol is None for symbol in shapes) else tuple(shapes)


def _mod1(point: _Point) -> _Point:
    return tuple(x % 1 for x in point)


def _shifted(point: _Point, shift: _Point) -> _Point:
    return tuple(a + b for a, b in zip(point, shift, strict=True))


def _cosets(phi: _Op) -> list[_Point]:
    """Return the images of the lattice translations under ``phi``'s linear part, modulo the lattice."""
    W = phi[0]
    columns = [tuple(W[3 * r + c] for r in range(3)) for c in range(3)]
    return [tuple(v.to_fractions()) for v in finite_translation_cosets(columns)]


def _network(
    source: FundamentalDomainStructure,
    target: FundamentalDomainStructure,
    phi: _Op,
    snap: Callable[[_Point], _Point] | None = None,
) -> tuple[SiteBond, ...] | None:
    """Carry the bond set of ``source`` onto ``target``'s group, or return ``None`` if it is not invariant under it.

    Every bond pair of the source (under its group and lattice, at the ``snap``-ped positions)
    is mapped and folded into target bonds. Those expand under the target's group and lattice;
    pulled back, they must give the source's pairs and no others, compared modulo the source
    lattice, so a translation the target has and the source lacks is checked too.
    """
    pairs: dict[tuple[object, ...], tuple[_Point, _Point, tuple[str, str]]] = {}
    for p, q, tags in _bond_atoms(source):
        for g in _group(source.spacegroup)[0]:
            _checkpoint()
            first, second = _apply(g, p), _apply(g, q)
            if snap is not None:
                first, second = snap(first), snap(second)
            pairs.setdefault(_pair_key(first, second, tags), (first, second, tags))
    index = _orbit_index(target)
    folded = set()
    forward, backward = _cosets(phi), _cosets(_inv(phi))
    for first, second, tags in pairs.values():
        for c in forward:
            _checkpoint()
            folded.add(_fold(index, _shifted(_apply(phi, first), c), _shifted(_apply(phi, second), c), tags))
    bonds = _normalized_bonds(target.spacegroup, target.wyckoff_sites, tuple(folded))
    phi_inverse = _inv(phi)
    pulled = set()
    for first, second, tags in _bond_atoms(target, bonds):
        for g in _group(target.spacegroup)[0]:
            p, q = _apply(phi_inverse, _apply(g, first)), _apply(phi_inverse, _apply(g, second))
            for c in backward:
                _checkpoint()
                pulled.add(_pair_key(_shifted(p, c), _shifted(q, c), tags))
    return bonds if pulled == pairs.keys() else None


def _carried[T: FundamentalDomainStructure](
    source: FundamentalDomainStructure, target: T, operation: AffineOperation | None = None
) -> T:
    """Return ``target`` with the network of ``source`` carried by the exact map ``operation``.

    ``operation`` maps the source's stored-setting coordinates to the target's (identity when
    omitted), and the target's atoms must be the images of the source's. A source without a
    network gives ``target`` unchanged; so does a target whose group the bond set is not
    invariant under, or one where sites with different assigned shapes merge (both logged).
    Assigned shapes and the sources are carried with the network. Coincident sites (several at
    one point) are told apart by their species, so a carry that renames or changes species must
    not involve them.

    :param source: The structure whose network is carried.
    :param target: The re-described structure.
    :param operation: The map from the source's stored-setting coordinates to the target's.
    :return: ``target`` with the carried network.
    :raises _CoincidentAtoms: If coincident atoms cannot be told apart.
    :raises ValueError: If a mapped atom is not an atom of the target, an internal invariant
        violation.
    """
    network = source._network
    if network is None or target is source:
        return target
    phi = _op(AffineOperation.identity() if operation is None else operation)
    same = (operation is None or operation.is_identity()) and source.spacegroup == target.spacegroup
    if same and [(x.wyckoff, x.free_params) for x in source.wyckoff_sites] == [
        (x.wyckoff, x.free_params) for x in target.wyckoff_sites
    ]:
        return target._with_bonds(network)
    bonds: tuple[SiteBond, ...] | None
    if same or (
        _contains(target.spacegroup, source.spacegroup, phi)
        and _contains(source.spacegroup, target.spacegroup, _inv(phi))
    ):
        # one group: one image of each bond names its orbit
        index = _orbit_index(target)
        bonds = tuple(_fold(index, _apply(phi, p), _apply(phi, q), tags) for p, q, tags in _bond_atoms(source))
    else:
        bonds = _network(source, target, phi)
        if bonds is None:
            _dropped("the bond set is not invariant under the target's group and lattice")
            return target
    try:
        shapes = _mapped_shapes(source, target, lambda point: _apply(phi, point), _cosets(phi))
    except _ShapeConflict as error:  # not carried, like a bond set that is not invariant
        _dropped(f"shape conflict: {error}")
        return target
    return target._with_bonds(_rebuilt(network, bonds, shapes))


def _bonds_key(structure: FundamentalDomainStructure) -> tuple[object, ...]:
    """Order structures with equal sites by their normalized networks, then assigned shapes (empty without a network)."""
    network = structure.bonds
    if network is None:
        return ()
    key: tuple[object, ...] = tuple((bond.site, bond.partner, *_op(bond.operation)) for bond in network.pairs)
    # -1 sorts before every bond entry (sites are >= 0), so a string never meets a number
    return key if network.shapes is None else (*key, (-1, tuple(symbol or "" for symbol in network.shapes)))


def _pair_key(first: _Point, second: _Point, tags: tuple[str, str] = ("", "")) -> tuple[object, ...]:
    """Return an unordered pair of exact (tagged) atoms, modulo lattice translations."""
    keys = []
    for (p, a), (q, b) in (((first, tags[0]), (second, tags[1])), ((second, tags[1]), (first, tags[0]))):
        shift = [math.floor(x) for x in p] * 2
        keys.append((*(x - s for x, s in zip((*p, *q), shift, strict=True)), a, b))
    return min(keys)


def _matched_bonds(
    source: FundamentalDomainStructure, target: FundamentalDomainStructure, operation: AffineOperation
) -> tuple[SiteBond, ...] | None:
    """Carry the network of ``source`` onto a recognized ``target``, or ``None`` if it is not invariant.

    ``operation`` is the exact map from the source's stored-setting coordinates to the target's.
    Recognition may have moved atoms, so every atom of the source is matched once to the nearest
    atom of the target, and the match must be one to one; the source's bonds are then taken at
    the matched positions and carried exactly. They describe the source's bond set only if it is
    invariant under the target's group and lattice.

    :param source: The structure whose network is carried.
    :param target: The recognized structure.
    :param operation: The map from the source's stored-setting coordinates to the target's.
    :return: The target's bonds, or ``None`` when the bond set is not invariant under its group.
    :raises _AmbiguousMatch: If an atom has no clearly nearest target atom, or the atoms do not
        match one to one.
    """
    phi = _op(operation)
    return _network(source, target, phi, _snap(source, target, phi))


def _matched(
    source: FundamentalDomainStructure, target: FundamentalDomainStructure, operation: AffineOperation
) -> BondNetwork | None:
    """Carry the network of ``source`` onto a recognized ``target``, matching once.

    As :func:`_matched_bonds`; the assigned shapes follow the same matched atoms, and the sources
    are copied.

    :param source: The structure (with a network) whose network is carried.
    :param target: The recognized structure.
    :param operation: The map from the source's stored-setting coordinates to the target's.
    :return: The target's network, or ``None`` when the bond set is not invariant.
    :raises _AmbiguousMatch: As :func:`_matched_bonds`.
    :raises _ShapeConflict: If sites with different assigned shapes merge.
    :raises _CoincidentAtoms: If coincident atoms cannot be told apart.
    """
    network = source._network
    assert network is not None, "only a structure with a network is matched"
    phi = _op(operation)
    snap = _snap(source, target, phi)
    bonds = _network(source, target, phi, snap)
    if bonds is None:
        return None
    shapes = _mapped_shapes(source, target, lambda point: _apply(phi, snap(point)), _cosets(phi))
    return _rebuilt(network, bonds, shapes)


def _snap(
    source: FundamentalDomainStructure, target: FundamentalDomainStructure, phi: _Op
) -> Callable[[_Point], _Point]:
    """Match every atom of ``source`` once to the nearest atom of ``target``.

    :param source: The structure whose atoms are matched.
    :param target: The recognized structure.
    :param phi: The exact map from the source's stored-setting coordinates to the target's.
    :return: The map moving a source atom (any lattice translate) onto its matched position.
    :raises _AmbiguousMatch: If an atom has no clearly nearest target atom, or the match is not one to one.
    """
    phi_inverse = _inv(phi)
    nearest = _nearest(target)
    matched = {}
    for atom in _orbit_index(source):  # every source atom, modulo the source lattice
        _checkpoint()
        matched[atom] = _apply(phi_inverse, nearest(_apply(phi, atom)))
    images = {_mod1(point) for point in matched.values()}
    backward = _cosets(phi_inverse)
    pulled = {
        _mod1(_shifted(_apply(phi_inverse, entry[2]), c))
        for entries in _orbit_index(target).values()
        for entry in entries
        for c in backward
    }
    if len(images) != len(matched) or images != pulled:
        raise _AmbiguousMatch("the atoms do not match the recognized structure's one to one")

    def snap(point: _Point) -> _Point:
        atom = _mod1(point)
        return _shifted(matched[atom], tuple(a - b for a, b in zip(point, atom, strict=True)))

    return snap


def _nearest(target: FundamentalDomainStructure) -> Callable[[_Point], _Point]:
    """Return a function matching a point to the nearest exact atom of ``target`` (Cartesian distance).

    The distances are compared in floats (one atom per point, its nearest lattice translate per
    component); the returned atom is exact.
    """
    points = [entries[0][2] for entries in _orbit_index(target).values()]  # coincident atoms are one point
    floats = [tuple(float(x) for x in image) for image in points]
    basis = target.cell.basis.to_floats()
    M = target.transform.matrix.to_floats()  # column convention: own = M stored + v
    own = [[sum(basis[i][k] * basis[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
    (g00, g01, g02), (g10, g11, g12), (g20, g21, g22) = (
        [sum(M[a][i] * own[a][b] * M[b][j] for a in range(3) for b in range(3)) for j in range(3)] for i in range(3)
    )

    def nearest(point: _Point) -> _Point:
        _checkpoint()
        x, y, z = (float(value) for value in point)
        best, second, index = math.inf, math.inf, 0
        for number, (a, b, c) in enumerate(floats):
            r0, r1, r2 = x - a, y - b, z - c
            r0, r1, r2 = r0 - round(r0), r1 - round(r1), r2 - round(r2)
            distance = (
                r0 * (g00 * r0 + g01 * r1 + g02 * r2)
                + r1 * (g10 * r0 + g11 * r1 + g12 * r2)
                + r2 * (g20 * r0 + g21 * r1 + g22 * r2)
            )
            if distance < best:
                best, second, index = distance, best, number
            elif distance < second:
                second = distance
        # ponytail: "clearly nearest" is within half the runner-up's distance; the one-to-one check backs it up
        if len(points) > 1 and not best * 4 < second:
            raise _AmbiguousMatch("an atom has no clearly nearest atom in the recognized structure")
        image = points[index]
        return tuple(q + round(p - q) for p, q in zip(point, image, strict=True))

    return nearest


def _cell_domain(structure: StructureAPI, *, network: bool = True) -> FundamentalDomainStructure:
    """Present a three-dimensional unit cell as the P1 fundamental domain of its sites, with its network.

    One ``WyckoffSite("a", x, species)`` per site, unwrapped, so the domain's normalized pairs are
    the unit cell's own. The domain is a private bridge: it is never expanded, so coincident sites
    never meet the expansion's coincidence check.

    :param structure: A structure presented as a unit cell (its network indexes ``sites``).
    :param network: Whether the domain takes the structure's network (else it has none, and the
        structure's network is never read).
    :return: The bridge domain.
    """
    rows = structure.sites.reduced_coords.to_fractions()
    sites = [WyckoffSite("a", row, name) for row, name in zip(rows, structure.species_at_sites, strict=True)]
    domain = FundamentalDomainStructure(structure.cell, 1, sites, structure.species)
    return domain._with_bonds(structure.bonds) if network else domain


def _cell_carried[T: UnitcellStructure](source: StructureAPI, target: T, operation: AffineOperation | None = None) -> T:
    """Return ``target`` with the unit-cell network of ``source`` carried by the exact map ``operation``.

    As :func:`_carried` through :func:`_cell_domain` bridges: ``operation`` maps the source's
    reduced coordinates to the target's and the target's atoms must be the images of the
    source's. Dropped (and logged) where the network is not invariant under the target's lattice,
    shapes conflict, or coincident atoms cannot be told apart.

    :param source: A structure presented as a unit cell, with or without a network.
    :param target: The re-described unit cell.
    :param operation: The map from the source's coordinates to the target's (identity when omitted).
    :return: ``target`` with the carried network, or without a network (any network of its own
        is never kept).
    """
    if source.bonds is None:
        return target._with_bonds(None)
    try:
        carried = _carried(_cell_domain(source), _cell_domain(target, network=False), operation)
    except _CoincidentAtoms as error:
        _dropped(f"coincident atoms: {error}")
        return target._with_bonds(None)
    return target._with_bonds(carried.bonds)


def _expanded_network(structure: FundamentalDomainStructure) -> BondNetwork | None:
    """Return a fundamental domain's network on the unit-cell sites it presents (``sites``), or ``None``.

    Every bond orbit is expanded into explicit pairs (lattice translations of the presented
    sites, in normal form) and every atom gets its site's assigned shape; the sources are copied.
    A molecular domain presents its retained representatives: when one is not an exact orbit
    point, the network is dropped (and logged).

    :param structure: The fundamental domain.
    :return: The unit-cell network, or ``None`` without a network or when it is dropped.
    """
    network = structure._network
    if network is None:
        return None
    if structure.molecular and any(
        site.representative is not None
        and site.representative.normalize() not in structure._representatives_for_site(site)
        for site in structure.wyckoff_sites
    ):
        _dropped("a retained molecular representative is not an exact orbit point")
        return None
    cell = UnitcellStructure(structure.cell, structure.sites, structure.species, structure.species_at_sites)
    target = _cell_domain(cell, network=False)
    phi = _op(structure.transform.operation)
    try:
        pairs = _network(structure, target, phi)
        shapes = _mapped_shapes(structure, target, lambda point: _apply(phi, point), _cosets(phi))
    except _CoincidentAtoms as error:
        _dropped(f"coincident atoms: {error}")
        return None
    except _ShapeConflict as error:  # an orbit's atoms share their site's shape, so this is unexpected
        _dropped(f"shape conflict: {error}")
        return None
    if pairs is None:  # every bond orbit is a union of unit-cell bonds, so this is unexpected
        _dropped("the bond set is not invariant under the unit cell's lattice")
        return None
    return _rebuilt(network, pairs, shapes)
