"""Carrying a fundamental domain's bond network through re-descriptions (private).

Every operation that returns a :class:`~httk.atomistic.FundamentalDomainStructure` from another
and only re-describes the same sites carries the network with ``_carried``: given the exact
affine map from the source's stored-setting coordinates to the target's, both atoms of every
bond are mapped, located exactly in the target's orbits and folded back into a
:class:`~httk.atomistic.SiteBond` of the target. ``_matched_bonds`` does the same for a
target whose coordinates were snapped by recognition, matching each mapped atom to the nearest
target atom. Where the target's group is not the source's, every image of every bond is
carried, and the bond set must be invariant under the target's group, which is checked exactly.

Supplied shapes (``FundamentalDomainStructure.shapes``) travel with the network: every image of
a site with a symbol hands it to the target site it lands on (``_mapped_shapes``). A site
without a symbol gives way to one; two different symbols on one target site are a
``_ShapeConflict``, which means "not carried", as a bond set that is not invariant does.
"""

import math
from collections.abc import Callable
from fractions import Fraction
from functools import lru_cache

from httk.atomistic.models.structure.asu import (
    FundamentalDomainStructure,
    SiteBond,
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
from httk.atomistic.symmetry._lattice import finite_translation_cosets
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.limits import _checkpoint
from httk.atomistic.symmetry.spacegroup import Spacegroup

__all__: list[str] = []

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


def _locate(index: _Index, point: _Point, tag: str = "") -> tuple[int, _Op]:
    """Return the site with an atom at an exact point (the one of species ``tag`` where several are) and its element.

    Where several sites share the point, exactly one must have the species ``tag``: a carry that
    renames coincident sites, or two sites of one species at one point, fails loudly.
    """
    entries = index.get(_mod1(point), [])
    matches = entries if len(entries) == 1 else [entry for entry in entries if entry[3] == tag]
    if len(matches) != 1:
        raise ValueError(
            f"internal error: bond atom {point} is not one atom of the re-described structure "
            f"({len(entries)} site(s) there, {len(matches)} of the atom's species)"
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
    bonds = structure._bonds if bonds is None else bonds
    assert bonds is not None
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
    """Raise when sites with different supplied shapes land on one site of the target."""


def _mapped_shapes(
    source: FundamentalDomainStructure,
    target: FundamentalDomainStructure,
    image: Callable[[_Point], _Point],
    cosets: list[_Point],
) -> tuple[str | None, ...] | None:
    """Hand the source's supplied shapes to the target sites its atoms land on.

    :param source: The structure whose shapes are carried.
    :param target: The re-described structure.
    :param image: Maps a source atom to the target's coordinates.
    :param cosets: The target-lattice shifts that complete the images where the lattices differ.
    :return: The target's shapes, or ``None`` when no target site gets a symbol.
    :raises _ShapeConflict: If two different symbols land on one target site.
    """
    if source._shapes is None:
        return None
    index = _orbit_index(target)
    shapes: list[str | None] = [None] * len(target.wyckoff_sites)
    for site, symbol, tag in zip(source.wyckoff_sites, source._shapes, _tags(source), strict=True):
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
    invariant under, or one where sites with different supplied shapes merge. Supplied shapes
    are carried with the network. Coincident sites (several at one point) are told apart by
    their species, so a carry that renames or changes species must not involve them.

    :param source: The structure whose network is carried.
    :param target: The re-described structure.
    :param operation: The map from the source's stored-setting coordinates to the target's.
    :return: ``target`` with the carried network.
    :raises ValueError: If a mapped atom is not an atom of the target, an internal invariant
        violation.
    """
    if source._bonds is None or target is source:
        return target
    phi = _op(AffineOperation.identity() if operation is None else operation)
    same = (operation is None or operation.is_identity()) and source.spacegroup == target.spacegroup
    if same and [(x.wyckoff, x.free_params) for x in source.wyckoff_sites] == [
        (x.wyckoff, x.free_params) for x in target.wyckoff_sites
    ]:
        return target._with_bonds(source._bonds, shapes=source._shapes)
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
            return target
    if source._shapes is None:
        return target._with_bonds(bonds, shapes=None)
    try:
        shapes = _mapped_shapes(source, target, lambda point: _apply(phi, point), _cosets(phi))
    except _ShapeConflict:  # not carried, like a bond set that is not invariant
        return target
    return target._with_bonds(bonds, shapes=shapes)


def _bonds_key(structure: FundamentalDomainStructure) -> tuple[object, ...]:
    """Order structures with equal sites by their normalized networks, then supplied shapes (empty without a network)."""
    if structure.bonds is None:
        return ()
    key: tuple[object, ...] = tuple((bond.site, bond.partner, *_op(bond.operation)) for bond in structure.bonds)
    # -1 sorts before every bond entry (sites are >= 0), so a string never meets a number
    return key if structure.shapes is None else (*key, (-1, tuple(symbol or "" for symbol in structure.shapes)))


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
) -> tuple[tuple[SiteBond, ...] | None, tuple[str | None, ...] | None]:
    """Carry the network and supplied shapes of ``source`` onto a recognized ``target``, matching once.

    As :func:`_matched_bonds`; the shapes follow the same matched atoms.

    :param source: The structure whose network and shapes are carried.
    :param target: The recognized structure.
    :param operation: The map from the source's stored-setting coordinates to the target's.
    :return: The target's bonds (``None`` when the bond set is not invariant) and shapes (``None``
        without bonds or supplied shapes).
    :raises _AmbiguousMatch: As :func:`_matched_bonds`.
    :raises _ShapeConflict: If sites with different supplied shapes merge.
    """
    phi = _op(operation)
    snap = _snap(source, target, phi)
    bonds = _network(source, target, phi, snap)
    if bonds is None or source._shapes is None:
        return bonds, None
    return bonds, _mapped_shapes(source, target, lambda point: _apply(phi, snap(point)), _cosets(phi))


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
    """Return a function matching a point to the nearest exact atom of ``target`` (Cartesian distance)."""
    points = [entry[2] for entries in _orbit_index(target).values() for entry in entries]
    basis = target.cell.basis.to_floats()
    M = target.transform.matrix.to_floats()  # column convention: own = M stored + v
    own = [[sum(basis[i][k] * basis[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
    metric = [
        [sum(M[a][i] * own[a][b] * M[b][j] for a in range(3) for b in range(3)) for j in range(3)] for i in range(3)
    ]

    def nearest(point: _Point) -> _Point:
        found: dict[_Point, float] = {}
        for image in points:
            _checkpoint()
            shift = [round(p - q) for p, q in zip(point, image, strict=True)]
            r = [float(p - q - s) for p, q, s in zip(point, image, shift, strict=True)]
            distance = sum(r[a] * metric[a][b] * r[b] for a in range(3) for b in range(3))
            exact = tuple(q + s for q, s in zip(image, shift, strict=True))
            found[exact] = min(distance, found.get(exact, math.inf))
        ranked = sorted(found.values())
        # ponytail: "clearly nearest" is within half the runner-up's distance; the one-to-one check backs it up
        if len(ranked) > 1 and not ranked[0] * 4 < ranked[1]:
            raise _AmbiguousMatch("an atom has no clearly nearest atom in the recognized structure")
        return min(found, key=found.__getitem__)

    return nearest
