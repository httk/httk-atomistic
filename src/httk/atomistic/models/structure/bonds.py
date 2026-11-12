"""Carrying a fundamental domain's bond network through re-descriptions (private).

Every operation that returns a :class:`~httk.atomistic.FundamentalDomainStructure` from another
and only re-describes the same sites carries the network with ``_carried``: given the exact
affine map from the source's stored-setting coordinates to the target's, both atoms of every
bond are mapped, located exactly in the target's orbits and folded back into a
:class:`~httk.atomistic.SiteBond` of the target. ``_matched_bonds`` does the same for a
target whose coordinates were snapped by recognition, matching each mapped atom to the nearest
target atom. Where the target's group is not the source's, every image of every bond is
carried, and the bond set must be invariant under the target's group, which is checked exactly.
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

type _Index = dict[_Point, tuple[int, _Op, _Point]]


def _orbit_index(structure: FundamentalDomainStructure) -> _Index:
    """Map every wrapped orbit point of every site to ``(site, operation, unwrapped image)``."""
    index: _Index = {}
    for number, site in enumerate(structure.wyckoff_sites):
        reference = _reference(structure.spacegroup, site)
        for op in _group(structure.spacegroup)[0]:
            image = _apply(op, reference)
            index.setdefault(tuple(x % 1 for x in image), (number, op, image))
    return index


def _locate(index: _Index, point: _Point) -> tuple[int, _Op]:
    """Return the site whose orbit holds an exact point and the element moving its reference atom there."""
    hit = index.get(tuple(x % 1 for x in point))
    if hit is None:
        raise ValueError(f"internal error: bond atom {point} is not an atom of the re-described structure")
    number, (W, w), image = hit
    return number, (W, tuple(a + p - q for a, p, q in zip(w, point, image, strict=True)))


def _fold(index: _Index, first: _Point, second: _Point) -> SiteBond:
    """Return the bond joining two exact atoms of a structure, in terms of its reference atoms."""
    site, element = _locate(index, first)
    partner, operation = _locate(index, _apply(_inv(element), second))
    return SiteBond(site, partner, _affine(operation))


def _bond_atoms(
    structure: FundamentalDomainStructure, bonds: tuple[SiteBond, ...] | None = None
) -> list[tuple[_Point, _Point]]:
    """Return both atoms of every bond (the stored ones by default), in the stored setting."""
    bonds = structure._bonds if bonds is None else bonds
    assert bonds is not None
    sites, group = structure.wyckoff_sites, structure.spacegroup
    return [
        (_reference(group, sites[bond.site]), _apply(_op(bond.operation), _reference(group, sites[bond.partner])))
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
    pairs: dict[tuple[Fraction, ...], tuple[_Point, _Point]] = {}
    for p, q in _bond_atoms(source):
        for g in _group(source.spacegroup)[0]:
            _checkpoint()
            first, second = _apply(g, p), _apply(g, q)
            if snap is not None:
                first, second = snap(first), snap(second)
            pairs.setdefault(_pair_key(first, second), (first, second))
    index = _orbit_index(target)
    folded = set()
    forward, backward = _cosets(phi), _cosets(_inv(phi))
    for first, second in pairs.values():
        for c in forward:
            _checkpoint()
            folded.add(_fold(index, _shifted(_apply(phi, first), c), _shifted(_apply(phi, second), c)))
    bonds = _normalized_bonds(target.spacegroup, target.wyckoff_sites, tuple(folded))
    phi_inverse = _inv(phi)
    pulled = set()
    for first, second in _bond_atoms(target, bonds):
        for g in _group(target.spacegroup)[0]:
            p, q = _apply(phi_inverse, _apply(g, first)), _apply(phi_inverse, _apply(g, second))
            for c in backward:
                _checkpoint()
                pulled.add(_pair_key(_shifted(p, c), _shifted(q, c)))
    return bonds if pulled == pairs.keys() else None


def _carried[T: FundamentalDomainStructure](
    source: FundamentalDomainStructure, target: T, operation: AffineOperation | None = None
) -> T:
    """Return ``target`` with the network of ``source`` carried by the exact map ``operation``.

    ``operation`` maps the source's stored-setting coordinates to the target's (identity when
    omitted), and the target's atoms must be the images of the source's. A source without a
    network gives ``target`` unchanged; so does a target whose group the bond set is not
    invariant under.

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
        return target._with_bonds(source._bonds)
    if same or (
        _contains(target.spacegroup, source.spacegroup, phi)
        and _contains(source.spacegroup, target.spacegroup, _inv(phi))
    ):
        # one group: one image of each bond names its orbit
        index = _orbit_index(target)
        return target._with_bonds(tuple(_fold(index, _apply(phi, p), _apply(phi, q)) for p, q in _bond_atoms(source)))
    bonds = _network(source, target, phi)
    return target if bonds is None else target._with_bonds(bonds)


def _bonds_key(structure: FundamentalDomainStructure) -> tuple[object, ...]:
    """Order structures with equal sites by their normalized networks (empty without a network)."""
    if structure.bonds is None:
        return ()
    return tuple((bond.site, bond.partner, *_op(bond.operation)) for bond in structure.bonds)


def _pair_key(first: _Point, second: _Point) -> tuple[Fraction, ...]:
    """Return an unordered pair of exact points, modulo lattice translations."""
    keys = []
    for p, q in ((first, second), (second, first)):
        shift = [math.floor(x) for x in p] * 2
        keys.append(tuple(x - s for x, s in zip((*p, *q), shift, strict=True)))
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
    phi_inverse = _inv(phi)
    nearest = _nearest(target)
    matched = {}
    for atom in _orbit_index(source):  # every source atom, modulo the source lattice
        _checkpoint()
        matched[atom] = _apply(phi_inverse, nearest(_apply(phi, atom)))
    images = {_mod1(point) for point in matched.values()}
    backward = _cosets(phi_inverse)
    pulled = {
        _mod1(_shifted(_apply(phi_inverse, image), c))
        for _site, _element, image in _orbit_index(target).values()
        for c in backward
    }
    if len(images) != len(matched) or images != pulled:
        raise _AmbiguousMatch("the atoms do not match the recognized structure's one to one")

    def snap(point: _Point) -> _Point:
        atom = _mod1(point)
        return _shifted(matched[atom], tuple(a - b for a, b in zip(point, atom, strict=True)))

    return _network(source, target, phi, snap)


def _nearest(target: FundamentalDomainStructure) -> Callable[[_Point], _Point]:
    """Return a function matching a point to the nearest exact atom of ``target`` (Cartesian distance)."""
    points = [image for _site, _element, image in _orbit_index(target).values()]
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
