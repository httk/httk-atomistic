"""Torsion symbols of the bonds of molecular pieces (private; protonet follow-ups 10 and 11).

A bond orbit B-C gets a symbol when all of these hold (decided from the net, the shapes and the
exact coordinates, never from element names):

1. it lies in a piece of the network that repeats in no direction (a molecule);
2. B and C each have four or fewer bonds;
3. it is not in a ring;
4. at each end at least one other bonded atom has more than one bond (not an end group);
5. the sorted pair of the shapes of B and C, as used in the label, is a key of ``_TORSION_EDGES``;
6. each end has a unique reference neighbour (A at B, D at C, below);
7. (decision D1) B, C, A and D sit on fully occupied, unmixed sites.

The candidates at B are B's other bonded atoms. The one with the most bonds is A; when several
share the most, the net breaks the tie: each tied candidate X has the signature (S1, S2, S3),
where Sd is the bond counts, sorted in descending order, of the atoms at graph distance exactly
d from X once B is removed, and the largest signature, compared as tuples (S1 first; a level that
is a prefix of another is the smaller), is A. At a planar (``TP3``) end a tie by bond count is
final; refinement applies at ``T4`` and ``TPY3`` ends only (protonet follow-up 11: refined planar
ends spread flat across the edge). A tie left gives no symbol. D is chosen likewise at C. Rules 1
to 6 read only the net and the shapes (``_net_judge``), so the label validator checks them with
the same code.

The symbol classifies the IUPAC torsion magnitude ``|A-B-C-D|`` (0 cis, 180 trans): below the
pair's edge the pair's low letter, from it ``t``; with A-B-C or B-C-D within 3 degrees of straight,
no symbol. The magnitude comes from dot products of the exact standard-setting bond vectors under
the group-averaged exact metric (as the shapes' angles), converted to a float once, so every exact
re-description and the mirror image give the same symbol.
"""

import math
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from httk.atomistic.symmetry.shapes import _averaged_metric, _bond_counts, _stabilisers

if TYPE_CHECKING:
    from httk.atomistic.models.species.species import Species
    from httk.atomistic.models.structure.asu import FundamentalDomainStructure
    from httk.atomistic.symmetry.protonet import _Element, _GroupTables

# sorted shape pair -> (edge in degrees, letter below it); ``t`` from the edge up
_TORSION_EDGES = {
    **dict.fromkeys((("T4", "T4"), ("T4", "TPY3"), ("TPY3", "TPY3"), ("A2", "A2")), (120.0, "g")),
    **dict.fromkeys((("TP3", "TP3"), ("A2", "TP3")), (90.0, "c")),
}
_STRAIGHT = math.sin(math.radians(3.0)) ** 2  # a squared sine at or below this is straight

type _Point = tuple[Fraction, Fraction, Fraction]
type _Atom = tuple[int, tuple[Fraction, ...]]  # (site, point modulo 1) of the translation quotient
type _Torsion = tuple[str | None, float | None, str | None]
type _Ends = tuple[int, _Point, _Point, _Point, int, _Point]  # (site of A, A, B, C, site of D, D)


def _edge(first: str | None, second: str | None) -> tuple[float, str] | None:
    """Return the ``(edge, low letter)`` of a bond's two shapes, in either order, or ``None``."""
    p, q = first or "", second or ""
    return _TORSION_EDGES.get((min(p, q), max(p, q)))


def _atom(site: int, point: Sequence[Fraction]) -> _Atom:
    """Return the translation-quotient node of the atom of ``site`` at ``point``."""
    return site, tuple(c % 1 for c in point)


def _whole(species: "Species") -> bool:
    """Return whether a species is one fully occupied constituent (decision D1)."""
    present = [c for s, c in zip(species.chemical_symbols, species.concentration, strict=True) if s != "vacancy"]
    return present == [1]


def _sign(x: Any) -> int:
    return (x > 0) - (x < 0) if isinstance(x, Fraction) else x.sign()


def _float(x: Any) -> float:
    """Convert an exact ``Fraction`` or surd scalar to a float (the one conversion, as in ``_angle``)."""
    if not isinstance(x, Fraction) and x.is_rational:
        x = x._rational_fraction()
    return float(x) if isinstance(x, Fraction) else x.to_float()


def _bond_torsions(structure: "FundamentalDomainStructure", shapes: Sequence[str | None]) -> list[_Torsion]:
    """Return the torsion of every bond orbit of a structure's network.

    :param structure: The structure; its bond orbits are ``structure.bonds.pairs``.
    :param shapes: The shape per Wyckoff site as used in the label (``_site_shapes``).
    :return: Per entry of ``structure.bonds.pairs``, ``(letter, |torsion| in degrees, None)`` or,
        without a symbol, ``(None, None, reason)``; empty without a network.
    """
    from httk.atomistic.models.structure.asu import _apply, _op, _reference, _standard_elements

    if structure.bonds is None:
        return []
    spacegroup, sites = structure.spacegroup, structure.wyckoff_sites
    elements = _standard_elements(spacegroup)
    tables = elements.tables
    rows = [(bond.site, bond.partner, elements.element(_op(bond.operation))) for bond in structure.bonds.pairs]
    refs = [(p[0], p[1], p[2]) for p in (_apply(elements.to_standard, _reference(spacegroup, site)) for site in sites)]
    judge = _net_judge(tables, _stabilisers(spacegroup, sites), refs, rows, shapes)
    species = {value.name: value for value in structure.species}
    metric: list[list[Any]] | None = None
    result: list[_Torsion] = []
    for a, b, g in rows:
        ends = judge(a, b, g)
        if isinstance(ends, str):
            result.append((None, None, ends))
            continue
        site_a, A, B, C, site_d, D = ends
        if not all(_whole(species[sites[s].species]) for s in (a, b, site_a, site_d)):
            result.append((None, None, "partial or mixed site"))
            continue
        if metric is None:
            metric = _averaged_metric(structure, tables)
        tau = _torsion(metric, A, B, C, D)
        if tau is None:
            result.append((None, None, "collinear"))
            continue
        edge = _edge(shapes[a], shapes[b])
        assert edge is not None  # the judge checked rule 5
        result.append((edge[1] if round(tau, 6) < edge[0] else "t", tau, None))
    return result


def _net_judge(
    tables: "_GroupTables",
    stabilisers: Sequence[Sequence["_Element"]],
    refs: Sequence[_Point],
    rows: Sequence[tuple[int, int, "_Element"]],
    shapes: Sequence[str | None],
) -> Callable[[int, int, "_Element"], _Ends | str]:
    """Return the judge of rules 1 to 6, which read only the net and the shapes.

    The graph is the translation quotient of the atoms grown from ``refs``: a node is a site and a
    point modulo 1, one node per atom modulo the lattice whenever ``stabilisers[s]`` is exactly the
    stabiliser of ``refs[s]``, and an edge carries the lattice translation between its atoms, whose
    sum around a cycle does not depend on which point of its Wyckoff flat each reference atom
    takes. So the structure's own reference atoms and the generic probe points of the validator
    give the same answers.

    :param tables: The standard setting's group tables.
    :param stabilisers: Per site, the elements fixing its reference atom (translations included).
    :param refs: Per site, its exact reference atom in standard coordinates.
    :param rows: The bonds ``(site, partner, element)`` as standard-setting elements.
    :param shapes: The shape per site as used in the label.
    :return: A function of one bond ``(site, partner, element)`` that returns
        ``(site of A, A, B, C, site of D, D)`` when the bond passes rules 1 to 6, else the reason it
        gets no symbol.
    """
    counts = _bond_counts(tables, stabilisers, rows)
    # the atoms bonded to each site's reference atom, in standard coordinates (as ``_baseline_shapes``)
    neighbours = [
        {
            (other, tables.apply(tables.mul(h, x), refs[other]))
            for a, b, g in rows
            for site, x, other in ((a, g, b), (b, tables.inv(g), a))
            if site == s
            for h in stabilisers[s]
        }
        for s in range(len(refs))
    ]
    # the translation quotient: every atom of the cell with its bonded atoms and their translations
    adjacency: dict[_Atom, set[tuple[_Atom, tuple[int, ...]]]] = {}
    for s, atoms in enumerate(neighbours):
        for k in range(tables.nops):
            x = (k, 0, 0, 0)
            p = tables.apply(x, refs[s])
            edges = adjacency.setdefault(_atom(s, p), set())
            for other, q in atoms:
                q = tables.apply(x, q)
                edges.add((_atom(other, q), tuple(math.floor(c) - math.floor(d) for c, d in zip(q, p))))
    finite: dict[_Atom, bool] = {}

    def molecular(start: _Atom) -> bool:
        """Return whether the piece of ``start`` repeats in no direction (rule 1)."""
        if start not in finite:
            offset: dict[_Atom, tuple[int, ...]] = {start: (0, 0, 0)}
            stack, ok = [start], True
            while stack:
                atom = stack.pop()
                for other, t in adjacency[atom]:
                    shifted = tuple(u + v for u, v in zip(offset[atom], t, strict=True))
                    if other not in offset:
                        offset[other] = shifted
                        stack.append(other)
                    elif offset[other] != shifted:
                        ok = False
            finite.update(dict.fromkeys(offset, ok))
        return finite[start]

    def ring(first: _Atom, second: _Atom) -> bool:
        """Return whether another path joins two bonded atoms of a finite piece (rule 3).

        A finite piece lifts to one copy of its quotient atoms, so translations play no part and
        the one edge between the two atoms is left out.
        """
        # one search per candidate bond, O(bonds x piece); Tarjan's bridge finding if large molecules make it matter
        seen, stack = {first}, [first]
        while stack:
            atom = stack.pop()
            for other, _t in adjacency[atom]:
                if atom == first and other == second:
                    continue
                if other == second:
                    return True
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        return False

    def signature(start: _Atom, removed: _Atom) -> tuple[tuple[int, ...], ...]:
        """Return ``(S1, S2, S3)``: the descending bond counts at each distance from ``start`` without ``removed``."""
        seen, level, result = {removed, start}, [start], []
        for _ in range(3):
            level = [other for atom in level for other, _t in adjacency[atom] if other not in seen]
            level = list(dict.fromkeys(level))  # an atom two paths reach counts once
            seen.update(level)
            result.append(tuple(sorted((counts[site] for site, _ in level), reverse=True)))
        return tuple(result)

    def reference(candidates: list[tuple[int, _Point]], end: _Atom) -> tuple[int, _Point] | None:
        """Return the unique reference neighbour among an end's candidates (rule 6), or ``None`` on a tie.

        At a planar (``TP3``) end a tie by bond count is final; the signatures refine it only at
        ``T4`` and ``TPY3`` ends.
        """
        top = max(counts[o] for o, _ in candidates)
        tied = [(o, P) for o, P in candidates if counts[o] == top]
        if len(tied) > 1 and shapes[end[0]] != "TP3":
            keys = [signature(_atom(o, P), end) for o, P in tied]
            tied = [x for x, key in zip(tied, keys, strict=True) if key == max(keys)]
        return tied[0] if len(tied) == 1 else None

    def judge(a: int, b: int, g: "_Element") -> _Ends | str:
        B, C = refs[a], tables.apply(g, refs[b])
        start, partner = _atom(a, B), _atom(b, C)
        if not molecular(start):
            return "periodic piece"
        if counts[a] > 4 or counts[b] > 4:  # the brief's rule 2; rule 5 implies it for validated shapes
            return "more than four bonds"
        before = [(o, p) for o, p in neighbours[a] if (o, p) != (b, C)]
        after = [(o, q) for o, q in ((o, tables.apply(g, p)) for o, p in neighbours[b]) if (o, q) != (a, B)]
        if not before or not after:
            return "terminal atom"
        if 1 in (max(counts[o] for o, _ in before), max(counts[o] for o, _ in after)):
            return "end group"
        if ring(start, partner):
            return "ring bond"
        if _edge(shapes[a], shapes[b]) is None:
            return "shape pair without a letter"
        first, second = reference(before, start), reference(after, partner)
        if first is None or second is None:
            return "tied reference neighbour"
        return first[0], first[1], B, C, second[0], second[1]

    return judge


def _torsion(metric: Sequence[Sequence[Any]], A: _Point, B: _Point, C: _Point, D: _Point) -> float | None:
    """Return ``|A-B-C-D|`` in degrees from exact points, or ``None`` when an end is within 3 degrees of straight."""

    def form(x: Sequence[Fraction], y: Sequence[Fraction]) -> Any:
        return sum((x[i] * metric[i][j] * y[j] for i in range(3) for j in range(3)), Fraction(0))

    b0, b1, b2 = (tuple(q - p for p, q in zip(u, v, strict=True)) for u, v in ((A, B), (B, C), (C, D)))
    d00, d11, d22 = form(b0, b0), form(b1, b1), form(b2, b2)
    d01, d02, d12 = form(b0, b1), form(b0, b2), form(b1, b2)
    if not (_sign(d00) and _sign(d11) and _sign(d22)):
        return None  # a zero-length bond has no direction
    first, second = d00 * d11 - d01 * d01, d11 * d22 - d12 * d12  # |b0 x b1|^2, |b1 x b2|^2
    if _float(first / (d00 * d11)) <= _STRAIGHT or _float(second / (d11 * d22)) <= _STRAIGHT:
        return None
    numerator = d01 * d12 - d02 * d11
    cosine = math.copysign(math.sqrt(_float(numerator * numerator / (first * second))), _sign(numerator))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _site_pair_torsions(
    structure: "FundamentalDomainStructure", shapes: Sequence[str | None]
) -> dict[tuple[int, int], str]:
    """Return the torsion letter of every qualifying bond, keyed by its Wyckoff-site pair.

    In a finite piece, two bond orbits between one site pair (or one site and itself) are both
    ring bonds: every atom of the pair's sites then has a bond of each orbit, and a bridge orbit
    would make some piece of the other orbit's bonds a single atom. So at most one orbit per site
    pair has a letter, and the letter is a function of the unordered site pair.

    :param structure: A fundamental domain with a bond network, as ``Protonetstructure.from_asu``
        reads it (its re-described ``structure``).
    :param shapes: The shape per entry of ``structure.wyckoff_sites``, as used in the label
        (``_site_shapes(structure)``).
    :return: ``{(i, j): letter}`` with ``i <= j`` the 0-based indices into
        ``structure.wyckoff_sites``, holding only the qualifying pairs; empty without a network.
    """
    result: dict[tuple[int, int], str] = {}
    pairs = () if structure.bonds is None else structure.bonds.pairs
    for bond, (letter, _tau, _reason) in zip(pairs, _bond_torsions(structure, shapes), strict=True):
        if letter is not None:
            pair = (min(bond.site, bond.partner), max(bond.site, bond.partner))
            if pair in result:
                raise RuntimeError(f"two qualifying bond orbits on the site pair {pair}; they must be ring bonds")
            result[pair] = letter
    return result
