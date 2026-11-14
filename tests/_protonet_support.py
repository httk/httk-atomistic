"""Test-only helpers for the bonding-net canonicaliser, ported from ``protonet_reference_check.py``.

Floats are used only here, to find bonds by distance and to verify decoded bonds geometrically.
The helpers accept either the oracle's ``GroupTables`` (dict-valued letters and cosets) or the
production ``_GroupTables``.
"""

import itertools
import math
import random
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from fractions import Fraction
from typing import Any, cast

import httk.core

from httk.atomistic.symmetry.spacegroup import Spacegroup

type Site = tuple[Hashable, str]
type Element = tuple[int, int, int, int]
type Row = tuple[int, int, Element]
type Lattice = list[list[float]]


def letter_field(tables: Any, letter: str, name: str) -> Any:
    """Return one per-letter field from either table implementation."""
    info = tables.letter(letter)
    return info[name] if isinstance(info, dict) else getattr(info, name)


def moves(tables: Any, letter: str) -> list[Element]:
    """Return a letter's gauge moves from either table implementation."""
    info = tables.letter(letter)
    return info["moves"] if isinstance(info, dict) else tables.moves(letter)


def reps(tables: Any, letter: str) -> list[Element]:
    """Return a letter's gauge-move classes modulo translations along its flat, from either implementation."""
    info = tables.letter(letter)
    return info["reps"] if isinstance(info, dict) else tables.reps(letter)


def coset_letter(coset: Any, letter: str) -> tuple[str, Element]:
    """Return ``(target letter, b)`` from either coset implementation."""
    return coset["letters"][letter] if isinstance(coset, dict) else coset.letter(letter)


def coset_action(coset: Any) -> Any:
    """Return a coset's affine action from either implementation."""
    return coset["action"] if isinstance(coset, dict) else coset.action


def cart_len(lattice: Lattice, d: Sequence[float]) -> float:
    """Cartesian length of a reduced difference vector."""
    v = [sum(d[i] * lattice[i][c] for i in range(3)) for c in range(3)]
    return math.sqrt(sum(x * x for x in v))


def fapply(tables: Any, e: Element, p: Sequence[float]) -> tuple[float, float, float]:
    """Apply an element to a float point."""
    W, w = tables.W[e[0]], tables.w[e[0]]
    x, y, z = (sum(float(W[r][c]) * p[c] for c in range(3)) + float(w[r]) + e[1 + r] for r in range(3))
    return x, y, z


def raw_rows(
    tables: Any,
    sites: Sequence[Site],
    refs: Sequence[Sequence[float]],
    lattice: Lattice,
    cutoffs: Mapping[Any, float],
    reach: int = 2,
) -> tuple[list[Row], dict[Row, float]]:
    """Find rows by a distance cutoff keyed by ``frozenset`` of class ids; returns rows and distances.

    Neighbours are looked for among the translations with entries up to ``reach``.
    """
    rows: dict[Row, float] = {}
    for i, (si, li) in enumerate(sites):
        for j, (sj, lj) in enumerate(sites):
            rmax = cutoffs.get(frozenset((si, sj)))
            if rmax is None or j < i:
                continue
            Hi, Hj = letter_field(tables, li, "H"), letter_field(tables, lj, "H")
            for k in range(tables.nops):
                for n in itertools.product(range(-reach, reach + 1), repeat=3):
                    e = (k, n[0], n[1], n[2])
                    q = fapply(tables, e, refs[j])
                    d = cart_len(lattice, [a - b for a, b in zip(q, refs[i])])
                    if 1e-6 < d < rmax:
                        rows[(i, j, tables.canon(Hi, Hj, e, i == j))] = d
    return sorted(rows), rows


def keeps_cell(coset: Any) -> bool:
    """Whether a coset's matrix is a signed permutation, so its action keeps the cell as a lattice."""
    P = coset["P"] if isinstance(coset, dict) else coset.P
    rows = [P[3 * r : 3 * r + 3] for r in range(3)]
    return all(sorted(map(abs, line)) == [0, 0, 1] for line in (*rows, *zip(*rows, strict=True)))


def scramble(
    tables: Any,
    sites: Sequence[Site],
    rows: Sequence[Row],
    rng: random.Random,
    change_description: bool = True,
    coset: Any = None,
    keep_hand: bool = True,
) -> tuple[list[Site], list[Row]]:
    """Same net, other description: random coset, reference atoms, site/row order and row direction.

    ``change_description=False`` keeps the identity coset, and ``coset`` imposes one. In triclinic
    groups the random coset keeps the cell (a signed permutation matrix): there periodic nets are
    canonical only relative to the cell (D7). With ``keep_hand`` the random coset of a Sohncke group
    is proper: an improper one describes the mirror image, which the kept hand tells apart.
    """
    if coset is None:
        cosets = tables.cosets if tables.spacegroup.it_number > 2 else [c for c in tables.cosets if keeps_cell(c)]
        if keep_hand and all(operation.determinant() > 0 for operation in tables.spacegroup.symmetry_operations):
            cosets = [c for c in cosets if coset_action(c).determinant() > 0]
        coset = rng.choice(cosets) if change_description else tables.cosets[0]
    letters = [coset_letter(coset, letter)[0] for _, letter in sites]
    rows = [
        (
            a,
            b,
            tables.mul(
                tables.mul(coset_letter(coset, sites[a][1])[1], tables.conjugate(coset, g)),
                tables.inv(coset_letter(coset, sites[b][1])[1]),
            ),
        )
        for a, b, g in rows
    ]
    gauge = []
    for letter in letters:
        if letter_field(tables, letter, "general"):
            m = (rng.randrange(tables.nops), rng.randint(-3, 3), rng.randint(-3, 3), rng.randint(-3, 3))
        else:
            m = rng.choice(moves(tables, letter))
        gauge.append(m)
    rows = [(a, b, tables.mul(tables.mul(gauge[a], g), tables.inv(gauge[b]))) for a, b, g in rows]
    perm = list(range(len(sites)))
    rng.shuffle(perm)
    new_index = {old: new for new, old in enumerate(perm)}
    sites2 = [(sites[old][0], letters[old]) for old in perm]
    rows2 = []
    for a, b, g in rows:
        if rng.random() < 0.5 and a != b:
            a, b, g = b, a, tables.inv(g)
        rows2.append((new_index[a], new_index[b], g))
    rng.shuffle(rows2)
    return sites2, rows2


def scramble_shaped(
    tables: Any, sites: Sequence[Site], rows: Sequence[Row], shapes: Sequence[str | None], rng: random.Random, **kw: Any
) -> tuple[list[Site], list[Row], list[str | None]]:
    """:func:`scramble` with per-site shapes carried along with their sites."""
    tagged: list[Site] = [((cls, s), letter) for s, (cls, letter) in enumerate(sites)]
    moved, rows2 = scramble(tables, tagged, rows, rng, **kw)
    tags = [cast(tuple[Hashable, int], tag) for tag, _ in moved]
    return (
        [(cls, letter) for (cls, _), (_, letter) in zip(tags, moved, strict=True)],
        rows2,
        [shapes[s] for _, s in tags],
    )


def random_shapes(tables: Any, sites: Sequence[Site], rows: Sequence[Row], rng: random.Random) -> list[str | None]:
    """A random valid shape per site of a 0-based net: any table symbol with its bond count, or ``I<n>`` for ``n > 2``."""
    from httk.atomistic.symmetry.shapes import _KNOWN, _bond_count, _bond_counts

    counts = _bond_counts(tables, [letter_field(tables, letter, "H") for _, letter in sites], rows)
    return [
        None if n < 2 else rng.choice([s for s in sorted(_KNOWN) if _bond_count(s) == n] + [f"I{n}"] * (n > 2))
        for n in counts
    ]


def random_net(tables: Any, rng: random.Random) -> tuple[list[Site], list[Row]]:
    """An arbitrary abstract net: random sites and random rows.  Any set of rows is a valid net."""
    wy = tables.spacegroup.wyckoff
    fixed = {w.letter for w in wy if w.free_count == 0}
    sites: list[Site] = []
    used = set()
    for _ in range(rng.randint(2, 4)):
        letter = rng.choice(wy).letter
        if letter in fixed and letter in used:
            continue
        used.add(letter)
        sites.append((rng.choice("XYZ"[: rng.randint(1, 3)]), letter))
    rows = set()
    for _ in range(rng.randint(2, 6)):
        i, j = sorted((rng.randrange(len(sites)), rng.randrange(len(sites))))
        g = (rng.randrange(tables.nops), rng.randint(-1, 1), rng.randint(-1, 1), rng.randint(-1, 1))
        c = tables.canon(letter_field(tables, sites[i][1], "H"), letter_field(tables, sites[j][1], "H"), g, i == j)
        if not (i == j and c == (0, 0, 0, 0)):  # skip an atom bonded to itself
            rows.add((i, j, c))
    return sites, sorted(rows)


def synthetic_sweep(
    tables: Any,
    net: Callable[[Sequence[Site], Sequence[Row]], Hashable],
    rng: random.Random,
    nets: int = 12,
    scrambles: int = 3,
    change_description: bool = True,
    keep_hand: bool = True,
) -> int:
    """Invariance on exactly ``nets`` random nets of one group; returns how many were not invariant.

    ``net`` maps ``(sites, rows)`` to a comparable canonical result.  A drawn net without rows is
    drawn again, at most 100 times per net. ``keep_hand`` is passed to :func:`scramble`.
    """
    bad = 0
    for _ in range(nets):
        for _ in range(100):
            sites, rows = random_net(tables, rng)
            if rows:
                break
        else:
            raise AssertionError("random_net drew 100 nets without rows")
        seen = {net(sites, rows)}
        for _ in range(scrambles):
            seen.add(net(*scramble(tables, sites, rows, rng, change_description, keep_hand=keep_hand)))
        bad += len(seen) > 1
    return bad


def reference_point(it_number: int, letter: str, params: Any) -> tuple[float, float, float]:
    """Float reference atom: the first-listed Wyckoff coordinate, not reduced modulo one."""
    point = Spacegroup.standard(it_number).wyckoff_position(letter).representative.coordinate(params)
    x, y, z = (float(v) for v in point.to_fractions())
    return x, y, z


def lattice_from_parameters(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> Lattice:
    """Row lattice vectors from cell parameters (degrees)."""
    ca, cb, cg = (math.cos(math.radians(x)) for x in (alpha, beta, gamma))
    sg = math.sin(math.radians(gamma))
    lattice = [[a, 0.0, 0.0], [b * cg, b * sg, 0.0], [c * cb, c * (ca - cb * cg) / sg, 0.0]]
    lattice[2][2] = math.sqrt(max(c * c - lattice[2][0] ** 2 - lattice[2][1] ** 2, 0.0))
    return lattice


def handmade(
    it_number: int, spec: Sequence[tuple[str, str, tuple[Fraction, ...]]], cell: Sequence[float]
) -> tuple[list[Site], list[tuple[float, float, float]], Lattice]:
    """Build ``(sites, refs, lattice)`` from ``[(species, letter, free parameters)]`` and cell parameters."""
    sites: list[Site] = [(species, letter) for species, letter, _ in spec]
    refs = [reference_point(it_number, letter, params) for _, letter, params in spec]
    a, b, c, alpha, beta, gamma = cell
    return sites, refs, lattice_from_parameters(a, b, c, alpha, beta, gamma)


def load_structure(path: str) -> tuple[int, list[Site], list[tuple[float, float, float]], Lattice] | None:
    """Load a CIF as ``(it_number, sites keyed by element, refs, lattice)``.

    Returns ``None`` unless it is ordered, element-pure and in its standard setting.
    """
    asu = httk.core.load(path)
    it_number = asu.space_group_it_number
    if asu.spacegroup != Spacegroup.standard(it_number):
        return None
    element = {}
    for species in asu.species:
        if len(species.chemical_symbols) != 1 or tuple(species.concentration) != (1,):
            return None
        element[species.name] = species.chemical_symbols[0]
    sites: list[Site] = [(element[ws.species], ws.wyckoff) for ws in asu.wyckoff_sites]
    refs = [reference_point(it_number, ws.wyckoff, ws.free_params) for ws in asu.wyckoff_sites]
    lattice = [[float(x) for x in row] for row in asu.lattice_vectors]
    return it_number, sites, refs, lattice


def geometric_round_trip(
    tables: Any,
    sites: Sequence[Site],
    refs: Sequence[Sequence[float]],
    lattice: Lattice,
    dist: Mapping[Row, float],
    slots: Sequence[tuple[int, int]],
    codes: Sequence[Element],
    coset: Any,
    number: Mapping[int, int],
    gauge: Mapping[int, Element | None],
    cell: Any = None,
) -> bool:
    """Lay the canonical net back onto the input geometry through one realisation and compare bonds.

    ``cell`` is the realisation's compiled cell chosen from the net, applied before ``coset``.
    """
    act = coset_action(coset)
    inv = (act if cell is None else act * coset_action(cell)).inverse()

    def f(op: Any, p: Sequence[float]) -> list[float]:
        return [float(x) for x in op.apply([Fraction(x).limit_denominator(10**9) for x in p]).to_fractions()]

    site_of = {n: s for s, n in number.items()}
    canon_ref = {}
    for s, (_, letter) in enumerate(sites):
        ref = refs[s]
        if cell is not None:
            letter, b = coset_letter(cell, letter)
            ref = fapply(tables, b, f(coset_action(cell), ref))
        moved = fapply(tables, coset_letter(coset, letter)[1], f(act, ref))
        g = gauge.get(s)
        canon_ref[s] = fapply(tables, g, moved) if g is not None else moved
    got: Counter[tuple[frozenset[Hashable], float]] = Counter()
    for (i, j), code in zip(slots, codes, strict=True):
        a, b = site_of[i], site_of[j]
        p0, q0 = f(inv, canon_ref[a]), f(inv, fapply(tables, code, canon_ref[b]))
        got[(frozenset((sites[a][0], sites[b][0])), round(cart_len(lattice, [x - y for x, y in zip(q0, p0)]), 3))] += 1
    want = Counter((frozenset((sites[i][0], sites[j][0])), round(d, 3)) for (i, j, _), d in dist.items())
    return got == want


def generic_cutoffs(
    tables: Any, sites: Sequence[Site], refs: Sequence[Sequence[float]], lattice: Lattice, factor: float = 1.15
) -> dict[frozenset[Hashable], float]:
    """Cutoff for every class pair: ``factor`` times the structure's shortest interatomic distance.

    httk has no covalent-radius table, so this stands in for a chemistry-aware bond finder.
    """
    classes = {cls for cls, _ in sites}
    probe = {frozenset((a, b)): 6.0 for a in classes for b in classes}
    _, dist = raw_rows(tables, sites, refs, lattice, probe)
    shortest = min(dist.values())
    return dict.fromkeys(probe, factor * shortest)


def default_shapes(it_number: int, letters: Sequence[str], rows: Sequence[Sequence[int]]) -> list[str | None]:
    """The default shape of every site of a 1-based net (``I<n>`` where its bond count has none)."""
    from httk.atomistic.models.protonettype.protonettype import _net_counts
    from httk.atomistic.symmetry.shapes import _DEFAULTS

    counts = _net_counts(Spacegroup.standard(it_number), letters, rows, "default_shapes")
    return [None if count < 2 else _DEFAULTS.get(count, f"I{count}") for count in counts]
