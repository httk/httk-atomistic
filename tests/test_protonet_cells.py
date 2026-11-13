"""Monoclinic bonding nets choose their cell from the net (task 4.2).

The construction and its proofs are in the workspace design note
``docs/plans/PROTONET_CELL_FROM_NET_DESIGN.md``; the test numbers below are its section 9.
"""

import random
from collections.abc import Callable, Sequence
from fractions import Fraction as F
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any

import httk.core
import protonet_reference as oracle
import pytest
from _protonet_support import letter_field, load_structure, raw_rows, reference_point, scramble
from httk.core import FracVector

from httk.atomistic import ASUStructure, Cell, Species, WyckoffSite
from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
from httk.atomistic.models.prototype.notation import class_order_key
from httk.atomistic.symmetry import _protonet_cells as cells
from httk.atomistic.symmetry import protonet
from httk.atomistic.symmetry._wyckoff_actions import compile_wyckoff_action
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.canonical_classification import _signature, canonical_protonetstructure
from httk.atomistic.symmetry.protonet import _canonical_net, _cell, _key, _moved, _tables
from httk.atomistic.symmetry.spacegroup import Spacegroup

MONOCLINIC = tuple(range(3, 16))
IDENTITY = (0, 0, 0, 0)
UNIT = ((1, 0), (0, 1))

type Matrix = tuple[tuple[int, int], tuple[int, int]]


def _profile(normal: Sequence[Any], extended: Sequence[Any]) -> list[Any]:
    return [*normal, *(pytest.param(*value, marks=pytest.mark.extended) for value in extended)]


def _small(bound: int) -> list[Matrix]:
    return [((a, b), (c, d)) for a, b, c, d in product(range(-bound, bound + 1), repeat=4) if a * d - b * c in (1, -1)]


def _action(X: Matrix) -> AffineOperation:
    (a, b), (c, d) = X
    return AffineOperation(((a, 0, b), (0, 1, 0), (c, 0, d)))


def _move(it: int, sites: Sequence[Any], rows: Sequence[Any], X: Matrix) -> tuple[list[Any], list[Any]]:
    """The same net described in the cell ``X``."""
    tables, cell = _tables(it), _cell(it, X)
    return [(cls, cell.letter(letter)[0]) for cls, letter in sites], [_moved(tables, cell, sites, row) for row in rows]


def _net(it: int, sites: Sequence[Any], rows: Sequence[Any]) -> tuple[Any, ...]:
    result = _canonical_net(it, sites, rows)
    return result.groups, result.slots, result.codes


def _generators(it: int) -> list[Matrix]:
    return [X for X in _small(2) if X != UNIT and cells._in_gamma(it, X)]


def _random_cell(it: int, rng: random.Random) -> Matrix:
    """A random product of one to three cell changes with entries up to 2."""
    X = UNIT
    for _ in range(rng.randint(1, 3)):
        X = cells._mul(rng.choice(_generators(it)), X)
    return X


def _kinds(it: int, rng: random.Random) -> dict[str, Callable[[], list[Any]]]:
    """The prototype's net kinds on three general sites, plus components periodic along b."""
    tables = _tables(it)
    centres = [k for k in range(tables.nops) if tables.Wi[k] == (-1, 0, 0, 0, -1, 0, 0, 0, -1)]
    rotations = [k for k in range(tables.nops) if tables.Wi[k] == (-1, 0, 0, 0, 1, 0, 0, 0, -1)]
    molecule = [(0, 1, IDENTITY), (1, 2, IDENTITY)]

    def shift() -> tuple[int, int, int]:
        return rng.randint(-1, 1), rng.randint(-1, 1), rng.randint(-1, 1)

    def helix() -> tuple[int, int, int, int]:
        while True:  # a 2-fold or 2_1 element whose square is a translation along b
            e = (rng.choice(rotations), *shift())
            square = tables.mul(e, e)
            if square[0] == 0 and square[2] and not square[1] and not square[3]:
                return e

    kinds: dict[str, Callable[[], list[Any]]] = {
        "molecule": lambda: molecule,
        "chain along a": lambda: [*molecule, (0, 0, (0, 1, 0, 0))],
        "chain along b": lambda: [*molecule, (0, 0, (0, 0, 1, 0))],
        "layer in the ac plane": lambda: [*molecule, (0, 0, (0, 1, 0, 0)), (2, 2, (0, 0, 0, 1))],
    }
    if centres:
        kinds["molecule on an inversion centre"] = lambda: [*molecule, (0, 0, (rng.choice(centres), *shift()))]
    if rotations:
        kinds["two helices along b"] = lambda: [(0, 0, helix()), (1, 1, helix()), (1, 2, IDENTITY)]
        kinds["helix and molecule on an axis"] = lambda: [
            (0, 0, helix()),
            (1, 2, IDENTITY),
            (2, 2, (rng.choice(rotations), *shift())),
        ]
        kinds["layer containing b"] = lambda: [(0, 0, helix()), (0, 0, (0, 1, 0, 0)), (1, 1, helix()), (1, 2, IDENTITY)]
        kinds["3-periodic"] = lambda: [
            (0, 0, (0, 1, 0, 0)),
            (0, 0, helix()),
            (0, 1, IDENTITY),
            (1, 1, (0, 0, 0, 1)),
            (1, 2, IDENTITY),
        ]
    return kinds


def harness(it: int, nets: int, moves: int, seed: str) -> dict[str, int]:
    """Per kind, how many random cell changes of scrambled nets changed the canonical net."""
    tables, rng = _tables(it), random.Random(seed)
    general = max(tables.spacegroup.wyckoff, key=lambda w: (w.free_count, w.multiplicity)).letter
    sites = [("X", general), ("Y", general), ("Z", general)]
    changed = {}
    for name, make in _kinds(it, rng).items():
        changed[name] = 0
        for _ in range(nets):
            sites0, rows0 = scramble(tables, sites, make(), rng)
            expected = _net(it, sites0, rows0)
            for _ in range(moves):
                changed[name] += _net(it, *_move(it, sites0, rows0, _random_cell(it, rng))) != expected
    return changed


# ---------------------------------------------------------------------------------------------
# (1), (14): the prototype's harness, with every optimisation in place
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "it,nets,moves", _profile([(it, 1, 2) for it in MONOCLINIC], [(it, 5, 4) for it in MONOCLINIC])
)
def test_nets_do_not_depend_on_the_cell(it: int, nets: int, moves: int) -> None:
    changed = harness(it, nets, moves, f"cells-{it}-{nets}")
    assert changed == dict.fromkeys(changed, 0)


# ---------------------------------------------------------------------------------------------
# (2): the counterexamples of the design note, before and after their cell change
# ---------------------------------------------------------------------------------------------
def _net_e_only(monkeypatch: pytest.MonkeyPatch, it: int, sites: Sequence[Any], rows: Sequence[Any]) -> Any:
    """The 4.1 net: the equivalent descriptions alone, without the cells chosen from the net."""
    with monkeypatch.context() as patch:
        patch.setattr(protonet, "_cell_candidates", lambda *_: ((UNIT,), False))
        protonet._search.cache_clear()
        net = _net(it, sites, rows)
    protonet._search.cache_clear()
    return net


COUNTEREXAMPLES = {
    # P2_1: two (b,c) layers, X around the 2_1 axes at x = 0 and Y around those at x = 1/2 (rank 1)
    "p21-layers": (
        4,
        ["a", "a"],
        [(0, 0, (1, 0, 0, 0)), (0, 0, (0, 0, 0, 1)), (1, 1, (1, 1, 0, 0)), (1, 1, (0, 0, 0, 1))],
        ((1, 0), (1, 1)),
    ),
    # P2_1: two helices along b on the two classes of 2_1 axes (rank 0)
    "p21-helices": (4, ["a", "a"], [(0, 0, (1, 0, 0, 0)), (1, 1, (1, 1, 0, 0))], ((1, 0), (1, 1))),
    # C2: a helix on the 2_1 axis at x = 1/4 and a pair across the 2-fold axis at the origin (rank 0)
    "c2-helix-pair": (5, ["c", "c"], [(0, 0, (2, 0, 0, 0)), (1, 1, (1, 0, 0, 0))], ((-1, -1), (-2, -1))),
    # P2_1/c: a helix on the 2_1 axis at (0, y, 1/4) and a dimer on the origin centre (rank 0, d = 4)
    "p21c-helix-dimer": (14, ["e", "e"], [(0, 0, (2, 0, 0, 0)), (1, 1, (1, 0, 0, 0))], ((1, 2), (0, 1))),
    # P2_1/c: an (a,b) layer pinned by that 2_1 axis and the dimer (rank 1, d = 4)
    "p21c-layer-dimer": (
        14,
        ["e", "e"],
        [(0, 0, (2, 0, 0, 0)), (0, 0, (0, 1, 0, 0)), (1, 1, (1, 0, 0, 0))],
        ((1, 2), (0, 1)),
    ),
}


@pytest.mark.parametrize("name", list(COUNTEREXAMPLES))
def test_counterexamples_of_the_e_only_search_keep_their_net(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    it, letters, rows, X = COUNTEREXAMPLES[name]
    sites = [(cls, letter) for cls, letter in zip("XY", letters, strict=True)]
    moved = _move(it, sites, rows, X)
    assert _net_e_only(monkeypatch, it, sites, rows) != _net_e_only(monkeypatch, it, *moved)
    assert _net(it, sites, rows) == _net(it, *moved)


@pytest.mark.parametrize("X", [((1, 2), (0, 1)), ((1, 0), (1, 1))])
def test_rank_one_p21c_needs_anchor_level_four(X: Matrix, monkeypatch: pytest.MonkeyPatch) -> None:
    # the reviewer's case: shear representatives modulo 2 are not a complete transversal here
    it, letters, rows, _ = COUNTEREXAMPLES["p21c-layer-dimer"]
    sites = [("X", letters[0]), ("Y", letters[1])]
    moved = _move(it, sites, rows, X)
    assert _net(it, sites, rows) == _net(it, *moved)
    with monkeypatch.context() as patch:
        patch.setitem(cells._ANCHOR_LEVEL, 14, 2)
        protonet._search.cache_clear()
        assert _net(it, sites, rows) != _net(it, *moved)
    protonet._search.cache_clear()


def _search_key(it: int, result: Any) -> tuple[Any, ...]:
    """The search's key of a canonical net: bare key, pair list, code keys."""
    tables = _tables(it)
    ranks = sorted(class_order_key(g, sum(tables.letter(x).multiplicity for x in g))[:2] for g in result.groups)
    return tuple(ranks), tuple(result.slots), tuple(_key(c) for c in result.codes)


@pytest.mark.parametrize("it", _profile([4, 14], [(it,) for it in MONOCLINIC if it not in (4, 14)]))
def test_rank_zero_cells_reach_the_least_e_only_key_over_a_box(it: int, monkeypatch: pytest.MonkeyPatch) -> None:
    # independent of the transversal: the least E-only key over the Γ cells of a box that meets every
    # class modulo d (entries up to 2, or 3 where d = 4) is the production key of a b-periodic net
    tables, rng = _tables(it), random.Random(f"rank0-box-{it}")
    general = max(tables.spacegroup.wyckoff, key=lambda w: (w.free_count, w.multiplicity)).letter
    sites = [("X", general), ("Y", general), ("Z", general)]
    kinds = _kinds(it, rng)
    rows = (kinds.get("two helices along b") or kinds["chain along b"])()
    sites0, rows0 = scramble(tables, sites, rows, rng)
    G, gbb = cells._bond_gram(tables, sites0, rows0)
    assert G == ((0, 0), (0, 0)) and gbb > 0
    expected = _search_key(it, _canonical_net(it, sites0, rows0))
    d = cells._ANCHOR_LEVEL[it]
    box = [X for X in _small(3 if d == 4 else 2) if cells._in_gamma(it, X)]
    assert len({tuple(x % d for row in X for x in row) for X in box}) == len(cells._congruence_cells(it, d)) * (
        2 if d == 4 else 1
    )
    with monkeypatch.context() as patch:
        patch.setattr(protonet, "_cell_candidates", lambda *_: ((UNIT,), False))
        protonet._search.cache_clear()
        least = min(_search_key(it, _canonical_net(it, *_move(it, sites0, rows0, X))) for X in box)
    protonet._search.cache_clear()
    assert least == expected


# ---------------------------------------------------------------------------------------------
# (3): Γ, (4) and (10): the enumerations against brute force
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("it,bound", _profile([(it, 2) for it in MONOCLINIC], [(it, 3) for it in MONOCLINIC]))
def test_gamma_is_the_set_of_cell_changes_mapping_the_group_onto_itself(it: int, bound: int) -> None:
    group = _tables(it).spacegroup
    for X in _small(bound):
        assert cells._in_gamma(it, X) == (_signature(group, _action(X)) is not None), X


def test_congruence_cells_are_a_transversal() -> None:
    # |Γ mod d| / 2 classes modulo ±I: GL2(Z/2) has 6 elements and -I = I there; GL2(Z/4) has 96
    sizes = {it: len(cells._congruence_cells(it, cells._ANCHOR_LEVEL[it])) for it in MONOCLINIC}
    assert sizes == {3: 6, 4: 6, 5: 16, 6: 1, 7: 1, 8: 1, 9: 1, 10: 6, 11: 6, 12: 16, 13: 16, 14: 16, 15: 8}
    for it in MONOCLINIC:
        assert all(cells._in_gamma(it, X) for X in cells._congruence_cells(it, cells._ANCHOR_LEVEL[it]))


def _form_value(G: Any, v: tuple[int, int]) -> Any:
    return G[0][0] * v[0] * v[0] + 2 * G[0][1] * v[0] * v[1] + G[1][1] * v[1] * v[1]


def _forms(count: int, rng: random.Random) -> list[Any]:
    """Square, hexagonal and rectangular forms, then random definite ones in random bases."""
    special = [((1, 0), (0, 1)), ((1, F(1, 2)), (F(1, 2), 1)), ((1, F(-1, 2)), (F(-1, 2), 1)), ((2, 0), (0, 5))]
    out = [((F(a), F(b)), (F(c), F(d))) for (a, b), (c, d) in special]
    out += [((F(2), F(1)), (F(1), F(2))), ((F(4), F(2)), (F(2), F(4)))]
    while len(out) < count:
        a, b, c = (F(rng.randint(1, 30), rng.randint(1, 6)) for _ in range(3))
        b *= rng.choice((1, -1))
        if a * c > b * b:
            out.append(cells._form(rng.choice(_small(3)), ((a, b), (b, c))))
    return out


@pytest.mark.parametrize("count", [8, pytest.param(46, marks=pytest.mark.extended)])
def test_reduced_cells_equal_a_brute_force(count: int) -> None:
    # with X = M Q0, the first row of M has entries in -1..1 and |j - j*| <= 2 with |j*| <= 3/2 for
    # the second, so every reduced cell has M entries in -5..5
    box = _small(5)
    for G in _forms(count, random.Random(5)):
        Q0 = cells._lagrange(G)
        F0 = cells._form(Q0, G)
        assert 2 * abs(F0[0][1]) <= F0[0][0] <= F0[1][1]
        for it in MONOCLINIC:
            keyed = [
                (cells._form_key(cells._form(X, G)), X) for M in box if cells._in_gamma(it, X := cells._mul(M, Q0))
            ]
            least = min(key for key, _ in keyed)
            expected = sorted({X for key, X in keyed if key == least})
            found = cells._reduced_cells(it, G)
            assert sorted(found) == expected and len(found) <= 12


# the design note's t*: per group, the class of t modulo 2 -> t*
_GL = {(1, 0): (0, 1), (0, 1): (0, 1), (1, 1): (0, 1)}
_C_EVEN = {(1, 0): (1, 0), (0, 1): (0, 1), (1, 1): (0, 1)}
_B_EVEN = {(1, 0): (1, 0), (0, 1): (0, 1), (1, 1): (1, 0)}
_BOTH = {(1, 0): (1, 0), (0, 1): (0, 1), (1, 1): (1, -1)}
T_STAR = {3: _GL, 4: _GL, 6: _GL, 10: _GL, 11: _GL, 5: _C_EVEN, 8: _C_EVEN, 12: _C_EVEN}
T_STAR |= {7: _B_EVEN, 13: _B_EVEN, 14: _B_EVEN, 9: _BOTH, 15: _BOTH}


def _image(X: Matrix, t: tuple[int, int]) -> tuple[int, int]:
    return X[0][0] * t[0] + X[0][1] * t[1], X[1][0] * t[0] + X[1][1] * t[1]


@pytest.mark.parametrize("it", MONOCLINIC)
def test_rank_one_cells_reach_t_star_and_cover_every_reduced_cell(it: int) -> None:
    d = cells._ANCHOR_LEVEL[it]
    for t in [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (3, -2), (-2, 5)]:
        star = T_STAR[it][t[0] % 2, t[1] % 2]
        found = cells._rank1_cells(it, t)
        assert len(found) <= 2 * d and {_image(X, t) for X in found} == {star}
        # every cell of Γ (entries up to 4) mapping t onto ±t* is ±Y Z for a found Z and a shear Y
        # that fixes t* and is the identity modulo d (it changes no string, Lemma A)
        for X in _small(4):
            if not cells._in_gamma(it, X) or _image(X, t) not in (star, (-star[0], -star[1])):
                continue
            shears = [
                cells._mul(((s * X[0][0], s * X[0][1]), (s * X[1][0], s * X[1][1])), cells._inverse(Z))
                for Z in found
                for s in (1, -1)
            ]
            assert any(
                _image(Y, star) == star and all((Y[r][c] - (r == c)) % d == 0 for r in range(2) for c in range(2))
                for Y in shears
            ), (t, X)


# ---------------------------------------------------------------------------------------------
# (5) and (9): the bond Gram form
# ---------------------------------------------------------------------------------------------
def _p1_gram(tables: Any, sites: Sequence[Any], rows: Sequence[Any]) -> list[list[F]]:
    """The 3x3 bond Gram form of the barycentric placement of the P1 expansion (one pin per component)."""
    atoms: dict[Any, int] = {}
    where = {}
    for s, (_, letter) in enumerate(sites):
        for k in range(tables.nops):
            p = tables.apply((k, 0, 0, 0), tables.letter(letter).ref)
            home = tuple(x - (x.numerator // x.denominator) for x in p)
            where[s, k] = (atoms.setdefault((s, home), len(atoms)), tuple(x.numerator // x.denominator for x in p))
    edges = set()
    for i, j, g in rows:
        for h in range(tables.nops):
            (a, na), hg = where[i, h], tables.mul((h, 0, 0, 0), g)
            b, nb = where[j, hg[0]]
            v = tuple(y + t - x for y, t, x in zip(nb, hg[1:], na, strict=True))
            edges.add(min((a, b, v), (b, a, tuple(-x for x in v))))
    n = len(atoms)
    root = list(range(n))

    def find(x: int) -> int:
        while root[x] != x:
            x = root[x]
        return x

    for a, b, _ in edges:
        root[find(a)] = find(b)
    # Laplacian rows sum_B (x_B + v - x_A) = 0, the first atom of each component pinned to 0
    system = [[F(0)] * n + [F(0)] * 3 for _ in range(n)]
    for a, b, v in edges:
        for x, y, sign in ((a, b, 1), (b, a, -1)):
            system[x][x] -= 1
            system[x][y] += 1
            for c in range(3):
                system[x][n + c] -= sign * v[c]
    pinned = set()
    for x in range(n):
        if find(x) not in pinned:
            pinned.add(find(x))
            system[x] = [F(int(y == x)) for y in range(n)] + [F(0)] * 3
    for col in range(n):  # Gauss-Jordan; the pinned Laplacian is nonsingular
        pivot = next(r for r in range(col, n) if system[r][col])
        system[col], system[pivot] = system[pivot], system[col]
        system[col] = [x / system[col][col] for x in system[col]]
        for r in range(n):
            if r != col and system[r][col]:
                system[r] = [x - system[r][col] * y for x, y in zip(system[r], system[col], strict=True)]
    G = [[F(0)] * 3 for _ in range(3)]
    for a, b, v in edges:
        d = [system[b][n + c] + v[c] - system[a][n + c] for c in range(3)]
        for r, c in product(range(3), repeat=2):
            G[r][c] += d[r] * d[c]
    return G


def _gram_nets(it: int, rng: random.Random) -> list[tuple[list[Any], list[Any]]]:
    """Every harness kind, and special-position sites with rows whose set stabiliser swaps the ends."""
    tables = _tables(it)
    general = max(tables.spacegroup.wyckoff, key=lambda w: (w.free_count, w.multiplicity)).letter
    sites = [("X", general), ("Y", general), ("Z", general)]
    nets = [(sites, make()) for make in _kinds(it, rng).values()]
    specials = [w.letter for w in tables.spacegroup.wyckoff if 0 < w.free_count < 3] or [general]
    centres = [k for k in range(tables.nops) if tables.Wi[k] == (-1, 0, 0, 0, -1, 0, 0, 0, -1)] or [0]
    while len(nets) < len(_kinds(it, rng)) + 4:
        g = (rng.choice(centres), rng.randint(-1, 1), 0, rng.randint(-1, 1))
        if g != IDENTITY:
            two = [("X", rng.choice(specials)), ("Y", general)]
            nets.append((two, [(0, 1, IDENTITY), (1, 1, g), (0, 0, (0, 1, 0, 0))]))
    return nets


@pytest.mark.parametrize("it", MONOCLINIC)
def test_bond_gram_equals_the_p1_expansion(it: int) -> None:
    tables, rng = _tables(it), random.Random(f"gram-{it}")
    for sites, rows in _gram_nets(it, rng):
        sites0, rows0 = scramble(tables, sites, rows, rng)
        expected = _p1_gram(tables, sites0, rows0)
        assert expected[0][1] == expected[1][2] == 0  # the b axis is orthogonal to the ac plane
        G, gbb = cells._bond_gram(tables, sites0, rows0)
        assert G == ((expected[0][0], expected[0][2]), (expected[2][0], expected[2][2])) and gbb == expected[1][1]


# kind -> (rank of G_ac, whether G_bb > 0, i.e. whether some component is periodic along b)
RANKS = {
    "molecule": (0, False),
    "molecule on an inversion centre": (0, False),
    "chain along a": (1, False),
    "chain along b": (0, True),
    "layer in the ac plane": (2, False),
    "two helices along b": (0, True),
    "helix and molecule on an axis": (0, True),
    "layer containing b": (1, True),
    "3-periodic": (2, True),
}


@pytest.mark.parametrize("it", MONOCLINIC)
def test_bond_gram_transforms_with_the_cell_and_has_the_rank_of_the_periodicity(it: int) -> None:
    tables, rng = _tables(it), random.Random(f"gram-cell-{it}")
    general = max(tables.spacegroup.wyckoff, key=lambda w: (w.free_count, w.multiplicity)).letter
    sites = [("X", general), ("Y", general), ("Z", general)]
    for name, make in _kinds(it, rng).items():
        rows = make()
        G, gbb = cells._bond_gram(tables, sites, rows)
        rank = 2 if G[0][0] * G[1][1] != G[0][1] ** 2 else int(bool(G[0][0] or G[1][1]))
        assert (rank, gbb > 0) == RANKS[name], name
        # other reference atoms and numbering; a coset of E would change the cell
        assert cells._bond_gram(tables, *scramble(tables, sites, rows, rng, change_description=False)) == (G, gbb)
        X = _random_cell(it, rng)
        assert cells._bond_gram(tables, *_move(it, sites, rows, X)) == (cells._form(X, G), gbb)


# ---------------------------------------------------------------------------------------------
# the bond weights do not depend on the reference atoms: distinct sites of one letter share a
# probe point, which once let a gauge make their bond look swappable and halve its weight
# ---------------------------------------------------------------------------------------------
GAUGE_REPROS = {
    "P2_1/m": (
        11,
        [("Y", "f"), ("Y", "f")],
        [(0, 0, (0, 0, -1, 1)), (0, 1, (1, -1, 0, 0)), (0, 1, (1, 1, 1, -1)), (1, 1, (2, -1, 1, 0))],
    ),
    "C2/c": (
        15,
        [("X", "f")] * 5,
        [
            (0, 2, (5, 1, 1, -1)),
            (0, 4, (0, -1, 0, 1)),
            (0, 4, (5, -1, -1, 1)),
            (0, 4, (7, 0, -1, 0)),
            (1, 1, (2, 1, -1, 0)),
            (1, 4, (0, -1, -1, 1)),
            (2, 3, (5, -1, 0, -1)),
        ],
    ),
}


@pytest.mark.parametrize("name", list(GAUGE_REPROS))
def test_reference_atoms_change_neither_the_bond_gram_nor_the_net(name: str) -> None:
    it, sites, rows = GAUGE_REPROS[name]
    tables, rng = _tables(it), random.Random(name)
    gram, expected = cells._bond_gram(tables, sites, rows), _net(it, sites, rows)
    for _ in range(12):
        scrambled = scramble(tables, sites, rows, rng, change_description=False)
        assert cells._bond_gram(tables, *scrambled) == gram
        assert _net(it, *scrambled) == expected


def _many_site_net(tables: Any, rng: random.Random) -> tuple[list[Any], list[Any]]:
    """Five or six sites, letters repeated, with same-site rows and translations of up to one cell."""
    wyckoff = [w for w in tables.spacegroup.wyckoff if w.free_count] * 3 + list(tables.spacegroup.wyckoff)
    sites: list[Any] = []
    count = rng.randint(5, 6)
    while len(sites) < count:
        w = rng.choice(wyckoff)
        if w.free_count or all(letter != w.letter for _, letter in sites):  # a fixed position holds one site
            sites.append((rng.choice("XY"), w.letter))
    H = [letter_field(tables, letter, "H") for _, letter in sites]
    rows = {(k, k + 1, IDENTITY) for k in range(len(sites) - 1)}
    while len(rows) < 2 * len(sites):
        i, j = sorted((rng.randrange(len(sites)), rng.randrange(len(sites))))
        g = tables.canon(H[i], H[j], (rng.randrange(tables.nops), *(rng.randint(-1, 1) for _ in range(3))), i == j)
        if i != j or g != IDENTITY:
            rows.add((i, j, g))
    return sites, sorted(rows)


@pytest.mark.parametrize("it,nets", _profile([(it, 1) for it in MONOCLINIC], [(it, 6) for it in MONOCLINIC]))
def test_many_site_nets_have_one_canonical_net(it: int, nets: int) -> None:
    tables, rng = _tables(it), random.Random(f"many-sites-{it}-{nets}")
    for _ in range(nets):
        sites, rows = _many_site_net(tables, rng)
        gram, expected = cells._bond_gram(tables, sites, rows), _net(it, sites, rows)
        for _ in range(3):
            assert cells._bond_gram(tables, *scramble(tables, sites, rows, rng, change_description=False)) == gram
            assert _net(it, *scramble(tables, sites, rows, rng)) == expected, (sites, rows)


# ---------------------------------------------------------------------------------------------
# (12): the maintainer's group-14 family
# ---------------------------------------------------------------------------------------------
@cache
def _signature_trivial_cells() -> list[Matrix]:
    """The cell changes of P2_1/c with entries up to 2 that keep every letter and operation."""
    group = _tables(14).spacegroup
    identity = _signature(group, _action(UNIT))
    return [X for X in _small(2) if X != UNIT and _signature(group, _action(X)) == identity]


def _family_member(n: Sequence[int], m: Sequence[int], m2: Sequence[int]) -> tuple[list[Any], list[Any]]:
    """A helix on a 2_1 axis (X), dimers on two inversion centres (Y, Z) and an atom bonded to Z (W)."""
    sites = [(cls, "e") for cls in "XYZW"]  # operations 1 identity, 2 inversion, 3 -x,y+1/2,-z+1/2
    return sites, [(0, 0, (2, *n)), (1, 1, (1, *m)), (2, 2, (1, *m2)), (2, 3, IDENTITY)]


FAMILY = [(n, m, m2) for n, m, m2 in product(product((0, 1), repeat=3), repeat=3) if m != m2]


def test_group_14_family_instance() -> None:
    # under the 4.1 set this member gave ..._4~4@3@565 before and ..._4~4@3@465 after x+2z, y, -z (nets 5;
    # nets 6 compares the codes in listing order, which numbers the two self-bonded sites the other way)
    assert len(_signature_trivial_cells()) == 19 and ((1, 2), (0, -1)) in _signature_trivial_cells()
    sites, rows = _family_member((0, 1, 0), (1, 1, 0), (0, 0, 1))
    for X in (UNIT, ((1, 2), (0, -1))):
        result = _canonical_net(14, *_move(14, sites, rows, X))
        assert oracle.render(result.slots, result.codes) == "1~1@2.2_3~3@3@565_4~4@2@455"


@pytest.mark.parametrize(
    "members",
    [
        random.Random("family").sample(FAMILY, 3),
        *(pytest.param(FAMILY[k::8], marks=pytest.mark.extended) for k in range(8)),
    ],
)
def test_group_14_family_does_not_depend_on_the_cell(members: list[Any]) -> None:
    for member in members:
        sites, rows = _family_member(*member)
        expected = _net(14, sites, rows)
        for X in _signature_trivial_cells():
            assert _net(14, *_move(14, sites, rows, X)) == expected, (member, X)


# ---------------------------------------------------------------------------------------------
# (13): real monoclinic chains and layers, cell-changed as structures, through from_asu
# ---------------------------------------------------------------------------------------------
FIXTURES = Path(__file__).parent / "fixtures" / "cod"
# name -> (COD entry, cutoff table in Å, cell changes of Γ); the cutoffs keep the covalent contacts only
REAL = {
    # diopside CaMgSi2O6 (C2/c): corner-sharing SiO4 tetrahedra form chains along c (rank 1)
    "diopside-chains": ("1000008", {("Si", "O"): 1.75}, [((1, 2), (0, 1)), ((1, 0), (2, 1)), ((-1, 2), (2, -5))]),
    # with the Mg-O and Ca-O contacts the net is 3-periodic (rank 2)
    "diopside-full": (
        "1000008",
        {("Si", "O"): 1.75, ("Mg", "O"): 2.2, ("Ca", "O"): 2.75},
        [((1, 2), (0, 1)), ((-1, 2), (2, -5))],
    ),
    # CuCl2 (C2/m): edge-sharing CuCl4 squares form chains along b (rank 0, periodic along b); the
    # E-only search gives one label for every Γ cell with entries up to 4, so this case covers only the
    # rank-0 path for a real structure, not a form-2 failure
    "cucl2-chains": ("1539945", {("Cu", "Cl"): 2.4}, [((1, 1), (0, 1)), ((1, 0), (2, 1)), ((-1, 1), (2, -3))]),
    # AlCl3 (C2/m): edge-sharing AlCl6 octahedra form layers containing a and b (rank 1); the E-only
    # search changes the label under the last cell (...3.3@4@554 instead of ...3.3@4)
    "alcl3-layers": (
        "1535669",
        {("Al", "Cl"): 2.4},
        [((1, 1), (0, 1)), ((1, 0), (2, 1)), ((-1, 1), (2, -3)), ((-3, -1), (-2, -1))],
    ),
    # AgO = Ag(I)Ag(III)O2 (P2_1/c): a 3-periodic net (rank 2); some changes move Ag(III) from d to b
    "ago-framework": ("1509688", {("Ag", "O"): 2.3}, [((-1, 0), (-2, 1)), ((-1, 2), (0, -1)), ((-1, 0), (1, 1))]),
}


def _structure(path: Path, X: Matrix) -> tuple[ASUStructure, list[Any], list[Any], list[list[float]]]:
    """The structure of a CIF described in the cell ``X``: exact free parameters, lattice ``P⁻ᵀ A``."""
    loaded = load_structure(str(path))
    assert loaded is not None
    it, sites, _, lattice = loaded
    group = Spacegroup.standard(it)
    action = _action(X)
    params = [tuple(site.free_params.to_fractions()) for site in httk.core.load(str(path)).wyckoff_sites]
    moved = []
    for (element, letter), values in zip(sites, params, strict=True):
        compiled = compile_wyckoff_action(group, action, letter)
        moved.append((element, compiled.target_letter, tuple(compiled.apply(values).to_fractions())))
    (a, b), (c, d) = cells._inverse(X)  # rows of the new lattice: (P^-1)ᵀ times the old rows
    new = [
        [a * lattice[0][k] + c * lattice[2][k] for k in range(3)],
        list(lattice[1]),
        [b * lattice[0][k] + d * lattice[2][k] for k in range(3)],
    ]
    asu = ASUStructure(
        Cell(new),
        it,
        [WyckoffSite(letter, FracVector(values), element) for element, letter, values in moved],
        [Species(name=e, chemical_symbols=(e,), concentration=(1,)) for e in dict.fromkeys(e for e, _, _ in moved)],
    )
    refs = [reference_point(it, letter, values) for _, letter, values in moved]
    return asu, [(element, letter) for element, letter, _ in moved], refs, new


@pytest.mark.parametrize("name", list(REAL))
def test_real_chains_and_layers_do_not_depend_on_the_cell(name: str) -> None:
    entry, cutoffs, changes = REAL[name]
    table = {frozenset(pair): cutoff for pair, cutoff in cutoffs.items()}
    labels: dict[tuple[str, ...], set[str]] = {}
    canonical, distances = set(), set()
    for X in [UNIT, *changes]:
        asu, sites, refs, lattice = _structure(FIXTURES / f"{entry}.cif", X)
        rows, found = raw_rows(_tables(asu.spacegroup.it_number), sites, refs, lattice, table, reach=4)
        distances.add(tuple(sorted(round(x, 4) for x in found.values())))
        value = Protonetstructure.from_asu(asu, [(i + 1, j + 1, g[0] + 1, *g[1:]) for i, j, g in rows])
        # the constructor keeps the bare description it is given (here the letters of the cell)
        labels.setdefault(tuple(sorted(letter for _, letter in sites)), set()).add(str(value.label))
        canonical.add(str(canonical_protonetstructure(value).label))
    assert len(distances) == 1  # the same bonds in every cell
    # the canonical labels are the load-bearing check; the from_asu labels grouped by letters can be
    # vacuous, since a group may hold a single cell
    assert all(len(group) == 1 for group in labels.values()) and len(canonical) == 1, (labels, canonical)
