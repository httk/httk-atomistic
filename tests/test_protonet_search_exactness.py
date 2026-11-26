"""The protonet search shortcuts change nothing: twins, fragments, the pre-gauge and the deadline.

The pattern bound, the lazy and collapsed twin expansion and the leaf-first reduction are compared
with ``protonet_baseline``, the search before them.
"""

import math
import random
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import protonet_baseline as baseline
import protonet_reference as oracle
import pytest
from _protonet_support import (
    generic_cutoffs,
    geometric_round_trip,
    load_structure,
    random_leafy_net,
    random_net,
    raw_rows,
)
from test_protonet_canonical_api import CASES, _case
from test_protonet_molecular import MOLECULAR, _described

from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
from httk.atomistic.symmetry import protonet
from httk.atomistic.symmetry.canonical_classification import canonical_protonetstructure, canonical_protonettype
from httk.atomistic.symmetry.limits import CanonicalizationLimitError, _budget_scope
from httk.atomistic.symmetry.protonet import _canonical_net, _cell, _minimal_numberings, _search, _tables

FIXTURES = Path(__file__).parent / "fixtures"
IDENTITY = (0, 0, 0, 0)


def _numberings(result: tuple[Any, list[dict[int, int]]]) -> tuple[Any, Counter[frozenset[tuple[int, int]]]]:
    pattern, numberings = result
    return tuple(pattern), Counter(frozenset(n.items()) for n in numberings)


def _random_pairs(rng: random.Random) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Blocks and site pairs with twin classes (of up to four), self-loops and adjacent twins."""
    nsites = rng.randint(2, 9)
    ids = list(range(nsites))
    rng.shuffle(ids)
    cuts = sorted(rng.sample(range(1, nsites), min(rng.randint(0, 2), nsites - 1)))
    blocks = [ids[a:b] for a, b in zip([0, *cuts], [*cuts, nsites], strict=True)]
    rows = []
    for _ in range(rng.randint(1, 10)):
        a, b = rng.randrange(nsites), rng.randrange(nsites)
        b = a if rng.random() < 0.1 else b
        rows.append((min(a, b), max(a, b)))
    for _ in range(rng.randint(0, 4)):  # copy one site's rows onto another of its block
        block = rng.choice(blocks)
        if len(block) < 2:
            continue
        s1, s2 = rng.sample(block, 2)
        rows = [r for r in rows if s2 not in r]
        for a, b in list(rows):
            if a == b == s1:
                rows.append((s2, s2))
            elif s1 in (a, b):
                other = b if a == s1 else a
                rows.append((min(s2, other), max(s2, other)))
        if rng.random() < 0.3:
            rows.append((min(s1, s2), max(s1, s2)))
    rng.shuffle(rows)
    return blocks, rows or [(0, min(1, nsites - 1))]


def _leafy_pairs(rng: random.Random) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Blocks and pairs biased to the leaf-first reduction: a first block of leaves on 1-4 neighbours.

    The neighbours spread over 1-3 other blocks with equal and unequal leaf counts (twins, 2-5 per
    neighbour); sometimes a block of holders (only leaves, an equal number each), isolated sites in the first block, a second all-leaf block, tied neighbour
    blocks with mirrored rows, self-loops and double rows on the neighbours, and (making the
    reduction inapplicable) a row inside the first block, a second row of a leaf or a doubled leaf row.
    """
    others = [[f"n{b}.{x}" for x in range(rng.randint(1, 3))] for b in range(rng.randint(1, 3))]
    if rng.random() < 0.25:  # a tie: a copy of a block, mirrored rows added below
        others.append([f"m{x}" for x in range(len(others[0]))])
    heavy = [s for block in others for s in block]
    rows: list[tuple[str, str]] = []
    for _ in range(rng.randint(0, 5)):
        a, b = rng.choice(heavy), rng.choice(heavy)
        rows.append((a, b))
        if rng.random() < 0.15:
            rows.append((a, b))  # a double row (a self-loop when a == b)
    if others[-1][0].startswith("m"):
        mirror = dict(zip(others[0], others[-1], strict=True))
        rows += [(mirror.get(a, a), mirror.get(b, b)) for a, b in rows if a in mirror or b in mirror]
    first: list[str] = []
    budget = 8  # leaves, so that the oracle's twin expansion stays small
    for neighbour in rng.sample(heavy, min(len(heavy), rng.randint(1, 4))):
        for _ in range(min(budget, rng.choice([1, 1, 2, 2, 3, 4, 5]))):
            leaf = f"h{len(first)}"
            first.append(leaf)
            rows.append((leaf, neighbour))
            budget -= 1
    if rng.random() < 0.3:  # holders: a block whose sites have only leaves, an equal number each
        count = rng.randint(1, 2)
        holders = [f"o{x}" for x in range(rng.randint(2, 3)) if budget >= count * (x + 1)]
        for holder in holders:
            first += [f"h{len(first) + x}" for x in range(count)]
            rows += [(leaf, holder) for leaf in first[-count:]]
            budget -= count
        others.insert(rng.randrange(len(others) + 1), holders)
    first += [f"i{x}" for x in range(rng.choice([0, 0, 0, 1, 2]))]
    blocks = [first, *others]
    if rng.random() < 0.25:  # a second all-leaf block on the heavy sites
        second = [f"k{x}" for x in range(rng.randint(1, 3))]
        rows += [(s, rng.choice(heavy)) for s in second]
        blocks.insert(1, second)
    if first and rng.random() < 0.1:
        rows.append((rng.choice(first), rng.choice(first)))  # a row inside the first block
    if first and rng.random() < 0.1:
        rows.append((rng.choice(first), rng.choice(heavy)))  # a second (or doubled) leaf row
    names = [s for block in blocks for s in block]
    ids = list(range(len(names)))
    rng.shuffle(ids)
    index = dict(zip(names, ids, strict=True))
    rng.shuffle(rows)
    pairs = [(index[a], index[b]) for a, b in rows]
    return [[index[s] for s in block] for block in blocks if block], [(min(a, b), max(a, b)) for a, b in pairs]


def _twin_classes(blocks: list[list[int]], rows: list[tuple[int, int]]) -> list[list[int]]:
    """The twin classes of size two or more: one block, equal neighbours with multiplicity, not isolated."""
    neighbours: dict[int, Counter[int]] = {s: Counter() for block in blocks for s in block}
    for a, b in rows:
        neighbours[a][b] += 1
        if a != b:
            neighbours[b][a] += 1
    classes: dict[tuple[int, frozenset[tuple[int, int]]], list[int]] = {}
    for index, block in enumerate(blocks):
        for s in block:
            if neighbours[s]:
                classes.setdefault((index, frozenset(neighbours[s].items())), []).append(s)
    return [c for c in classes.values() if len(c) > 1]


def _smaller(pattern: tuple[tuple[int, int], ...], rng: random.Random) -> tuple[tuple[int, int], ...] | None:
    """A pattern of the same length just below ``pattern``, or ``None`` when every entry is (0, 0)."""
    spots = [x for x, (a, b) in enumerate(pattern) if a or b]
    if not spots:
        return None
    x = rng.choice(spots)
    a, b = pattern[x]
    return (*pattern[:x], (a, b - 1) if b > a else (a - 1, b), *pattern[x + 1 :])


def _check_numberings(blocks: list[list[int]], rows: list[tuple[int, int]], rng: random.Random) -> None:
    """Compare ``_minimal_numberings`` with the baseline: plain, with a collapse set and with bounds."""
    expected_pattern, expected_list = baseline._minimal_numberings(blocks, rows)
    expected = _numberings((expected_pattern, expected_list))
    found = _minimal_numberings(blocks, rows)
    assert isinstance(found[1], Iterator)
    assert _numberings(found) == expected
    # a collapse set: each collapsed twin class keeps only its numbering in block order
    collapse = {s for block in blocks for s in block if rng.random() < 0.7}
    collapsed = [c for c in _twin_classes(blocks, rows) if all(s in collapse for s in c)]

    def ordered(number: dict[int, int]) -> frozenset[tuple[int, int]]:
        moved = dict(number)
        for c in collapsed:
            moved.update(zip(c, sorted(number[s] for s in c), strict=True))
        return frozenset(moved.items())

    pattern, numberings = _minimal_numberings(blocks, rows, collapse=collapse)
    got = Counter(frozenset(n.items()) for n in numberings)
    assert pattern == expected_pattern and all(count == 1 and ordered(dict(n)) == n for n, count in got.items())
    weight = math.prod(math.factorial(len(c)) for c in collapsed)
    assert Counter(ordered(n) for n in expected_list) == {n: weight for n in got}
    # bounds: the minimum itself, a pattern of any numbering (never smaller) and one just below the minimum
    shuffled = [s for block in blocks for s in rng.sample(block, len(block))]
    number = {s: n for n, s in enumerate(shuffled)}
    other = tuple(sorted((min(number[a], number[b]), max(number[a], number[b])) for a, b in rows))
    for bound in (expected_pattern, other):
        assert bound >= expected_pattern
        assert _numberings(_minimal_numberings(blocks, rows, bound=bound)) == expected
    below = _smaller(expected_pattern, rng)
    if below is not None:
        pattern, numberings = _minimal_numberings(blocks, rows, bound=below, collapse=collapse)
        assert pattern is None and not list(numberings)


@pytest.mark.parametrize(
    "seed", [*range(8), *(pytest.param(seed, marks=pytest.mark.extended) for seed in range(8, 24))]
)
def test_numbering_shortcuts_match_the_baseline(seed: int, monkeypatch: pytest.MonkeyPatch) -> None:
    # 8 x 2500 = 20,000 cases in the normal profile, 16 x 31,250 = 500,000 more in the extended one
    reduced = 0
    leaf_first = protonet._leaf_first

    def counted(*args: Any) -> Any:
        nonlocal reduced
        out = leaf_first(*args)
        reduced += out is not None
        return out

    monkeypatch.setattr(protonet, "_leaf_first", counted)
    rng = random.Random(f"numberings-{seed}")
    cases = 2500 if seed < 8 else 31250
    for k in range(cases):
        blocks, rows = _random_pairs(rng) if k % 4 == 0 else _leafy_pairs(rng)
        _check_numberings(blocks, rows, rng)
    assert reduced > 2 * cases  # the reduction ran, nested too: each case calls the search four times or more


def test_twin_expansion_is_lazy_and_bounded() -> None:
    # ten twins on one site: 10! = 3,628,800 numberings, read one at a time, leaves first (the
    # leaf-first reduction) and centre first (the unreduced search), never collapsed
    leaves = list(range(10))
    for blocks in ([leaves, [10]], [[10], leaves]):
        rows = [(s, 10) for s in leaves]
        tracemalloc.start()
        with _budget_scope(60.0):  # a safety net only: the memory peak shows the laziness
            pattern, numberings = _minimal_numberings(blocks, rows)
            first = list(islice(numberings, 5))
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert peak < 2_000_000  # a list of the 10! numberings would take gigabytes
        assert len({frozenset(n.items()) for n in first}) == 5
        assert isinstance(numberings, Iterator) and len(pattern or ()) == 10
        # collapsed, the class gives its block order only
        assert len(list(_minimal_numberings(blocks, rows, collapse=set(leaves))[1])) == 1


def test_special_position_twins_are_evaluated_one_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    # C2/c: a general carbon with nine leaves on the 2-fold axis (4e): twins that never collapse
    tables = _tables(15)
    sites = [("C", "f")] + [("H", "e")] * 9
    H = [tables.letter(letter).H for _, letter in sites]
    rows = [(0, s, tables.canon(H[0], H[s], (s % tables.nops, s % 2, 0, 0))) for s in range(1, 10)]
    _canonical_net(15, sites[:2], rows[:1])  # the group's descriptions are compiled outside the budget

    class Enough(Exception):
        pass

    calls = 0
    exact_codes = protonet._exact_codes

    def counted(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 200:
            raise Enough
        return exact_codes(*args)

    monkeypatch.setattr(protonet, "_exact_codes", counted)
    tracemalloc.start()
    with pytest.raises(Enough), _budget_scope(120.0):  # 200 evaluations come before any limit
        _search.__wrapped__(15, tuple(sites), frozenset(rows), (None,) * 10, (), True)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 50_000_000  # the 9! numberings were never listed


def _fields(net: Any) -> tuple[Any, ...]:
    return (net.it_number, net.groups, net.slots, net.codes, net.shapes, net.torsions)


def _realisations(net: Any) -> set[tuple[Any, ...]]:
    return {(r.coset, r.classes, r.letters, r.number, r.gauge, r.cell) for r in net.realisations}


@pytest.mark.parametrize(
    "it,nets",
    [
        *((it, 12) for it in (1, 2, 4, 14, 15, 19, 62)),
        *(pytest.param(it, 60, marks=pytest.mark.extended) for it in (1, 2, 4, 14, 15, 19, 62)),
    ],
)
def test_search_matches_the_baseline_on_molecular_fragments(it: int, nets: int) -> None:
    tables, rng = _tables(it), random.Random(f"leafy-{it}-{nets}")
    fewer = 0
    for _ in range(nets):
        sites, rows, shapes, torsions = random_leafy_net(tables, rng)
        for chirality in (True, False):
            for own_bare in (False, True):
                kw = {"shapes": shapes, "torsions": torsions, "preserve_chirality": chirality, "own_bare": own_bare}
                expected = baseline._canonical_net(it, sites, rows, **kw)
                result = _canonical_net(it, sites, rows, **kw)
                assert _fields(result) == _fields(expected), (sites, rows, shapes, torsions, chirality, own_bare)
                assert _realisations(result) <= _realisations(expected)
                fewer += len(result.realisations) < len(expected.realisations)
    assert fewer  # collapsed leaf twins were met


@pytest.mark.parametrize("it", [2, 14, 15, 62])
def test_search_matches_the_baseline_on_random_nets(it: int) -> None:
    tables, rng = _tables(it), random.Random(f"plain-{it}")
    for _ in range(10):
        sites, rows = random_net(tables, rng)
        for chirality in (True, False):
            expected = baseline._canonical_net(it, sites, rows, preserve_chirality=chirality)
            result = _canonical_net(it, sites, rows, preserve_chirality=chirality)
            assert _fields(result) == _fields(expected)
            assert _realisations(result) <= _realisations(expected)


def test_a_methyl_rich_net_completes_within_a_small_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    # P2_1/c: a chain of six general carbons, each with three general hydrogens (18 twins in six
    # classes, 6^6 orders) and a nitrogen end; the leaf-first search and the collapsed twins finish it
    tables = _tables(14)
    sites = [("C", "e")] * 6 + [("N", "e")] + [("H", "e")] * 18
    rows = [(c, c + 1, (0, 0, 0, 0)) for c in range(6)]
    rows += [(h // 3, 7 + h, (h % tables.nops, h % 2, 0, 0)) for h in range(18)]
    _canonical_net(14, sites[:2], rows[:1])  # the group's descriptions are compiled outside the budget
    calls = 0
    exact_codes = protonet._exact_codes

    def counted(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        return exact_codes(*args)

    monkeypatch.setattr(protonet, "_exact_codes", counted)
    with _budget_scope(20.0):  # a safety net: the search before took 8 s
        result = _canonical_net(14, sites, rows)
    # the search before evaluated every one of the 6^6 = 46,656 twin orders and kept them all
    assert len(result.slots) == 24 and len(result.realisations) == 1 and calls < 100


FIXED_PAIRS = {
    "twin class of three": ([[0], [1, 2, 3]], [(0, 1), (0, 2), (0, 3)]),
    "loops on adjacent sites": ([[0, 1]], [(0, 1), (0, 0), (1, 1)]),
    "loops on two pairs": ([[0, 1, 2, 3]], [(0, 1), (0, 0), (1, 1), (2, 3), (2, 2), (3, 3)]),
    "twins in a cross-class block": ([[0, 1, 2], [3]], [(0, 3), (1, 3), (2, 3), (2, 2)]),
    "mutually adjacent twins": ([[0, 1], [2]], [(0, 1), (0, 2), (1, 2)]),
    "two twin classes": ([[0, 1], [2, 3, 4, 5]], [(0, 2), (0, 3), (1, 4), (1, 5), (0, 1)]),
}


@pytest.mark.parametrize("name", list(FIXED_PAIRS))
def test_twin_reduction_matches_the_full_enumeration_on_fixed_cases(name: str) -> None:
    blocks, rows = FIXED_PAIRS[name]
    assert _numberings(_minimal_numberings(blocks, rows)) == _numberings(oracle._minimal_numberings(blocks, rows))


def test_twin_reduction_matches_the_full_enumeration_on_random_pairs() -> None:
    rng = random.Random("twins")
    for _ in range(2000):
        blocks, rows = _random_pairs(rng)
        assert _numberings(_minimal_numberings(blocks, rows)) == _numberings(oracle._minimal_numberings(blocks, rows))


def _joining_component(tables: Any, rng: random.Random) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    """A connected local component whose slots (1,3), (2,4) open two fragments that (3,4) joins."""
    wyckoff = tables.spacegroup.wyckoff
    general = max(wyckoff, key=lambda w: (w.free_count, w.multiplicity)).letter
    fixed = {w.letter for w in wyckoff if w.free_count == 0}
    letters: list[str] = []
    size = rng.randint(4, 5)
    while len(letters) < size:
        letter = general if rng.random() < 0.5 else rng.choice(wyckoff).letter
        if not (letter in fixed and letter in letters):
            letters.append(letter)
    pairs = [(0, 2), (1, 3), (2, 3), *((rng.randrange(len(letters)), rng.randrange(len(letters))) for _ in range(2))]
    if len(letters) == 5:
        pairs.append((rng.randrange(4), 4))
    rows = set()
    for a, b in pairs:
        a, b = min(a, b), max(a, b)
        g = (rng.randrange(tables.nops), rng.randint(-1, 1), rng.randint(-1, 1), rng.randint(-1, 1))
        code = tables.canon(tables.letter(letters[a]).H, tables.letter(letters[b]).H, g, a == b)
        if code != IDENTITY or a != b:
            rows.add((a, b, code))
    return tuple(letters), tuple(sorted(rows))


@pytest.mark.parametrize("it", [12, 15, 62, pytest.param(225, marks=pytest.mark.extended)])
def test_fragment_factorisation_matches_the_oracle_state_search(it: int, monkeypatch: pytest.MonkeyPatch) -> None:
    joins = 0
    join = protonet._join

    def counted(x: Any, y: Any) -> Any:
        nonlocal joins
        joins += 1
        return join(x, y)

    monkeypatch.setattr(protonet, "_join", counted)
    tables, reference, rng = _tables(it), oracle.tables(it), random.Random(f"fragments-{it}")
    for _ in range(12):
        letters, rows = _joining_component(tables, rng)
        codes, gauge = protonet._component_codes.__wrapped__(tables, letters, rows)
        number = {s: s + 1 for s in range(len(letters))}
        _, expected, _ = oracle._exact_codes(reference, dict(enumerate(letters)), number, list(rows))
        assert [tuple(c) for c in codes] == [tuple(c) for c in expected]
        H = [tables.letter(letter).H for letter in letters]
        slots = sorted((a + 1, b + 1) for a, b, _ in rows)
        reached = Counter(
            (
                (a + 1, b + 1),
                tables.canon(
                    H[a],
                    H[b],
                    tables.mul(tables.mul(gauge[a] or IDENTITY, g), tables.inv(gauge[b] or IDENTITY)),
                    a == b,
                ),
            )
            for a, b, g in rows
        )
        assert reached == Counter(zip(slots, codes, strict=True))
    assert joins > 0


@pytest.mark.parametrize("it", [2, 14, 15, 62])
def test_regauged_general_positions_give_the_same_net(it: int) -> None:
    tables, rng = _tables(it), random.Random(f"pregauge-{it}")
    for _ in range(6):
        sites, rows = random_net(tables, rng)
        reference = _canonical_net(it, sites, rows)
        change = [
            (rng.randrange(tables.nops), rng.randint(-2, 2), rng.randint(-2, 2), rng.randint(-2, 2))
            if tables.letter(letter).general
            else IDENTITY
            for _, letter in sites
        ]
        moved = [(a, b, tables.mul(tables.mul(change[a], g), tables.inv(change[b]))) for a, b, g in rows]
        result = _canonical_net(it, sites, moved)
        assert (result.slots, result.codes, result.groups) == (reference.slots, reference.codes, reference.groups)


def test_a_non_identity_pre_gauge_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = load_structure(str(FIXTURES / "cod" / "1000008.cif"))
    assert loaded is not None
    it, sites, refs, lattice = loaded
    tables = _tables(it)
    rows, dist = raw_rows(tables, sites, refs, lattice, generic_cutoffs(tables, sites, refs, lattice))
    changes: list[list[Any]] = []
    pregauge = protonet._pregauge

    def recorded(*args: Any) -> Any:
        out = pregauge(*args)
        changes.append(out[0])
        return out

    monkeypatch.setattr(protonet, "_pregauge", recorded)
    _search.cache_clear()
    result = _canonical_net(it, sites, rows)
    assert any(e != IDENTITY for change in changes for e in change)
    for real in result.realisations:
        assert geometric_round_trip(
            tables,
            sites,
            refs,
            lattice,
            dist,
            result.slots,
            result.codes,
            tables.cosets[real.coset],
            dict(enumerate(real.number)),
            dict(enumerate(real.gauge)),
            None if real.cell == ((1, 0), (0, 1)) else _cell(it, real.cell),
        )


def test_a_factorial_numbering_node_observes_the_deadline() -> None:
    # a P-1 centre bonded to nine distinct leaves: 9! orders of its neighbours, visited lazily
    def star(k: int) -> tuple[list[tuple[str, str]], list[tuple[int, int, tuple[int, int, int, int]]]]:
        rows = {(0, y, (0, y, 0, 0)) for y in range(k + 1, 2 * k + 1)}
        rows |= {(y, y, (0, n + 1, 0, 0)) for y in range(k + 1, 2 * k + 1) for n in range(y - k - 1)}
        return [("X", "i")] * (k + 1) + [("Y", "i")] * k, sorted(rows)

    _canonical_net(2, *star(2), own_bare=True)  # the group's descriptions are compiled outside the budget
    sites, rows = star(9)
    start = time.perf_counter()
    with pytest.raises(CanonicalizationLimitError), _budget_scope(0.5):
        _canonical_net(2, sites, rows, own_bare=True)
    assert time.perf_counter() - start < 5.0  # ten times the budget: the deadline is observed, not finished


def _labels(kind: str, name: str) -> list[str]:
    """The labels of one fixture: its value, both canonical levels, with and without the hand."""
    asu, bonds = _described(name)[:2] if kind == "molecular" else _case(name)[:2]
    value = Protonetstructure.from_asu(asu, bonds)
    labels = [str(value.label)]
    for chirality in (True, False):
        labels.append(str(canonical_protonetstructure(value, preserve_chirality=chirality, timeout=None).label))
        labels.append(str(canonical_protonettype(value, preserve_chirality=chirality, timeout=None).label))
    return labels


@pytest.mark.parametrize("kind,name", [*(("molecular", e) for e in MOLECULAR), *(("api", n) for n in CASES)])
def test_labels_match_the_baseline_search(kind: str, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    labels = _labels(kind, name)
    calls = 0

    def search(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        return baseline._search(*args)

    monkeypatch.setattr(protonet, "_search", search)
    assert _labels(kind, name) == labels and calls


# reviewed counterexamples of a leaf-first reduction that numbers holders (sites whose only rows
# go to first-block leaves) in one fixed order, and edges of the reduction
LEAF_FIRST_PAIRS = {
    "holders and a loop": ([[0, 1, 2, 3, 4, 5], [6, 7, 8]], [(7, 7), (1, 8), (4, 6), (0, 7), (3, 6), (5, 8)]),
    "holders, a loop on a neighbour": (
        [[0, 1, 2, 3, 4], [9, 7, 6, 8, 5]],
        [(5, 8), (4, 8), (2, 9), (3, 7), (0, 8), (8, 8)],
    ),
    # h0-h4 on n1 (two, a holder), n2 (two, bonded to c1), n3 (one); c1-c2; then n2 a holder too
    "mixed sub-block": ([[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]], [(0, 5), (1, 5), (2, 6), (3, 6), (4, 7), (6, 8), (8, 9)]),
    "two holders in a sub-block": (
        [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]],
        [(0, 5), (1, 5), (2, 6), (3, 6), (4, 7), (8, 9)],
    ),
    "an unbonded first block": ([[0, 1], [2, 3], [4]], [(2, 3), (3, 4), (4, 4)]),
    "a neighbour with one rest row": ([[0, 1, 2], [3, 4], [5]], [(0, 3), (1, 3), (2, 4), (3, 5)]),
    # leaf counts 2, 0 in one block and 3, 1 in the next
    "interleaved leaf counts": (
        [[0, 1, 2, 3, 4, 5], [6, 7], [8, 9]],
        [(0, 6), (1, 6), (2, 8), (3, 8), (4, 8), (5, 9), (6, 7), (7, 9), (8, 9)],
    ),
    "a second all-leaf block": ([[0, 1, 2], [3, 4], [5, 6]], [(0, 5), (1, 5), (2, 6), (3, 5), (4, 6), (5, 6)]),
}
FALLBACK_PAIRS = {
    "a self-loop in the first block": ([[0, 1], [2]], [(0, 0), (1, 2)]),
    "a row inside the first block": ([[0, 1, 2], [3]], [(0, 1), (2, 3)]),
    "a doubled leaf row": ([[0, 1], [2]], [(0, 2), (0, 2), (1, 2)]),
    "a leaf with two neighbours": ([[0, 1], [2, 3]], [(0, 2), (0, 3), (1, 3)]),
}
# twins on one centre: five give 120 sorted children, seven 5040 visited lazily (both pending
# together in the unreduced search, the centre first), with and without other rows
TWIN_PAIRS = {
    f"{k} twins, {kind}": ([[0], list(range(1, k + 1)), [k + 1]], [(0, s) for s in range(1, k + 1)] + extra)
    for k in (5, 7)
    for kind, extra in (("alone", []), ("with a tail", [(0, k + 1), (1, k + 1)]))
}


@pytest.mark.parametrize("name", [*LEAF_FIRST_PAIRS, *FALLBACK_PAIRS, *TWIN_PAIRS])
def test_numbering_shortcuts_match_the_baseline_on_fixed_cases(name: str) -> None:
    blocks, rows = {**LEAF_FIRST_PAIRS, **FALLBACK_PAIRS, **TWIN_PAIRS}[name]
    reduced = protonet._leaf_first(blocks, rows, None, frozenset())
    assert (reduced is not None) is (name in LEAF_FIRST_PAIRS)
    rng = random.Random(name)
    for _ in range(20):  # random collapse sets and bounds
        _check_numberings(blocks, rows, rng)


I = IDENTITY
HOLDERS = [(0, 4, I), (1, 4, I), (2, 5, I), (3, 5, I), (6, 7, I)]
G1, G2 = (1, 1, 0, 0), (2, 0, 1, 0)
# name -> (it, sites, rows, shapes, collapsed): whether general leaf twins drop realisations
SEARCH_CASES: dict[str, tuple[int, list[tuple[str, str]], list[Any], list[str | None] | None, bool]] = {
    # two holders O (shapes L2, A2) of two H each: the A2 one must take number 5
    "holders with shapes": (
        1,
        [("H", "a")] * 4 + [("O", "a")] * 2 + [("C", "a")] * 2,
        HOLDERS,
        [None] * 4 + ["L2", "A2", None, None],
        True,
    ),
    # the holders are a tie block of one-site classes N, O: the D3 species order is H-C-O-N
    "holders in a tie block": (
        1,
        [("H", "a")] * 4 + [("N", "a"), ("O", "a")] + [("C", "a")] * 2,
        HOLDERS,
        [None] * 4 + ["L2", "A2", None, None],
        True,
    ),
    # P2/m: general H on a special C or N (a pre-gauge root that keeps its reference atom)
    "general leaves on a special site": (
        10,
        [("H", "o"), ("H", "o"), ("C", "i")],
        [(0, 2, G1), (1, 2, G2)],
        None,
        True,
    ),
    "general leaves on a special site with a tail": (
        10,
        [("H", "o"), ("H", "o"), ("N", "i"), ("O", "a")],
        [(0, 2, G1), (1, 2, G2), (2, 3, (1, 0, 0, 0))],
        None,
        True,
    ),
    "seven general leaves on a special site": (
        10,
        [("H", "o")] * 7 + [("C", "i")],
        [(h, 7, (h % 4, h % 2, h % 3 - 1, 0)) for h in range(7)],
        None,
        True,
    ),
    # negative controls: special-position leaves, a leaf of two rows, one-site classes in a tie block
    "special leaves": (10, [("H", "i"), ("H", "i"), ("C", "o")], [(0, 2, G1), (1, 2, G2)], None, False),
    "leaves of two rows": (
        2,
        [("H", "i"), ("H", "i"), ("C", "i")],
        [(0, 2, I), (0, 2, (0, 1, 0, 0)), (1, 2, I), (1, 2, (0, 1, 0, 0))],
        None,
        False,
    ),
    "one-site leaf classes": (
        2,
        [("H1", "i"), ("H2", "i"), ("C", "i")],
        [(0, 2, I), (1, 2, (0, 1, 0, 0))],
        None,
        False,
    ),
    "leaves with different torsion letters": (
        2,
        [("H", "i"), ("H", "i"), ("C", "i"), ("C", "i")],
        [(0, 2, I), (1, 2, (0, 1, 0, 0)), (2, 3, I)],
        None,
        False,
    ),
}


@pytest.mark.parametrize("name", list(SEARCH_CASES))
def test_search_matches_the_baseline_on_fixed_cases(name: str) -> None:
    it, sites, rows, shapes, collapsed = SEARCH_CASES[name]
    torsions = ["t", "c", None] if name == "leaves with different torsion letters" else None
    for chirality in (True, False):
        for own_bare in (False, True):
            kw = {"shapes": shapes, "torsions": torsions, "preserve_chirality": chirality, "own_bare": own_bare}
            expected = baseline._canonical_net(it, sites, rows, **kw)
            result = _canonical_net(it, sites, rows, **kw)
            assert _fields(result) == _fields(expected)
            assert _realisations(result) <= _realisations(expected)
            assert (len(result.realisations) < len(expected.realisations)) is collapsed
            # the D3 species order and every label part come from what all realisations share
            assert min(r.classes for r in result.realisations) == min(r.classes for r in expected.realisations)
    if name.startswith("holders"):  # the holders keep both orders: shapes by number decide
        result = _canonical_net(it, sites, rows, shapes=shapes)
        assert result.shapes[4:6] == ("A2", "L2") or result.shapes[6:8] == ("A2", "L2")


@pytest.mark.parametrize("it", [4, 19])
def test_a_bound_tie_between_descriptions_matches_the_baseline(it: int, monkeypatch: pytest.MonkeyPatch) -> None:
    # without the hand, mirror-related descriptions tie on the pattern and differ later
    ties = 0
    numberings = protonet._minimal_numberings

    def recorded(blocks: Any, rows: Any, **kw: Any) -> Any:
        nonlocal ties
        pattern, found = numberings(blocks, rows, **kw)
        ties += kw.get("bound") is not None and pattern == kw["bound"]
        return pattern, found

    monkeypatch.setattr(protonet, "_minimal_numberings", recorded)
    tables, rng = _tables(it), random.Random(f"ties-{it}")
    for _ in range(8):
        sites, rows, shapes, torsions = random_leafy_net(tables, rng)
        kw = {"shapes": shapes, "torsions": torsions, "preserve_chirality": False}
        expected = baseline._canonical_net(it, sites, rows, **kw)
        result = _canonical_net(it, sites, rows, **kw)
        assert _fields(result) == _fields(expected) and _realisations(result) <= _realisations(expected)
    assert ties
