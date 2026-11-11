"""The protonet search shortcuts change nothing: twins, fragments, the pre-gauge and the deadline."""

import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import protonet_reference as oracle
import pytest
from _protonet_support import generic_cutoffs, geometric_round_trip, load_structure, random_net, raw_rows

from httk.atomistic.symmetry import protonet
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
    assert time.perf_counter() - start < 1.5
