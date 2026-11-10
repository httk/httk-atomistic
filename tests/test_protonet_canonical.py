"""Bonding-net canonicaliser: production tables and search against the reference oracle."""

import random
from fractions import Fraction as F
from functools import cache
from pathlib import Path
from typing import Any

import protonet_reference as oracle
import pytest
from _protonet_support import (
    generic_cutoffs,
    geometric_round_trip,
    handmade,
    load_structure,
    random_net,
    raw_rows,
    reps,
    scramble,
    synthetic_sweep,
)

from httk.atomistic.symmetry._wyckoff_actions import compile_wyckoff_action
from httk.atomistic.symmetry.limits import CanonicalizationLimitError, _budget_scope
from httk.atomistic.symmetry.protonet import (
    _INVERSION,
    _apply,
    _canonical_net,
    _CanonicalNet,
    _mat_mul,
    _search,
    _tables,
)
from httk.atomistic.symmetry.spacegroup import Spacegroup

FIXTURES = Path(__file__).parent / "fixtures"
K2AGSBS4 = str(FIXTURES / "cod" / "1509425.cif")
FRAME = {frozenset(("Sb", "S")): 3.0, frozenset(("Ag", "S")): 3.0}
CUBIC = (5.64, 5.64, 5.64, 90, 90, 90)
NACL = {frozenset(("Na", "Cl")): 3.0}
# groups whose tabulated equivalent descriptions are not closed under composition (D7): scramble
# there without a change of description; pinned by test_tabulated_actions_close_under_composition
NON_CLOSED = frozenset({2})

# name -> (it number, spec or None for the CIF, cell, cutoffs, expected net or None)
CASES: dict[str, tuple[Any, ...]] = {
    "k2agsbs4-frame": (34, None, None, FRAME, "1~8.10_2~8.10_3~9.10_4~9.10@3"),
    "k2agsbs4-full": (
        34,
        None,
        None,
        {**FRAME, frozenset(("K", "S")): 3.75},
        (
            "1~5.6.7.7@4.8.10_2~5.6@4.7@4.7@2@556.8@1@556.10@1@556"
            "_3~5.6.7@1@556.9.10@4@455_4~5.7.7@3@445.9@4@454.10@4@455"
        ),
    ),
    "nacl": (225, [("Na", "a", ()), ("Cl", "b", ())], CUBIC, NACL, "1~2"),
    "nacl-exchanged": (225, [("Cl", "a", ()), ("Na", "b", ())], CUBIC, NACL, "1~2"),
    "diamond": (227, [("C", "a", ())], (3.567, 3.567, 3.567, 90, 90, 90), {frozenset(("C",)): 1.7}, "1~1@14@654"),
    "wurtzite": (
        186,
        [("Zn", "b", (F(0),)), ("S", "b", (F(3, 8),))],
        (3.82, 3.82, 6.26, 90, 90, 120),
        {frozenset(("Zn", "S")): 2.6},
        "1~2.2@2",
    ),
    "pdse2": (
        61,
        [("Pd", "a", ()), ("Se", "c", (F(1112, 10000), F(1179, 10000), F(4073, 10000)))],
        (5.7457, 5.8679, 7.6946, 90, 90, 90),
        {frozenset(("Pd", "Se")): 2.6, frozenset(("Se",)): 2.6},
        "1~1@2@455.2.2@4@455",
    ),
    "p21-chain": (
        4,
        [
            ("A", "a", (F(1, 10), F(1, 5), F(3, 10))),
            ("B", "a", (F(2, 5), F(3, 10), F(1, 2))),
            ("B", "a", (F(7, 10), F(1, 10), F(4, 5))),
        ],
        (5.0, 6.0, 7.0, 90, 100, 90),
        {frozenset(("A", "B")): 3.3, frozenset(("B",)): 3.3},
        "1~1@2.2.2@2.3_2~3",
    ),
    "p1-general": (
        1,
        [("A", "a", (F(1, 10), F(1, 5), F(3, 10))), ("B", "a", (F(2, 5), F(9, 20), F(3, 5)))],
        (4.0, 4.5, 5.0, 80, 95, 105),
        {frozenset(("A", "B")): 2.6, frozenset(("A",)): 4.1},
        None,
    ),
    "pminus1-centre": (
        2,
        [("A", "a", ()), ("B", "i", (F(1, 5), F(3, 10), F(1, 10)))],
        (4.0, 4.5, 5.0, 80, 95, 105),
        {frozenset(("A", "B")): 2.2, frozenset(("B",)): 2.6},
        None,
    ),
}


@cache
def _case(name: str) -> tuple[int, list[Any], list[Any], list[list[float]], list[Any], dict[Any, float]]:
    """Return ``(it, sites, refs, lattice, rows, distances)`` of one named case."""
    it, spec, cell, cutoffs, _ = CASES[name]
    if spec is None:
        loaded = load_structure(K2AGSBS4)
        assert loaded is not None
        it, sites, refs, lattice = loaded
    else:
        sites, refs, lattice = handmade(it, spec, cell)
    rows, dist = raw_rows(_tables(it), sites, refs, lattice, cutoffs)
    return it, sites, refs, lattice, rows, dist


@cache
def _oracle(name: str) -> dict[str, Any]:
    it, sites, _, _, rows, _ = _case(name)
    return oracle.canonical_net(it, sites, rows)


def _assert_matches_oracle(result: _CanonicalNet, expected: dict[str, Any]) -> None:
    assert list(result.slots) == list(expected["slots"])
    assert list(result.codes) == list(expected["codes"])
    assert list(result.groups) == list(expected["groups"])
    # D3 (review R1): the smallest name tuple over the realisations, not the oracle's first one
    assert min(r.classes for r in result.realisations) == min(tuple(r["classes"]) for r in expected["realisations"])


def _profile(normal: tuple[Any, ...], extended: tuple[Any, ...]) -> list[Any]:
    return [*normal, *(pytest.param(value, marks=pytest.mark.extended) for value in extended)]


# ---------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------
TABLE_GROUPS = (1, 2, 34, 47, 225, 227)


@pytest.mark.parametrize("it", _profile(TABLE_GROUPS, tuple(n for n in range(1, 231) if n not in TABLE_GROUPS)))
def test_tables_identity_and_site_symmetry_orders(it: int) -> None:
    tables = _tables(it)
    assert tables.operations[0].is_identity()
    for position in tables.spacegroup.wyckoff:
        info = tables.letter(position.letter)
        assert len(info.H) * info.multiplicity == tables.nops


@pytest.mark.parametrize("it", _profile((1, 4, 34, 62, 167, 186), (2, 225, 227)))
def test_cosets_match_oracle(it: int) -> None:
    tables, reference = _tables(it), oracle.tables(it)
    assert tables.mulk == reference.mulk and tables.invk == reference.invk
    assert len(tables.cosets) == len(reference.cosets)
    for mine, theirs in zip(tables.cosets, reference.cosets, strict=True):
        assert mine.action == theirs["action"]
        assert list(mine.relabel) == list(theirs["relabel"])
        assert mine.P == theirs["P"]
        for letter, (target, b) in theirs["letters"].items():
            assert mine.letter(letter) == (target, b)
            assert target == compile_wyckoff_action(tables.spacegroup, mine.action, letter).target_letter
    for position in tables.spacegroup.wyckoff:
        theirs = reference.letter(position.letter)
        assert reps(tables, position.letter) == theirs["reps"]
        assert list(tables.letter(position.letter).lattice) == list(theirs["lattice"])
        if not theirs["general"]:  # general moves are only the oracle's scramble aid
            assert tables.moves(position.letter) == theirs["moves"]


def _closes(it: int) -> bool:
    """Whether every product of two kept cosets is again a kept coset modulo the group."""
    tables = _tables(it)
    actions = [(coset._matrix, coset._vector) for coset in tables.cosets]
    forms = {tables._coset_form(P, p) for P, p in actions}
    assert len(forms) == len(actions)
    return all(
        tables._coset_form(_mat_mul(P1, P2), _apply(P1, p1, p2)) in forms for P1, p1 in actions for P2, p2 in actions
    )


CLOSURE_GROUPS = (1, 2, 14, 34, 62)


@pytest.mark.parametrize("it", _profile(CLOSURE_GROUPS, tuple(n for n in range(1, 231) if n not in CLOSURE_GROUPS)))
def test_tabulated_actions_close_under_composition(it: int) -> None:
    assert _closes(it) == (it not in NON_CLOSED)


REPS_GROUPS = (2, 34, 62, 139, 227)


@pytest.mark.parametrize("it", _profile(REPS_GROUPS, tuple(n for n in range(1, 231) if n not in REPS_GROUPS)))
def test_gauge_move_classes_count_flat_preserving_operations(it: int) -> None:
    # D6: one class per operation mapping the flat onto itself, modulo the site-symmetry group
    tables = _tables(it)
    for position in tables.spacegroup.wyckoff:
        info = tables.letter(position.letter)
        keeping = 0
        for W, w in zip(tables.W, tables.w, strict=True):
            image = _apply(W, w, info.ref)
            gap = [v - sum(r[x] * image[x] for x in range(3)) for r, v in zip(info.normal, info.offset, strict=True)]
            keeping += all(g.denominator == 1 for g in gap)
        assert len(tables.reps(position.letter)) * len(info.H) == keeping


def test_group_34_operation_numbering() -> None:
    assert [op.to_xyz() for op in _tables(34).operations] == [
        "x,y,z",
        "-x,-y,z",
        "-x+1/2,y+1/2,z+1/2",
        "x+1/2,-y+1/2,z+1/2",
    ]


def test_group_227_flat_test_is_exact_on_non_idempotent_chart() -> None:
    tables = _tables(227)
    info = tables.letter("f")
    assert info.on_flat(info.ref)
    assert tables.spacegroup.wyckoff_position("f").representative.operation.to_xyz() == "-x,7/8,7/8"
    assert info.on_flat((info.ref[0] + F(1, 7), info.ref[1], info.ref[2]))
    assert not info.on_flat((info.ref[0], info.ref[1] + F(1, 7), info.ref[2]))


# ---------------------------------------------------------------------------------------------
# canonical nets
# ---------------------------------------------------------------------------------------------
# smallest species-name tuple over the realisations (D3, review R1)
EXPECTED_CLASSES = {
    "k2agsbs4-frame": ("S", "K", "Ag", "Sb"),
    "k2agsbs4-full": ("S", "K", "Ag", "Sb"),
    "nacl": ("Cl", "Na"),
    "nacl-exchanged": ("Cl", "Na"),
    "diamond": ("C",),
    "wurtzite": ("S", "Zn"),
    "pdse2": ("Se", "Pd"),
    "p21-chain": ("B", "A"),
}
# the oracle needs 20-120 s to build its tables for these groups (225, 227, 2)
SLOW_ORACLE = ("nacl", "nacl-exchanged", "diamond", "pminus1-centre")


@pytest.mark.parametrize("name", list(CASES))
def test_case_net_and_round_trip(name: str) -> None:
    it, sites, refs, lattice, rows, dist = _case(name)
    result = _canonical_net(it, sites, rows)
    if CASES[name][4] is not None:
        assert oracle.render(result.slots, result.codes) == CASES[name][4]
    if name in EXPECTED_CLASSES:
        assert min(r.classes for r in result.realisations) == EXPECTED_CLASSES[name]
    tables = _tables(it)
    for real in result.realisations[:2]:
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
        )


@pytest.mark.parametrize("name", _profile(tuple(n for n in CASES if n not in SLOW_ORACLE), SLOW_ORACLE))
def test_case_matches_oracle(name: str) -> None:
    it, sites, _, _, rows, _ = _case(name)
    _assert_matches_oracle(_canonical_net(it, sites, rows), _oracle(name))


def test_k2agsbs4_groups_and_species_order() -> None:
    it, sites, _, _, rows, _ = _case("k2agsbs4-frame")
    result = _canonical_net(it, sites, rows)
    assert result.groups == (("c", "c", "c", "c"), ("a", "b", "c"), ("a", "b"), ("c",))
    assert min(r.classes for r in result.realisations) == ("S", "K", "Ag", "Sb")
    chain = _canonical_net(*_case("p21-chain")[:2], _case("p21-chain")[4])
    assert min(r.classes for r in chain.realisations) == ("B", "A")


@pytest.mark.parametrize("trials", [3, pytest.param(40, marks=pytest.mark.extended)])
@pytest.mark.parametrize("name", list(CASES))
def test_scrambled_descriptions_give_the_same_net(name: str, trials: int) -> None:
    it, sites, _, _, rows, _ = _case(name)
    reference = _canonical_net(it, sites, rows)
    tables, rng = _tables(it), random.Random(f"{name}-{trials}")
    for _ in range(trials):
        sites2, rows2 = scramble(tables, sites, rows, rng, it not in NON_CLOSED)
        result = _canonical_net(it, sites2, rows2)
        assert (result.slots, result.codes, result.groups) == (reference.slots, reference.codes, reference.groups)


def test_pdse2_is_one_string_over_150_scrambles() -> None:
    # the case that broke the box-limited search of the first brief
    it, sites, _, _, rows, _ = _case("pdse2")
    tables, rng = _tables(it), random.Random("pdse2")
    first = _canonical_net(it, sites, rows)
    seen = {oracle.render(first.slots, first.codes)}
    for _ in range(150):
        result = _canonical_net(it, *scramble(tables, sites, rows, rng))
        seen.add(oracle.render(result.slots, result.codes))
    assert seen == {"1~1@2@455.2.2@4@455"}


FIXTURE_NORMAL = (
    "cod/1000008.cif",
    "structreading/34.cif",
    "structreading/61.cif",
    "structreading/62.cif",
    "structreading/186.cif",
)
FIXTURE_EXTENDED = (
    "cod/1509425.cif",
    # cod/1001844.cif (P1, 32 general sites, 51 rows) is left out: _minimal_numberings does not
    # finish within minutes (D8), in the oracle as in production
    *(f"structreading/{n}.cif" for n in (12, 14, 47, 129, 139, 148, 166, 194, 221, 225, 227, 229)),
)
# The box search of the first brief failed 31 % (34) and 22 % (61) of these draws: 25 draws catch it
# with probability above 0.99.
FIXTURE_DRAWS = {"structreading/34.cif": 25, "structreading/61.cif": 25}


@pytest.mark.parametrize("path", _profile(FIXTURE_NORMAL, FIXTURE_EXTENDED))
def test_fixture_matches_oracle_and_is_invariant(path: str) -> None:
    loaded = load_structure(str(FIXTURES / path))
    assert loaded is not None, "fixture must be an ordered element-pure standard-setting structure"
    it, sites, refs, lattice = loaded
    tables = _tables(it)
    rows, dist = raw_rows(tables, sites, refs, lattice, generic_cutoffs(tables, sites, refs, lattice))
    result = _canonical_net(it, sites, rows)
    _assert_matches_oracle(result, oracle.canonical_net(it, sites, rows))
    real = result.realisations[0]
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
    )
    rng = random.Random(path)
    for _ in range(FIXTURE_DRAWS.get(path, 2)):
        sites2, rows2 = scramble(tables, sites, rows, rng, it not in NON_CLOSED)
        again = _canonical_net(it, sites2, rows2)
        assert (again.slots, again.codes, again.groups) == (result.slots, result.codes, result.groups)


SWEEP_GROUPS = (4, 14, 19, 33, 34, 61, 62, 136, 186)
SWEEP_WIDE = tuple(sorted(random.Random(0).sample([n for n in range(3, 231) if n not in SWEEP_GROUPS], 30)))


@pytest.mark.parametrize(
    "it,nets,scrambles",
    [
        *((it, 2, 2) for it in SWEEP_GROUPS),
        *(pytest.param(it, 30, 4, marks=pytest.mark.extended) for it in SWEEP_GROUPS + SWEEP_WIDE),
    ],
)
def test_random_nets_are_description_invariant(it: int, nets: int, scrambles: int) -> None:
    def net(sites: Any, rows: Any) -> Any:
        result = _canonical_net(it, sites, rows)
        return result.slots, result.codes, result.groups

    rng = random.Random(f"sweep-{it}-{nets}")
    assert synthetic_sweep(_tables(it), net, rng, nets, scrambles, it not in NON_CLOSED) == 0


@pytest.mark.parametrize("it", _profile((4, 33, 34, 62, 186), (14, 19, 61, 136)))
def test_random_nets_match_oracle(it: int) -> None:
    rng = random.Random(f"oracle-{it}")
    for _ in range(3):
        sites, rows = random_net(_tables(it), rng)
        _assert_matches_oracle(_canonical_net(it, sites, rows), oracle.canonical_net(it, sites, rows))


# several components and tied classes, small enough for the oracle (P2, P3)
SMALL_NETS: dict[str, tuple[Any, ...]] = {
    "g4-two-components-four-tied": (
        4,
        [("A", "a"), ("B", "a"), ("C", "a"), ("D", "a")],
        [(0, 1, (1, 0, 0, 0)), (2, 3, (0, 1, 0, 0))],
    ),
    "g14-two-components": (
        14,
        [("A", "e"), ("B", "a"), ("C", "e"), ("D", "b")],
        [(0, 1, (0, 0, 0, 0)), (2, 3, (1, 0, 1, 0))],
    ),
    "g14-three-components": (
        14,
        [("A", "e"), ("B", "a"), ("C", "e"), ("D", "b"), ("E", "e"), ("F", "c")],
        [(0, 1, (0, 0, 0, 0)), (2, 3, (1, 0, 1, 0)), (4, 5, (2, 1, 0, 0)), (4, 4, (1, 0, 0, 1))],
    ),
    "g4-three-components": (
        4,
        [("A", "a"), ("B", "a"), ("C", "a"), ("D", "a"), ("E", "a"), ("E", "a")],
        [(0, 1, (1, 0, 0, 0)), (2, 3, (0, 1, 0, 0)), (4, 5, (1, 0, 1, 0))],
    ),
    "p21-three-tied-one-connected": (4, [("X", "a"), ("Y", "a"), ("Z", "a")], [(1, 1, (1, 0, 0, 0))]),
    "g14-three-tied-empty": (14, [("X", "e"), ("Y", "e"), ("Z", "e")], []),
    "g14-tied-two-site-classes": (
        14,
        [("X", "a"), ("X", "e"), ("Y", "a"), ("Y", "e"), ("Z", "a"), ("Z", "e")],
        [(2, 3, (0, 0, 0, 0))],
    ),
}


@pytest.mark.parametrize("name", list(SMALL_NETS))
def test_components_and_tied_classes_match_oracle(name: str) -> None:
    it, sites, rows = SMALL_NETS[name]
    _assert_matches_oracle(_canonical_net(it, sites, rows), oracle.canonical_net(it, sites, rows))


def test_disconnected_tied_components_keep_every_numbering_and_one_sorted_empty_realisation() -> None:
    # review P2/P3: five general-general bonds between ten tied one-site classes, and an empty P-1 net
    sites = [(f"X{i}", "e") for i in range(5)] + [(f"Y{i}", "e") for i in range(5)]
    result = _canonical_net(14, sites, [(i, 5 + i, (0, 0, 0, 0)) for i in range(5)])
    empty = _canonical_net(2, [(f"C{i}", "i") for i in range(7)], [])
    assert oracle.render(result.slots, result.codes) == "1~2_3~4_5~6_7~8_9~10"
    assert len(result.realisations) == 3840  # 2^5 5! numberings, all distinct
    assert [r.classes for r in empty.realisations] == [tuple(f"C{i}" for i in range(7))]


# ---------------------------------------------------------------------------------------------
# chirality, validation, limits
# ---------------------------------------------------------------------------------------------
# alpha-quartz in httk's standard P3_121 setting; O z = 1/3 - 0.12 there (Si-O 1.60 and 1.63 A)
QUARTZ = [("Si", "a", (F(47, 100),)), ("O", "c", (F(41, 100), F(27, 100), F(1, 3) - F(12, 100)))]
QUARTZ_CELL = (4.913, 4.913, 5.405, 90, 90, 120)
SI_O = {frozenset(("Si", "O")): 1.8}


def _enantiomorph(spec: list[tuple[str, str, tuple[F, ...]]]) -> list[tuple[str, str, tuple[F, ...]]]:
    """Map a 152 description through the inversion onto 154 with httk's proven Wyckoff actions."""
    source, target = Spacegroup.standard(152), Spacegroup.standard(154)
    out = []
    for species, letter, params in spec:
        action = compile_wyckoff_action(source, _INVERSION, letter, target_spacegroup=target)
        out.append((species, action.target_letter, tuple(action.apply(params).to_fractions())))
    return out


def test_quartz_enantiomorphs_share_a_net_only_without_preserved_chirality() -> None:
    nets = {}
    for it, spec in ((152, QUARTZ), (154, _enantiomorph(QUARTZ))):
        sites, refs, lattice = handmade(it, spec, QUARTZ_CELL)
        rows, _ = raw_rows(_tables(it), sites, refs, lattice, SI_O)
        assert len(rows) == 2
        kept = _canonical_net(it, sites, rows)
        flipped = _canonical_net(it, sites, rows, preserve_chirality=False)
        assert kept.it_number == it and flipped.it_number == 152
        nets[it] = (kept, flipped)
        if it == 152:
            _assert_matches_oracle(kept, oracle.canonical_net(it, sites, rows))
    assert nets[152][0] == nets[152][1]
    flipped152, flipped154 = nets[152][1], nets[154][1]
    assert (flipped152.slots, flipped152.codes, flipped152.groups) == (
        flipped154.slots,
        flipped154.codes,
        flipped154.groups,
    )


def test_invalid_rows_and_letters_are_rejected() -> None:
    sites = [("Na", "a"), ("Cl", "b")]
    with pytest.raises(ValueError, match="outside"):
        _canonical_net(225, sites, [(0, 2, (0, 0, 0, 0))])
    with pytest.raises(ValueError, match="outside"):
        _canonical_net(225, sites, [(-1, 0, (0, 0, 0, 0))])
    with pytest.raises(ValueError, match="Wyckoff letter"):
        _canonical_net(225, [("Na", "z")], [])
    with pytest.raises(ValueError, match="element"):
        _canonical_net(225, sites, [(0, 1, (192, 0, 0, 0))])


def test_tiny_budget_raises_limit_error() -> None:
    it, sites, _, _, rows, _ = _case("k2agsbs4-frame")
    _search.cache_clear()
    with _budget_scope(1e-9), pytest.raises(CanonicalizationLimitError):
        _canonical_net(it, sites, rows)
