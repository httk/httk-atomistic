"""Baseline shape detector: the reference self-check, the thresholds and an exact cross-check."""

import itertools
import math
import random
from fractions import Fraction

import pytest

from httk.atomistic.symmetry import shapes
from httk.atomistic.symmetry.shapes import (
    _DEFAULTS,
    _IDEAL,
    _IDEAL_ANGLES,
    _NAMED_SYMBOLS,
    _bond_count,
    _check_shape,
    _detect,
    _exchange_symbols,
    _from_exchange,
    _is_known,
    _misfit,
    _to_exchange,
    _written,
)

# --- Inline copy of the reference math (workspace docs/protonet_followup/shape_reference.py), verbatim but
# for formatting and names.  Tests must not depend on the workspace docs, so the reference lives here. ---


def _ref_unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def _ref_ring(n, polar, phase=0.0, sign=1):
    t = math.radians(polar)
    return [
        (
            math.sin(t) * math.cos(math.radians(phase) + 2 * math.pi * k / n),
            math.sin(t) * math.sin(math.radians(phase) + 2 * math.pi * k / n),
            sign * math.cos(t),
        )
        for k in range(n)
    ]


def _ref_ideal():
    Z, MZ = (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)
    TPY = math.degrees(math.asin(math.sqrt((1 - math.cos(math.radians(107.0))) / 1.5)))
    TPR = math.degrees(math.atan2(2 / math.sqrt(7), math.sqrt(3 / 7)))
    G = (1 + math.sqrt(5)) / 2
    ICO = [
        _ref_unit(v)
        for v in [(0, a, b * G) for a in (1, -1) for b in (1, -1)]
        + [(a, b * G, 0) for a in (1, -1) for b in (1, -1)]
        + [(a * G, 0, b) for a in (1, -1) for b in (1, -1)]
    ]
    V0 = ICO[0]
    ADJ = [v for v in ICO if 0.4 < sum(x * y for x, y in zip(v, V0)) < 0.5]
    DDA, DDB = math.radians(36.85), math.radians(69.46)
    CTPR_A = math.degrees(math.atan2(0.735247575071 * math.sqrt(2), 0.203750780644 + 0.050373370753))
    CTPR_B = math.degrees(math.atan2(0.660960557032, -0.892328424325 + 0.050373370753))
    ring = _ref_ring
    return {
        3: {"TP3": ring(3, 90), "TPY3": ring(3, TPY), "TS3": [(1, 0, 0), (-1, 0, 0), (0, 1, 0)]},
        4: {"T4": [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], "SP4": ring(4, 90),
            "SS4": [Z, MZ, (1, 0, 0), (-0.5, math.sqrt(3) / 2, 0)]},
        5: {"TBPY5": [Z, MZ] + ring(3, 90), "SPY5": [Z] + ring(4, 100)},
        6: {"OC6": [Z, MZ] + ring(4, 90), "TPR6": ring(3, TPR) + ring(3, TPR, 0, -1),
            "PPY6": [tuple(a - b for a, b in zip(v, V0)) for v in ADJ] + [V0]},
        7: {"PBPY7": [Z, MZ] + ring(5, 90),
            "TPRS7": [Z] + ring(4, CTPR_A, 45) + ring(2, CTPR_B)},
        8: {"CU8": [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)],
            "DD8": [(math.sin(DDA) * s, 0, math.cos(DDA)) for s in (1, -1)]
                   + [(0, math.sin(DDA) * s, -math.cos(DDA)) for s in (1, -1)]
                   + [(0, math.sin(DDB) * s, math.cos(DDB)) for s in (1, -1)]
                   + [(math.sin(DDB) * s, 0, -math.cos(DDB)) for s in (1, -1)],
            "HBPY8": [Z, MZ] + ring(6, 90),
            "TPRT8": ring(3, TPR) + ring(3, TPR, 0, -1) + [Z, MZ]},
        9: {"TPRS9": ring(3, 45) + ring(3, 45, 0, -1) + ring(3, 90, 60),
            "HBPY9": [Z, MZ] + ring(7, 90)},
        12: {"COC12": [p for a in (1, -1) for b in (1, -1) for p in ((a, b, 0), (a, 0, b), (0, a, b))],
             "ACOC12": ring(6, 90) + ring(3, math.degrees(math.atan2(1 / math.sqrt(3), math.sqrt(2 / 3))), 30)
                       + ring(3, math.degrees(math.atan2(1 / math.sqrt(3), math.sqrt(2 / 3))), 30, -1),
             "IC12": list(ICO)},
    }  # fmt: skip


def _ref_sorted_angles(vectors):
    units = [_ref_unit(v) for v in vectors]
    return sorted(
        math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))))
        for a, b in itertools.combinations(units, 2)
    )


def _ref_misfit(angles, ideal):
    return round(math.sqrt(sum((a - b) ** 2 for a, b in zip(angles, ideal)) / len(angles)), 6)


_REF_IDEAL = _ref_ideal()
_REF_IDEAL_ANGLES = {n: {name: _ref_sorted_angles(vs) for name, vs in s.items()} for n, s in _REF_IDEAL.items()}


def _ref_detect(vectors):
    n = len(vectors)
    if n < 2:
        return None, None
    angles = _ref_sorted_angles(vectors)
    if n == 2:
        return ("L2" if round(angles[0], 6) >= 165.0 else "A2"), None
    if n not in _REF_IDEAL_ANGLES:
        return f"I{n}", None
    best_name, best = None, None
    for name, ideal in _REF_IDEAL_ANGLES[n].items():
        m = _ref_misfit(angles, ideal)
        if best is None or m < best:
            best_name, best = name, m
    return (best_name if best <= 15.0 else f"I{n}"), best


# --- Reference self-check, ported. ---


def test_every_ideal_shape_detects_as_itself() -> None:
    assert list(_IDEAL) == [3, 4, 5, 6, 7, 8, 9, 12]
    for n, table in _IDEAL.items():
        for name, vectors in table.items():
            assert len(vectors) == n
            assert _detect(vectors) == (name, 0.0)


def test_thresholds_and_lengths() -> None:
    assert shapes._LINEAR_FROM == 165.0 and shapes._IRREGULAR_ABOVE == 15.0
    assert _detect([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)])[0] == "T4"
    # a regular trigonal plane given with unequal bond lengths
    assert _detect([(2, 0, 0), (-0.5, math.sqrt(3) / 2, 0), (-1.5, -1.5 * math.sqrt(3), 0)])[0] == "TP3"
    assert _detect([(1, 0, 0), (-1, 0.2, 0)]) == ("L2", None)
    assert _detect([(1, 0, 0), (-1, 0.3, 0)]) == ("A2", None)
    # the 165 degree boundary is inclusive after rounding to 1e-6
    at = math.radians(165.0)
    assert _detect([(1, 0, 0), (math.cos(at), math.sin(at), 0)])[0] == "L2"
    below = math.radians(165.0 - 2e-6)
    assert _detect([(1, 0, 0), (math.cos(below), math.sin(below), 0)])[0] == "A2"


def _bend_t4(alpha: float) -> list[tuple[float, float, float]]:
    """A tetrahedron with one bond rotated by ``alpha`` radians about z."""
    v = (1.0, 1.0, 1.0)
    rotated = (v[0] * math.cos(alpha) - v[1] * math.sin(alpha), v[0] * math.sin(alpha) + v[1] * math.cos(alpha), v[2])
    return [rotated, (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]


def test_irregular_boundary_is_inclusive() -> None:
    # bisect the rotation where the best misfit crosses 15 degrees, then check both sides
    lo, hi = 0.0, math.pi / 2
    for _ in range(200):
        mid = (lo + hi) / 2
        misfit = _detect(_bend_t4(mid))[1]
        assert misfit is not None
        lo, hi = (mid, hi) if misfit <= 15.0 else (lo, mid)
    inside, outside = _detect(_bend_t4(lo)), _detect(_bend_t4(hi))
    assert inside[1] is not None and outside[1] is not None
    assert inside[1] <= 15.0 < outside[1]
    assert inside[0] != "I4" and outside[0] == "I4"
    # exactly 15.0 after rounding still names a shape
    angles = list(_IDEAL_ANGLES[3]["TP3"])
    shift = 15.0 * math.sqrt(3)  # one of three angles off by 15*sqrt(3) gives RMS 15
    assert _misfit([angles[0], angles[1], angles[2] + shift], angles) == 15.0


def test_counts_without_ideal_shapes_and_too_few_bonds() -> None:
    assert _detect([]) == (None, None)
    assert _detect([(1, 0, 0)]) == (None, None)
    rng = random.Random(1)
    for n in (10, 11, 13, 14):
        vectors = [(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]
        assert _detect(vectors) == (f"I{n}", None)
    assert _detect([(1, 0, 0)] * 10)[0] == "I10"


def test_invalid_vectors_raise() -> None:
    with pytest.raises(ValueError, match="three components"):
        _detect([(1, 0), (0, 1)])
    with pytest.raises(ValueError, match="three components"):
        _detect([(0, 0, 0), (0, 1, 0)])


def test_exact_inputs_convert_with_float() -> None:
    half = Fraction(1, 2)
    assert _detect([(1, 0, 0), (-half, 0, 0)]) == ("L2", None)
    assert _detect([(Fraction(1), 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]) == ("T4", 0.0)


# --- Symbols. ---


def test_defaults_table() -> None:
    assert _DEFAULTS == {
        2: "A2",
        3: "TP3",
        4: "T4",
        5: "SPY5",
        6: "OC6",
        7: "TPRS7",
        8: "DD8",
        9: "TPRS9",
        12: "COC12",
    }
    for count, symbol in _DEFAULTS.items():
        assert _bond_count(symbol) == count and _is_known(symbol)


def test_written() -> None:
    assert _written(None) == ""
    assert _written("A2") == "" and _written("L2") == "L2"
    assert _written("T4") == "" and _written("SP4") == "SP4" and _written("I4") == "I4"
    assert _written("COC12") == "" and _written("IC12") == "IC12" and _written("I10") == "I10"
    assert _written("SPY5") == "" and _written("TBPY5") == "TBPY5"
    assert _written("TPRS7") == "" and _written("PBPY7") == "PBPY7"
    assert _written("DD8") == "" and _written("CU8") == "CU8" and _written("TPRT8") == "TPRT8"


def test_bond_count_and_known() -> None:
    assert _bond_count("TPRS9") == 9 and _bond_count("I10") == 10 and _bond_count("ACOC12") == 12
    for bad in ("", "T", "4", "t4", "T04", "T4X", "T-4", "T0"):
        with pytest.raises(ValueError, match="not a shape symbol"):
            _bond_count(bad)
    known = ("L2", "A2", "TS3", "SS4", "SPY5", "PPY6", "TPRS7", "HBPY8", "TPRT8", "HBPY9", "ACOC12", "I3", "I7", "I13")
    for symbol in (*known, "SAPR8", "SPY4", "I121", "I2000"):
        assert _is_known(symbol)
    for symbol in ("I0", "I1", "I2", "I02", "I03", "I", "T5", "XX4", "L-2", "l2", "sapr8", ""):
        assert not _is_known(symbol)


# --- Follow-up 8: the supplied vocabulary and the exchange symbols. ---


def test_supplied_symbols() -> None:
    assert len(_NAMED_SYMBOLS) == len(set(_NAMED_SYMBOLS)) == 29
    assert shapes._KNOWN <= set(_NAMED_SYMBOLS)  # the detector's table is part of the vocabulary
    for symbol in (*_NAMED_SYMBOLS, "I3", "I13", "I121", "I2000"):
        count = _bond_count(symbol)
        _check_shape(symbol, count)
        for wrong in (count - 1, count + 1):
            with pytest.raises(ValueError, match="bond"):
                _check_shape(symbol, wrong)
    for bad, count in (("I2", 2), ("I03", 3), ("X5", 5), ("sapr8", 8)):
        with pytest.raises(ValueError, match="unknown shape symbol"):
            _check_shape(bad, count)


def test_the_detector_returns_only_its_table_or_irregular() -> None:
    ideals = [vs for table in _IDEAL.values() for vs in table.values()] + [vs for vs, _, _ in _ABSENT_IDEALS.values()]
    for vectors in ideals:
        assert _detect(vectors)[0] in shapes._KNOWN
    rng = random.Random(8)
    for n in (13, 40, 122):
        assert _detect([(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]) == (
            f"I{n}",
            None,
        )


def test_exchange_symbols() -> None:
    assert _exchange_symbols() == (*_NAMED_SYMBOLS, "I")
    assert _to_exchange("I3") == _to_exchange("I122") == "I"
    assert _to_exchange("SAPR8") == "SAPR8" and _to_exchange("IC12") == "IC12"
    assert _from_exchange("I", 3) == "I3" and _from_exchange("I", 122) == "I122"
    assert (
        _from_exchange("I6", 6) == "I6"
        and _from_exchange("SAPR8", 8) == "SAPR8"
        and _from_exchange("IC12", 12) == "IC12"
    )
    for symbol in (*_NAMED_SYMBOLS, "I5", "I13"):
        assert _from_exchange(_to_exchange(symbol), _bond_count(symbol)) == symbol
    for symbol, count, message in (
        ("I5", 6, "is for 5 bonds, but the site has 6"),
        ("I", 2, "unknown shape symbol 'I2'"),
        ("I", 1, "unknown shape symbol 'I1'"),
        ("SAPR8", 6, "is for 8 bonds"),
        ("X5", 5, "unknown shape symbol"),
    ):
        with pytest.raises(ValueError, match=message):
            _from_exchange(symbol, count)


# --- Pinned near pairs (degrees RMS between ideal shapes of one count). ---


@pytest.mark.parametrize(
    ("count", "a", "b", "expected"),
    [(12, "COC12", "ACOC12", 9.38), (12, "COC12", "IC12", 11.70), (3, "TP3", "TPY3", 13.00),
     (8, "CU8", "HBPY8", 14.49)],
)  # fmt: skip
def test_near_pairs(count: int, a: str, b: str, expected: float) -> None:
    assert round(_misfit(_IDEAL_ANGLES[count][a], _IDEAL_ANGLES[count][b]), 2) == expected


def test_no_other_pair_within_15_degrees() -> None:
    near = {(a, b) for table in _IDEAL_ANGLES.values() for a, b in itertools.combinations(table, 2)
            if _misfit(table[a], table[b]) < 15.0}  # fmt: skip
    assert near == {("COC12", "ACOC12"), ("COC12", "IC12"), ("TP3", "TPY3"), ("CU8", "HBPY8")}


# --- Follow-up 7: the added shapes against SHAPE 2.1's coordinates, and the shapes absent from the table. ---

# SHAPE 2.1 reference polyhedra about their central atom (cosymlib ideal_structures_center.yaml at
# commit 90533c88, vertex minus central atom; ETBPY-8's central atom is at z = 1.5e-8, which is
# dropped)
_SHAPE_CTPR7 = (
    [(0, 0, 1.07040046758)]
    + [(a * 0.735247575071, b * 0.735247575071, 0.254124151397) for a in (1, -1) for b in (1, -1)]
    + [(a * 0.660960557032, 0, -0.841955053572) for a in (1, -1)]
)
_SHAPE_ETBPY8 = (
    [(a * 0.694365079077, 0, 0.801783719423) for a in (1, -1)]
    + [(a * 0.694365079077, b * 0.694365079077, -0.400891871284) for a in (1, -1) for b in (1, -1)]
    + [(1.06066015629, 0, 0), (-1.06066015629, 0, 0)]
)
_SHAPE_COC7 = [
    (0, 0, 1.070845406746),
    (0, -1.046937018035, 0.225017246487),
    (0.906674032650, 0.523468509017, 0.225017246487),
    (0.672964536915, -0.388536257092, -0.736795854290),
    (-0.906674032650, 0.523468509017, 0.225017246487),
    (-0.672964536915, -0.388536257092, -0.736795854290),
    (0, 0.777072514184, -0.736795854290),
]


def test_added_shapes_match_shape_geometry() -> None:
    assert _detect(_SHAPE_CTPR7) == ("TPRS7", 0.0)
    assert _detect(_SHAPE_ETBPY8) == ("TPRT8", 0.0)
    assert _detect([(0, 0, 1), (0, 0, -1)] + _ref_ring(7, 90)) == ("HBPY9", 0.0)


def test_existing_ideals_keep_their_detection() -> None:
    # the table before follow-up 7, less the dropped SAPR8: every ideal still detects as itself
    before = {
        3: ("TP3", "TPY3", "TS3"), 4: ("T4", "SP4", "SS4"), 5: ("TBPY5", "SPY5"), 6: ("OC6", "TPR6", "PPY6"),
        7: ("PBPY7",), 8: ("CU8", "DD8", "HBPY8"), 9: ("TPRS9",), 12: ("COC12", "ACOC12", "IC12"),
    }  # fmt: skip
    for n, names in before.items():
        assert tuple(_IDEAL[n])[: len(names)] == names
        for name in names:
            assert _detect(_IDEAL[n][name]) == (name, 0.0)


_TPR = math.degrees(math.atan2(2 / math.sqrt(7), math.sqrt(3 / 7)))
_SAPR = math.degrees(math.atan2(1.0, 2**-0.75))
# shape absent from the table -> (ideal bond directions, the class it is detected as, the misfit)
_ABSENT_IDEALS = {
    "SPY4": (_ref_ring(4, 100), "SP4", 11.632878),
    "OCF7": (_SHAPE_COC7, "TPRS7", 8.722937),
    "SAPR8": (_ref_ring(4, _SAPR) + _ref_ring(4, _SAPR, 45, -1), "DD8", 9.650191),
    "TPRS8": (_ref_ring(3, _TPR) + _ref_ring(3, _TPR, 0, -1) + _ref_ring(3, 90, 60)[:2], "DD8", 9.117421),
    "OCT8": ([(0, 0, 1), (0, 0, -1)] + _ref_ring(4, 90) + [(1, 1, 1), (-1, -1, -1)], "HBPY8", 3.446352),
}


@pytest.mark.parametrize("symbol", sorted(_ABSENT_IDEALS))
def test_absent_shapes_detect_as_their_class_and_are_accepted_when_supplied(symbol: str) -> None:
    vectors, cls, misfit = _ABSENT_IDEALS[symbol]
    assert symbol not in shapes._KNOWN and cls in shapes._KNOWN
    assert _detect(vectors) == (cls, misfit)
    _check_shape(symbol, _bond_count(symbol))


# --- Cross-check against the inline reference copy. ---


def test_matches_reference_on_perturbed_ideals() -> None:
    assert _IDEAL_ANGLES == _REF_IDEAL_ANGLES
    rng = random.Random(20261112)
    starts: list[list[tuple[float, ...]]] = [[(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)], [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]]
    starts += [[tuple(float(x) for x in v) for v in vs] for table in _REF_IDEAL.values() for vs in table.values()]
    starts += [[(1.0, 0.0, 0.0)] * n for n in (10, 11)]
    seen: set[str | None] = set()
    for i in range(2000):
        base = starts[i % len(starts)]
        scale = rng.choice((0.02, 0.1, 0.3, 0.6))
        # random bond lengths and directions
        vectors = [
            tuple(x * length + rng.gauss(0.0, scale) for x in v) for v in base for length in [rng.uniform(0.5, 2)]
        ]
        expected = _ref_detect(vectors)
        assert _detect(vectors) == expected, vectors
        seen.add(expected[0])
    # the sample reaches every symbol, both two-bond outcomes and irregular results
    assert seen >= {"L2", "A2", "I3", "I4", "I6", "I8", "I10", "I11", "I12"} | set(shapes._KNOWN) - {"I2"}
