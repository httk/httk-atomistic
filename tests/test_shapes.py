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
    _bond_count,
    _detect,
    _is_known,
    _misfit,
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
    SAPR = math.degrees(math.atan2(1.0, 2**-0.75))
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
    ring = _ref_ring
    return {
        3: {"TP3": ring(3, 90), "TPY3": ring(3, TPY), "TS3": [(1, 0, 0), (-1, 0, 0), (0, 1, 0)]},
        4: {"T4": [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], "SP4": ring(4, 90),
            "SS4": [Z, MZ, (1, 0, 0), (-0.5, math.sqrt(3) / 2, 0)]},
        5: {"TBPY5": [Z, MZ] + ring(3, 90), "SPY5": [Z] + ring(4, 100)},
        6: {"OC6": [Z, MZ] + ring(4, 90), "TPR6": ring(3, TPR) + ring(3, TPR, 0, -1),
            "PPY6": [tuple(a - b for a, b in zip(v, V0)) for v in ADJ] + [V0]},
        7: {"PBPY7": [Z, MZ] + ring(5, 90)},
        8: {"CU8": [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)],
            "SAPR8": ring(4, SAPR) + ring(4, SAPR, 45, -1),
            "DD8": [(math.sin(DDA) * s, 0, math.cos(DDA)) for s in (1, -1)]
                   + [(0, math.sin(DDA) * s, -math.cos(DDA)) for s in (1, -1)]
                   + [(0, math.sin(DDB) * s, math.cos(DDB)) for s in (1, -1)]
                   + [(math.sin(DDB) * s, 0, -math.cos(DDB)) for s in (1, -1)],
            "HBPY8": [Z, MZ] + ring(6, 90)},
        9: {"TPRS9": ring(3, 45) + ring(3, 45, 0, -1) + ring(3, 90, 60)},
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
        2: "L2",
        3: "TP3",
        4: "T4",
        5: "TBPY5",
        6: "OC6",
        7: "PBPY7",
        8: "CU8",
        9: "TPRS9",
        12: "COC12",
    }
    for count, symbol in _DEFAULTS.items():
        assert _bond_count(symbol) == count and _is_known(symbol)


def test_written() -> None:
    assert _written(None) == ""
    assert _written("L2") == "" and _written("A2") == "A2"
    assert _written("T4") == "" and _written("SP4") == "SP4" and _written("I4") == "I4"
    assert _written("COC12") == "" and _written("IC12") == "IC12" and _written("I10") == "I10"


def test_bond_count_and_known() -> None:
    assert _bond_count("TPRS9") == 9 and _bond_count("I10") == 10 and _bond_count("ACOC12") == 12
    for bad in ("", "T", "4", "t4", "T04", "T4X", "T-4", "T0"):
        with pytest.raises(ValueError, match="not a shape symbol"):
            _bond_count(bad)
    for symbol in ("L2", "A2", "TS3", "SS4", "SPY5", "PPY6", "HBPY8", "ACOC12", "I2", "I7", "I13"):
        assert _is_known(symbol)
    for symbol in ("I0", "I1", "I02", "T5", "XX4", "L-2", "l2", ""):
        assert not _is_known(symbol)


# --- Pinned near pairs (degrees RMS between ideal shapes of one count). ---


@pytest.mark.parametrize(
    ("count", "a", "b", "expected"),
    [(8, "SAPR8", "DD8", 9.65), (12, "COC12", "ACOC12", 9.38), (12, "COC12", "IC12", 11.70),
     (3, "TP3", "TPY3", 13.00), (8, "CU8", "HBPY8", 14.49)],
)  # fmt: skip
def test_near_pairs(count: int, a: str, b: str, expected: float) -> None:
    assert round(_misfit(_IDEAL_ANGLES[count][a], _IDEAL_ANGLES[count][b]), 2) == expected


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
