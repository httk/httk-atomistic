"""Baseline shape detector for the bonds around one atom (private; protonet follow-up 5, part B).

A site's *shape* is the arrangement of the bonds of one of its atoms: the directions from the atom
to every atom it is bonded to.  The baseline detector looks at directions only, not bond lengths:

1. take the angle between every pair of bond directions and sort the angles;
2. for every ideal shape with the same number of bonds, the misfit is the root mean square
   difference between the sorted angles and the ideal shape's sorted angles (degrees), rounded
   to 1e-6;
3. the shape with the smallest misfit wins (ties: table order); a misfit above
   ``_IRREGULAR_ABOVE`` gives ``I<n>``;
4. two bonds: ``L2`` if the angle is at least ``_LINEAR_FROM``, else ``A2``;
5. a bond count without ideal shapes gives ``I<n>``; fewer than two bonds give no shape.

This is a port of the workspace reference ``docs/protonet_followup/shape_reference.py`` and must
reproduce it exactly (floats: geometry presentation only).  Some ideal shapes lie closer to each
other than the irregular limit in this measure (``SAPR8``/``DD8`` 9.65, ``COC12``/``ACOC12``
9.38, ``COC12``/``IC12`` 11.70, ``TP3``/``TPY3`` 13.00, ``CU8``/``HBPY8`` 14.49 degrees RMS);
the closest shape still wins deterministically.

Symbols are IUPAC polyhedral symbols without the hyphen where IUPAC defines one [IUPAC]_, the
SHAPE program's labels otherwise [SHAPE]_, and ``I<n>`` (irregular, ``n`` bonds) of our own:

========  ==========================================  =============================
Symbol    Name                                        Source
========  ==========================================  =============================
L2        linear                                      IUPAC L-2
A2        angular                                     IUPAC A-2
TP3       trigonal plane                              IUPAC TP-3
TPY3      trigonal pyramid                            IUPAC TPY-3
TS3       T-shape                                     IUPAC TS-3
T4        tetrahedron                                 IUPAC T-4
SP4       square plane                                IUPAC SP-4
SS4       see-saw                                     IUPAC SS-4
TBPY5     trigonal bipyramid                          IUPAC TBPY-5
SPY5      square pyramid                              IUPAC SPY-5
OC6       octahedron                                  IUPAC OC-6
TPR6      trigonal prism                              IUPAC TPR-6
PPY6      pentagonal pyramid                          SHAPE PPY-6
PBPY7     pentagonal bipyramid                        IUPAC PBPY-7
CU8       cube                                        IUPAC CU-8
SAPR8     square antiprism                            IUPAC SAPR-8
DD8       dodecahedron                                IUPAC DD-8 (SHAPE: TDD-8)
HBPY8     hexagonal bipyramid                         IUPAC HBPY-8
TPRS9     trigonal prism, square-face tricapped       IUPAC TPRS-9 (SHAPE: TCTPR-9)
COC12     cuboctahedron                               SHAPE COC-12
ACOC12    anticuboctahedron (triangular               SHAPE ACOC-12
          orthobicupola, J27)
IC12      icosahedron                                 SHAPE IC-12
I<n>      irregular, n bonds                          httk
========  ==========================================  =============================

The ideal geometries are the reference's own choices where neither source fixes one (for example
the trigonal pyramid at 107 degrees between bonds, the square pyramid at 100 degrees between apex
and base, and ``PPY6`` as an icosahedron vertex with its five neighbours and one outward bond).

.. [IUPAC] N. G. Connelly, T. Damhus, R. M. Hartshorn, A. T. Hutton (eds.), *Nomenclature of
   Inorganic Chemistry: IUPAC Recommendations 2005*, RSC Publishing, Cambridge, 2005,
   section IR-9.3.2.1 and Table IR-9.2 (p. 176).
.. [SHAPE] M. Llunell, D. Casanova, J. Cirera, P. Alemany, S. Alvarez, *SHAPE: Program for the
   Stereochemical Analysis of Molecular Fragments by Means of Continuous Shape Measures and
   Associated Tools*, User's Manual, Version 2.1, Universitat de Barcelona, March 2013,
   "List of reference shapes" (pp. 7-9).
"""

import itertools
import math
import re
from collections.abc import Iterable, Sequence
from typing import SupportsFloat

__all__: list[str] = []

_LINEAR_FROM = 165.0  # degrees: two bonds at least this far apart are linear (decision S3)
_IRREGULAR_ABOVE = 15.0  # degrees RMS: beyond this no ideal shape is assigned (decision S3)
# The shape a missing symbol in the label means, per bond count.  One table: a change is a nets
# version bump.  Two bonds: linear (maintainer decision S2, Nov 12 2026).
_DEFAULTS = {2: "L2", 3: "TP3", 4: "T4", 5: "TBPY5", 6: "OC6", 7: "PBPY7", 8: "CU8", 9: "TPRS9", 12: "COC12"}


def _unit(v: Sequence[float]) -> tuple[float, ...]:
    n = math.sqrt(sum(x * x for x in v))
    return tuple(x / n for x in v)


def _ring(n: int, polar: float, phase: float = 0.0, sign: int = 1) -> list[tuple[float, float, float]]:
    """N directions at ``polar`` degrees from +z (from -z with ``sign=-1``), the first at azimuth ``phase``."""
    t = math.radians(polar)
    return [
        (
            math.sin(t) * math.cos(math.radians(phase) + 2 * math.pi * k / n),
            math.sin(t) * math.sin(math.radians(phase) + 2 * math.pi * k / n),
            sign * math.cos(t),
        )
        for k in range(n)
    ]


_Z, _MZ = (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)
_TPY = math.degrees(math.asin(math.sqrt((1 - math.cos(math.radians(107.0))) / 1.5)))  # three bonds 107 degrees apart
_TPR = math.degrees(math.atan2(2 / math.sqrt(7), math.sqrt(3 / 7)))  # trigonal prism with square faces
_SAPR = math.degrees(math.atan2(1.0, 2**-0.75))  # square antiprism, equal edges
_ACOC = math.degrees(math.atan2(1 / math.sqrt(3), math.sqrt(2 / 3)))
_G = (1 + math.sqrt(5)) / 2
_ICO = [
    _unit(v)
    for v in [(0, a, b * _G) for a in (1, -1) for b in (1, -1)]
    + [(a, b * _G, 0) for a in (1, -1) for b in (1, -1)]
    + [(a * _G, 0, b) for a in (1, -1) for b in (1, -1)]
]
_V0 = _ICO[0]
_ADJ = [v for v in _ICO if 0.4 < sum(x * y for x, y in zip(v, _V0, strict=True)) < 0.5]
_DDA, _DDB = math.radians(36.85), math.radians(69.46)  # Hoard-Silverton dodecahedron

# bond count -> ordered {symbol: ideal bond directions}; the order breaks ties
_IDEAL: dict[int, dict[str, Sequence[Sequence[float]]]] = {
    3: {"TP3": _ring(3, 90), "TPY3": _ring(3, _TPY), "TS3": [(1, 0, 0), (-1, 0, 0), (0, 1, 0)]},
    4: {
        "T4": [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
        "SP4": _ring(4, 90),
        "SS4": [_Z, _MZ, (1, 0, 0), (-0.5, math.sqrt(3) / 2, 0)],
    },
    5: {"TBPY5": [_Z, _MZ, *_ring(3, 90)], "SPY5": [_Z, *_ring(4, 100)]},
    6: {
        "OC6": [_Z, _MZ, *_ring(4, 90)],
        "TPR6": _ring(3, _TPR) + _ring(3, _TPR, 0, -1),
        # an icosahedron vertex: its five neighbours on the icosahedron and one bond pointing outwards
        "PPY6": [tuple(a - b for a, b in zip(v, _V0, strict=True)) for v in _ADJ] + [_V0],
    },
    7: {"PBPY7": [_Z, _MZ, *_ring(5, 90)]},
    8: {
        "CU8": [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)],
        "SAPR8": _ring(4, _SAPR) + _ring(4, _SAPR, 45, -1),
        "DD8": [(math.sin(_DDA) * s, 0, math.cos(_DDA)) for s in (1, -1)]
        + [(0, math.sin(_DDA) * s, -math.cos(_DDA)) for s in (1, -1)]
        + [(0, math.sin(_DDB) * s, math.cos(_DDB)) for s in (1, -1)]
        + [(math.sin(_DDB) * s, 0, -math.cos(_DDB)) for s in (1, -1)],
        "HBPY8": [_Z, _MZ, *_ring(6, 90)],
    },
    9: {"TPRS9": _ring(3, 45) + _ring(3, 45, 0, -1) + _ring(3, 90, 60)},
    12: {
        "COC12": [p for a in (1, -1) for b in (1, -1) for p in ((a, b, 0), (a, 0, b), (0, a, b))],
        "ACOC12": _ring(6, 90) + _ring(3, _ACOC, 30) + _ring(3, _ACOC, 30, -1),
        "IC12": list(_ICO),
    },
}

_KNOWN = frozenset({"L2", "A2"}.union(*_IDEAL.values()))
_SYMBOL = re.compile(r"[A-Z]+([1-9][0-9]*)")
_IRREGULAR = re.compile(r"I([1-9][0-9]*)")


def _sorted_angles(vectors: Sequence[Sequence[float]]) -> list[float]:
    units = [_unit(v) for v in vectors]
    return sorted(
        math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b, strict=True))))))
        for a, b in itertools.combinations(units, 2)
    )


_IDEAL_ANGLES = {n: {name: _sorted_angles(vs) for name, vs in shapes.items()} for n, shapes in _IDEAL.items()}


def _misfit(angles: Sequence[float], ideal: Sequence[float]) -> float:
    return round(math.sqrt(sum((a - b) ** 2 for a, b in zip(angles, ideal, strict=True)) / len(angles)), 6)


def _detect(vectors: Iterable[Iterable[SupportsFloat]]) -> tuple[str | None, float | None]:
    """Return the baseline shape of the bonds around one atom and its misfit.

    :param vectors: Cartesian vectors from the atom to each atom it is bonded to, each three
        values convertible with ``float`` (bond lengths do not matter).
    :return: ``(symbol, misfit)``: ``(None, None)`` for fewer than two bonds, ``("L2" | "A2",
        None)`` for two, ``("I<n>", None)`` for a count without ideal shapes, otherwise the
        closest ideal symbol (``I<n>`` above the irregular limit) and its misfit in degrees RMS.
    :raises ValueError: if a vector does not have three components or has zero length.
    """
    floats = [tuple(float(x) for x in v) for v in vectors]
    for v in floats:
        if len(v) != 3 or not any(v):
            raise ValueError(f"a bond vector must be three components, not all zero; got {v!r}")
    n = len(floats)
    if n < 2:
        return None, None
    angles = _sorted_angles(floats)
    if n == 2:
        return ("L2" if round(angles[0], 6) >= _LINEAR_FROM else "A2"), None
    if n not in _IDEAL_ANGLES:
        return f"I{n}", None
    best_name, best = "", math.inf
    for name, ideal in _IDEAL_ANGLES[n].items():  # table order breaks ties
        m = _misfit(angles, ideal)
        if m < best:
            best_name, best = name, m
    return (best_name if best <= _IRREGULAR_ABOVE else f"I{n}"), best


def _bond_count(symbol: str) -> int:
    """Return the bond count a shape symbol ends in.

    :param symbol: uppercase letters followed by the bond count, for example ``OC6`` or ``I10``.
    :return: the bond count.
    :raises ValueError: if ``symbol`` is not letters then a number without leading zeros.
    """
    match = _SYMBOL.fullmatch(symbol)
    if match is None:
        raise ValueError(f"not a shape symbol: {symbol!r}")
    return int(match.group(1))


def _is_known(symbol: str) -> bool:
    """Return whether ``symbol`` is a symbol of the table or ``I<n>`` with ``n >= 2``.

    :param symbol: the shape symbol.
    :return: ``True`` for a known symbol.
    """
    if symbol in _KNOWN:
        return True
    match = _IRREGULAR.fullmatch(symbol)
    return match is not None and int(match.group(1)) >= 2


def _written(symbol: str | None) -> str:
    """Return the symbol as written in the label: empty for no shape or the default of its count.

    :param symbol: the shape symbol, or ``None`` for a site without a shape.
    :return: ``symbol``, or ``""`` where the label omits it.
    :raises ValueError: if ``symbol`` is not a shape symbol.
    """
    if symbol is None:
        return ""
    return "" if _DEFAULTS.get(_bond_count(symbol)) == symbol else symbol
