"""Baseline shape detector for the bonds around one atom (private; protonet follow-up 5, part B).

A site's *shape* is the arrangement of the bonds of one of its atoms: the directions from the atom
to every atom it is bonded to.  The baseline detector looks at directions only, not bond lengths:

1. take the angle between every pair of bond directions and sort the angles;
2. for every ideal shape with the same number of bonds, the misfit is the root mean square
   difference between the sorted angles and the ideal shape's sorted angles (degrees), rounded
   to 1e-6;
3. the shape with the smallest misfit wins (ties: table order); a misfit above
   ``_IRREGULAR_ABOVE`` gives ``I``;
4. two bonds: ``L2`` if the angle is at least ``_LINEAR_FROM``, else ``A2``;
5. a bond count without ideal shapes gives ``I``; fewer than two bonds give no shape.

This is a port of the workspace reference ``docs/protonet_followup/shape_reference.py`` and must
reproduce it exactly (floats: geometry presentation only).  Some ideal shapes lie closer to each
other than the irregular limit in this measure (``COC12``/``ACOC12`` 9.38, ``COC12``/``IC12``
11.70, ``TP3``/``TPY3`` 13.00, ``CU8``/``HBPY8`` 14.49 degrees RMS); the closest shape still
wins deterministically.

Symbols are IUPAC polyhedral symbols without the hyphen where IUPAC defines one [IUPAC]_, the
SHAPE program's labels otherwise [SHAPE]_, and ``I`` (irregular) of our own:

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
TPRS7     trigonal prism, square-face monocapped      IUPAC TPRS-7 (SHAPE: CTPR-7)
CU8       cube                                        IUPAC CU-8
DD8       dodecahedron                                IUPAC DD-8 (SHAPE: TDD-8)
HBPY8     hexagonal bipyramid                         IUPAC HBPY-8
TPRT8     trigonal prism, triangular-face bicapped    IUPAC TPRT-8 (SHAPE: ETBPY-8)
TPRS9     trigonal prism, square-face tricapped       IUPAC TPRS-9 (SHAPE: TCTPR-9)
HBPY9     heptagonal bipyramid                        IUPAC HBPY-9 (SHAPE: HBPY-9)
COC12     cuboctahedron                               SHAPE COC-12
ACOC12    anticuboctahedron (triangular               SHAPE ACOC-12
          orthobicupola, J27)
IC12      icosahedron                                 SHAPE IC-12
I         irregular, three or more bonds              httk
========  ==========================================  =============================

The shapes that predate follow-up 7 keep the reference's own geometries (for example the
trigonal pyramid at 107 degrees between bonds).  Three of them deliberately differ from SHAPE's
reference coordinates [cosymlib]_: ``SPY5`` (square pyramid at 100 degrees between apex and
base) is 5.04 degrees from SHAPE ``SPY-5``; ``TPRS9`` (prism at 45 degrees from the axis) is 3.28
degrees from ``TCTPR-9``; and ``PPY6`` (an icosahedron vertex with its five neighbours and one
outward bond) is not SHAPE's ``PPY-6``, whose coordinates are detected as ``TPR6`` (14.56).
``TPRS7`` is SHAPE's reference geometry ``CTPR-7``, taken about the central atom from the
program's coordinates [cosymlib]_ (see also [Casanova2003]_); ``TPRT8`` and ``HBPY9`` are the
undistorted constructions, which equal SHAPE's ``ETBPY-8`` and ``HBPY-9``.

The table is what httk *recommends* and what the detector assigns.  Some IUPAC shapes are left out
of it: this measure barely tells each from a shape of the table, which its ideal geometry is detected
as and which is its *class*.  ``SPY4`` (four-bond square pyramid, 11.6 degrees away) is detected as
``SP4``; ``OCF7`` (face-monocapped octahedron, SHAPE ``COC-7``, 8.72) as ``TPRS7``; ``SAPR8``
(square antiprism, 9.65) and ``TPRS8`` (square-face bicapped trigonal prism, 9.12) as ``DD8``;
``OCT8`` (trans-bicapped octahedron, 3.45) as ``HBPY8``.  ``TPRS7`` is therefore the class of the
capped seven-bond arrangements, and ``DD8`` the class of the non-cubic eight-bond arrangements other
than the bipyramid ``HBPY8`` and the triangular-face bicapped prism ``TPRT8``.

A person or another program can tell such shapes apart, so a supplied symbol may be any of
``_NAMED_SYMBOLS`` (protonet follow-up 8): every IUPAC polyhedral symbol and the SHAPE labels of
the table, or ``I`` on a site with three or more bonds.  A symbol from the baseline (software
``httk_basic``) names a class, while a supplied symbol names the shape itself; the network's
``source`` and ``shapes_source`` tell the two apart.  ``I`` is never valid on two bonds (they are
``L2`` or ``A2``).  ``I`` carries no bond count, which follows from the network (protonet
follow-up 9): ``_SHAPE_SYMBOLS`` is the one list of symbols, in the code, labels, records and the
exchange formats (OPTIMADE, CIF) alike, and a former ``I<n>`` is refused with a message to write
``I``.
The defaults (``_DEFAULTS``) for five, seven and eight bonds are the most frequent shapes on an
ICSD sample (protonet follow-up 7): ``SPY5``, ``TPRS7`` and ``DD8``.

.. [IUPAC] N. G. Connelly, T. Damhus, R. M. Hartshorn, A. T. Hutton (eds.), *Nomenclature of
   Inorganic Chemistry: IUPAC Recommendations 2005*, RSC Publishing, Cambridge, 2005,
   section IR-9.3.2.1 and Table IR-9.2 (p. 176).
.. [SHAPE] M. Llunell, D. Casanova, J. Cirera, P. Alemany, S. Alvarez, *SHAPE: Program for the
   Stereochemical Analysis of Molecular Fragments by Means of Continuous Shape Measures and
   Associated Tools*, User's Manual, Version 2.1, Universitat de Barcelona, March 2013,
   "List of reference shapes" (pp. 7-9).
.. [Casanova2003] D. Casanova, P. Alemany, J. M. Bofill, S. Alvarez, "Shape and symmetry of
   heptacoordinate transition-metal complexes: structural trends", *Chem. Eur. J.* 9, 1281-1295
   (2003).
.. [cosymlib] Grup d'Estructura Electrònica i Simetria, cosymlib (MIT licence), file
   ``cosymlib/shape/ideal_structures_center.yaml``, commit ``90533c88``,
   https://github.com/GrupEstructuraElectronicaSimetria/cosymlib.
"""

import itertools
import math
import re
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any, SupportsFloat

if TYPE_CHECKING:  # the structure helpers import lazily: notation reads this module, and protonet imports it
    from httk.atomistic.models.structure.api import StructureAPI
    from httk.atomistic.models.structure.asu import FundamentalDomainStructure, WyckoffSite
    from httk.atomistic.models.structure.network import SiteBond
    from httk.atomistic.symmetry.protonet import _Element, _GroupTables
    from httk.atomistic.symmetry.spacegroup import Spacegroup

__all__: list[str] = []

_LINEAR_FROM = 165.0  # degrees: two bonds at least this far apart are linear (decision S3)
_IRREGULAR_ABOVE = 15.0  # degrees RMS: beyond this no ideal shape is assigned (decision S3)
# The shape a missing symbol in the label means, per bond count.  One table: a change is a nets
# version bump.  Two bonds: bent, settled from the follow-up 5 counts (decision S2, Nov 13 2026: 4% of
# carbon-free COD and 18% of ICSD inorganic two-bonded sites are linear).  Five, seven and eight bonds:
# the most frequent shape on ICSD (follow-up 7, Nov 13 2026).
_DEFAULTS = {2: "A2", 3: "TP3", 4: "T4", 5: "SPY5", 6: "OC6", 7: "TPRS7", 8: "DD8", 9: "TPRS9", 12: "COC12"}


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
# SHAPE CTPR-7 about its central atom (cosymlib ideal_structures_center.yaml, vertex minus central
# atom): polar angles from the cap of the capped square face (76.27) and the opposite edge (141.87)
_CTPR_A = math.degrees(math.atan2(0.735247575071 * math.sqrt(2), 0.203750780644 + 0.050373370753))
_CTPR_B = math.degrees(math.atan2(0.660960557032, -0.892328424325 + 0.050373370753))

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
    7: {"PBPY7": [_Z, _MZ, *_ring(5, 90)], "TPRS7": [_Z, *_ring(4, _CTPR_A, 45), *_ring(2, _CTPR_B)]},
    8: {
        "CU8": [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)],
        "DD8": [(math.sin(_DDA) * s, 0, math.cos(_DDA)) for s in (1, -1)]
        + [(0, math.sin(_DDA) * s, -math.cos(_DDA)) for s in (1, -1)]
        + [(0, math.sin(_DDB) * s, math.cos(_DDB)) for s in (1, -1)]
        + [(math.sin(_DDB) * s, 0, -math.cos(_DDB)) for s in (1, -1)],
        "HBPY8": [_Z, _MZ, *_ring(6, 90)],
        "TPRT8": [*_ring(3, _TPR), *_ring(3, _TPR, 0, -1), _Z, _MZ],
    },
    9: {"TPRS9": _ring(3, 45) + _ring(3, 45, 0, -1) + _ring(3, 90, 60), "HBPY9": [_Z, _MZ, *_ring(7, 90)]},
    12: {
        "COC12": [p for a in (1, -1) for b in (1, -1) for p in ((a, b, 0), (a, 0, b), (0, a, b))],
        "ACOC12": _ring(6, 90) + _ring(3, _ACOC, 30) + _ring(3, _ACOC, 30, -1),
        "IC12": list(_ICO),
    },
}

_KNOWN = frozenset({"L2", "A2"}.union(*_IDEAL.values()))  # the detector's table
# every named symbol a supplied shape may use: the IUPAC polyhedral symbols, then the SHAPE labels of the table
_NAMED_SYMBOLS = (
    *("L2", "A2", "TP3", "TPY3", "TS3", "T4", "SP4", "SPY4", "SS4", "TBPY5", "SPY5", "OC6", "TPR6", "PBPY7"),
    *("OCF7", "TPRS7", "CU8", "SAPR8", "DD8", "HBPY8", "OCT8", "TPRT8", "TPRS8", "TPRS9", "HBPY9"),
    *("PPY6", "COC12", "ACOC12", "IC12"),
)
# THE list of shape symbols (code, labels, records, the OPTIMADE enum, the CIF dictionary): ``I`` is irregular
_SHAPE_SYMBOLS = (*_NAMED_SYMBOLS, "I")
_SYMBOL = re.compile(r"[A-Z]+([1-9][0-9]*)")
_FORMER_IRREGULAR = re.compile(r"I(?:[3-9]|[1-9][0-9]+)")  # the valid I<n> of follow-ups 5 to 8 (n >= 3)


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
        None)`` for two, ``("I", None)`` for a count without ideal shapes, otherwise the
        closest ideal symbol (``I`` above the irregular limit) and its misfit in degrees RMS.
    :raises ValueError: if a vector does not have three components or has zero length.
    """
    floats = [tuple(float(x) for x in v) for v in vectors]
    for v in floats:
        if len(v) != 3 or not any(v):
            raise ValueError(f"a bond vector must be three components, not all zero; got {v!r}")
    return _detect_angles(len(floats), _sorted_angles(floats))


def _detect_angles(count: int, angles: Sequence[float]) -> tuple[str | None, float | None]:
    """Return the baseline shape and misfit of ``count`` bonds from their sorted pair angles.

    :param count: The number of bonds.
    :param angles: The ``count * (count - 1) / 2`` angles between the bond directions, in degrees,
        sorted ascending.
    :return: As :func:`_detect`.
    """
    if count < 2:
        return None, None
    if count == 2:
        return ("L2" if round(angles[0], 6) >= _LINEAR_FROM else "A2"), None
    if count not in _IDEAL_ANGLES:
        return "I", None
    best_name, best = "", math.inf
    for name, ideal in _IDEAL_ANGLES[count].items():  # table order breaks ties
        m = _misfit(angles, ideal)
        if m < best:
            best_name, best = name, m
    return (best_name if best <= _IRREGULAR_ABOVE else "I"), best


def _bond_count(symbol: str) -> int:
    """Return the bond count a named shape symbol ends in.

    :param symbol: uppercase letters followed by the bond count, for example ``OC6``.
    :return: the bond count.
    :raises ValueError: if ``symbol`` is not letters then a number without leading zeros (``I``
        has no number: its count comes from the network).
    """
    match = _SYMBOL.fullmatch(symbol)
    if match is None:
        raise ValueError(f"not a shape symbol with a bond count: {symbol!r}")
    return int(match.group(1))


def _is_known(symbol: str) -> bool:
    """Return whether ``symbol`` is in ``_SHAPE_SYMBOLS``.

    :param symbol: the shape symbol.
    :return: ``True`` for a valid symbol.
    """
    return symbol in _SHAPE_SYMBOLS


def _unknown(symbol: str) -> str:
    """Return the error message for a symbol that is not in ``_SHAPE_SYMBOLS``: a former ``I<n>`` says to write ``I``."""
    if _FORMER_IRREGULAR.fullmatch(symbol):
        return f"{symbol!r} is no longer a shape symbol; write 'I' for an irregular arrangement"
    return f"unknown shape symbol {symbol!r}"


def _written(symbol: str | None) -> str:
    """Return the symbol as written in the label: empty for no shape or the default of its count.

    :param symbol: the shape symbol, or ``None`` for a site without a shape.
    :return: ``symbol``, or ``""`` where the label omits it.
    """
    # each default is a named symbol ending in its own count, so membership is "the default of its count"
    return "" if symbol is None or symbol in _DEFAULTS.values() else symbol


def _check_shape(symbol: str, count: int) -> None:
    """Check a supplied shape symbol against the bond count of its site.

    :param symbol: The shape symbol.
    :param count: The site's bond count.
    :raises ValueError: If ``symbol`` is unknown (see ``_is_known``; a former ``I<n>`` says to
        write ``I``), the site has fewer than two bonds, ``I`` is on fewer than three, or a
        named symbol's number is not ``count``.
    """
    if not _is_known(symbol):
        raise ValueError(_unknown(symbol))
    if count < 2:
        raise ValueError(f"shape {symbol!r} given for a site with {count} bond(s); only two or more bonds have a shape")
    if symbol == "I":
        if count < 3:
            raise ValueError(f"shape 'I' (irregular) needs three or more bonds, but the site has {count}")
    elif _bond_count(symbol) != count:
        raise ValueError(f"shape {symbol!r} is for {_bond_count(symbol)} bonds, but the site has {count}")


def _bond_counts(
    tables: "_GroupTables", stabilisers: Sequence[Sequence["_Element"]], rows: Iterable[tuple[int, int, "_Element"]]
) -> tuple[int, ...]:
    """Return the number of distinct atoms bonded to the reference atom of each site.

    Each row ``(a, b, g)`` reaches the atoms ``h g H_b`` (``h`` in ``H_a``) from ``a`` and
    ``h g^-1 H_a`` (``h`` in ``H_b``) from ``b``; a coset is keyed by its least element, so
    repeated or overlapping rows count each atom once.

    :param tables: The standard setting's group tables.
    :param stabilisers: Per site, the elements fixing its reference atom (translations included).
    :param rows: The bonds ``(site, partner, element)`` as standard-setting elements.
    :return: The bond count per site.
    """
    from httk.atomistic.symmetry.protonet import _key

    reached: list[set[tuple[int, tuple[int, ...]]]] = [set() for _ in stabilisers]
    for a, b, g in rows:
        for site, x, other in ((a, g, b), (b, tables.inv(g), a)):
            for h in stabilisers[site]:
                y = tables.mul(h, x)
                reached[site].add((other, min(_key(tables.mul(y, z)) for z in stabilisers[other])))
    return tuple(len(atoms) for atoms in reached)


def _stabilisers(spacegroup: "Spacegroup", sites: Sequence["WyckoffSite"]) -> list[list["_Element"]]:
    """Return the standard elements fixing each stored reference atom, refusing a site at a special point."""
    from httk.atomistic.models.structure.asu import _group, _reference, _standard_elements

    elements, order = _standard_elements(spacegroup), len(_group(spacegroup)[0])
    stabilisers = []
    for number, site in enumerate(sites):
        fixed = elements.stabilizer(_reference(spacegroup, site))
        if len(fixed) * spacegroup.wyckoff_position(site.wyckoff).multiplicity != order:
            raise ValueError(
                f"site {number} lies at a special point of its Wyckoff letter {site.wyckoff!r}; its bond count "
                "and shape are undefined; give it the special letter"
            )
        stabilisers.append(fixed)
    return stabilisers


def _structure_bond_counts(
    spacegroup: "Spacegroup", sites: Sequence["WyckoffSite"], bonds: Sequence["SiteBond"]
) -> tuple[int, ...]:
    """Return the bond count of every site of a network, in its stored setting.

    :param spacegroup: The stored setting.
    :param sites: The Wyckoff sites.
    :param bonds: The bond network.
    :return: The bond count per site.
    :raises ValueError: If a site lies at a special point of its own Wyckoff letter.
    """
    from httk.atomistic.models.structure.asu import _op, _standard_elements

    elements = _standard_elements(spacegroup)
    rows = [(bond.site, bond.partner, elements.element(_op(bond.operation))) for bond in bonds]
    return _bond_counts(elements.tables, _stabilisers(spacegroup, sites), rows)


def _exact_metric(structure: "StructureAPI") -> list[list[Any]]:
    """Return the exact metric of a structure's own reduced coordinates: ``Fraction`` entries when rational, else surd scalars."""
    metric = structure.cell.metric()
    if metric.is_rational:
        return [list(row) for row in metric.coefficient(1).to_fractions()]
    return [[metric._element((r, c)) for c in range(3)] for r in range(3)]


def _averaged_metric(structure: "FundamentalDomainStructure", tables: "_GroupTables") -> list[list[Any]]:
    """Return the exact metric of the standard-setting coordinates, averaged over the point group.

    Entries are ``Fraction`` for a rational metric, else surd scalars.
    """
    A = structure.transform_from_standard.operation.matrix.to_fractions()  # own = A standard + a
    G = _exact_metric(structure)
    standard = [
        [sum((A[r][a] * G[r][c] * A[c][b] for r in range(3) for c in range(3)), Fraction(0)) for b in range(3)]
        for a in range(3)
    ]
    return [
        [
            sum((W[r][a] * standard[r][c] * W[c][b] for W in tables.W for r in range(3) for c in range(3)), Fraction(0))
            / tables.nops
            for b in range(3)
        ]
        for a in range(3)
    ]


def _angle(metric: Sequence[Sequence[Any]], u: Sequence[Fraction], v: Sequence[Fraction]) -> float:
    """Return the angle in degrees between two exact vectors, from the exact squared cosine."""

    def form(x: Sequence[Fraction], y: Sequence[Fraction]) -> Any:
        return sum((x[a] * metric[a][b] * y[b] for a in range(3) for b in range(3)), Fraction(0))

    p = form(u, v)
    square = p * p / (form(u, u) * form(v, v))
    if not isinstance(square, Fraction) and square.is_rational:
        square = square._rational_fraction()
    sign = (p > 0) - (p < 0) if isinstance(p, Fraction) else p.sign()
    cosine = math.copysign(math.sqrt(float(square) if isinstance(square, Fraction) else square.to_float()), sign)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _baseline_shapes(
    structure: "FundamentalDomainStructure", supplied: Sequence[str | None] | None = None
) -> tuple[str | None, ...]:
    """Return the baseline shape of every site of a structure's network.

    The detector reads the sorted angles between the bonds of each site's reference atom. They
    come from the exact metric of the standard-setting coordinates averaged over the point group,
    and each exact squared cosine is converted to a float once, so every exact re-description of
    one crystal (cell, origin, setting, mirror image) gives bit-identical angles and so the same
    shapes. Atoms are ``(site, point)`` pairs: two sites at one point are two atoms.

    :param structure: A structure with a bond network.
    :param supplied: Optional symbols per site; a site with one keeps it and is not measured.
    :return: Per site, its shape, or ``None`` for fewer than two bonds.
    :raises ValueError: If the structure has no network, a site lies at a special point of its
        Wyckoff letter, or a measured site has a zero-length bond.
    """
    from httk.atomistic.models.structure.asu import _apply, _op, _reference, _standard_elements

    if structure.bonds is None:
        raise ValueError("a structure without a bond network has no shapes")
    bonds = structure.bonds.pairs
    spacegroup, sites = structure.spacegroup, structure.wyckoff_sites
    elements = _standard_elements(spacegroup)
    tables = elements.tables
    stabilisers = _stabilisers(spacegroup, sites)
    rows = [(bond.site, bond.partner, elements.element(_op(bond.operation))) for bond in bonds]
    counts = _bond_counts(tables, stabilisers, rows)
    refs = [(p[0], p[1], p[2]) for p in (_apply(elements.to_standard, _reference(spacegroup, site)) for site in sites)]
    metric: list[list[Any]] | None = None
    result: list[str | None] = []
    for s, count in enumerate(counts):
        if supplied is not None and supplied[s] is not None:
            result.append(supplied[s])
            continue
        if count < 2:
            result.append(None)
            continue
        atoms = {
            (other, tables.apply(tables.mul(h, x), refs[other]))
            for a, b, g in rows
            for site, x, other in ((a, g, b), (b, tables.inv(g), a))
            if site == s
            for h in stabilisers[s]
        }
        assert len(atoms) == count, "the bond count and the neighbour atoms use the same groups"
        vectors = [tuple(p - q for p, q in zip(point, refs[s], strict=True)) for _other, point in atoms]
        if any(not any(d) for d in vectors):
            raise ValueError(f"site {s}: a zero-length bond has no direction; supply its shape")
        if metric is None:
            metric = _averaged_metric(structure, tables)
        angles = sorted(_angle(metric, u, v) for u, v in itertools.combinations(vectors, 2))
        result.append(_detect_angles(count, angles)[0])
    return tuple(result)


def _cell_neighbours(nsites: int, pairs: Iterable["SiteBond"]) -> list[set[tuple[int, tuple[Fraction, ...]]]]:
    """Return, per site of a unit cell, its distinct bonded atoms ``(partner, translation)``.

    A pair ``(i, j, n)`` adds ``(j, n)`` to ``i`` and ``(i, -n)`` to ``j``, so a bond of a site to a
    translate of itself adds two atoms.
    """
    atoms: list[set[tuple[int, tuple[Fraction, ...]]]] = [set() for _ in range(nsites)]
    for bond in pairs:
        n = tuple(Fraction(x) for x in bond.operation.vector.to_fractions())
        atoms[bond.site].add((bond.partner, n))
        atoms[bond.partner].add((bond.site, tuple(-x for x in n)))
    return atoms


def _cell_bond_counts(nsites: int, pairs: Iterable["SiteBond"]) -> tuple[int, ...]:
    """Return the bond count of every site of a unit-cell network (lattice-translation bonds).

    :param nsites: The number of sites.
    :param pairs: The bonds, each operation a lattice translation.
    :return: The number of distinct atoms bonded to each site.
    """
    return tuple(len(atoms) for atoms in _cell_neighbours(nsites, pairs))


def _cell_baseline_shapes(
    structure: "StructureAPI", supplied: Sequence[str | None] | None = None
) -> tuple[str | None, ...]:
    """Return the baseline shape of every site of a unit-cell structure's network.

    As :func:`_baseline_shapes`, without symmetry: the angles come from the cell's exact metric,
    not averaged. Every exact re-description (cell, origin, mirror image, site order) still gives
    bit-identical angles, since a unit cell lists the atoms of an orbit separately.

    :param structure: A structure presented as a unit cell (its network indexes ``sites``) with a
        bond network.
    :param supplied: Optional symbols per site; a site with one keeps it and is not measured.
    :return: Per site, its shape, or ``None`` for fewer than two bonds.
    :raises ValueError: If the structure has no network, or a measured site has a zero-length bond.
    """
    network = structure.bonds
    if network is None:
        raise ValueError("a structure without a bond network has no shapes")
    points = [[Fraction(x) for x in row] for row in structure.sites.reduced_coords.to_fractions()]
    metric: list[list[Any]] | None = None
    result: list[str | None] = []
    for s, atoms in enumerate(_cell_neighbours(len(points), network.pairs)):
        if supplied is not None and supplied[s] is not None:
            result.append(supplied[s])
            continue
        if len(atoms) < 2:
            result.append(None)
            continue
        vectors = [tuple(p + t - q for p, t, q in zip(points[other], n, points[s], strict=True)) for other, n in atoms]
        if any(not any(d) for d in vectors):
            raise ValueError(f"site {s}: a zero-length bond has no direction; supply its shape")
        if metric is None:
            metric = _exact_metric(structure)
        angles = sorted(_angle(metric, u, v) for u, v in itertools.combinations(vectors, 2))
        result.append(_detect_angles(len(atoms), angles)[0])
    return tuple(result)


def _site_shapes(structure: "StructureAPI") -> tuple[str | None, ...]:
    """Return every site's shape: the assigned symbol where the network has one, else the baseline.

    :param structure: A fundamental domain (its sites are its Wyckoff sites) or a unit-cell
        structure (its sites are ``sites``) with a bond network.
    :return: Per site, its shape, or ``None`` for fewer than two bonds.
    :raises ValueError: As :func:`_baseline_shapes` and :func:`_cell_baseline_shapes`.
    """
    from httk.atomistic.models.structure.asu import FundamentalDomainStructure

    network = structure.bonds
    if network is None:
        raise ValueError("a structure without a bond network has no shapes")
    if isinstance(structure, FundamentalDomainStructure):
        return _baseline_shapes(structure, network.shapes)
    return _cell_baseline_shapes(structure, network.shapes)
