"""Torsion symbols of bonds in molecular pieces (protonet follow-ups 10 and 11)."""

import math
from fractions import Fraction as F
from functools import cache
from pathlib import Path
from typing import Any

import httk.core
import pytest
from httk.core import SurdVector
from test_structure_shapes import _redescribed

from httk.atomistic import AffineOperation, ASUStructure, SiteBond, Species, WyckoffSite, distance_bonds
from httk.atomistic.models.protonetstructure.protonetstructure import _protonet_asu
from httk.atomistic.models.structure.asu import _apply, _op, _reference, _standard_elements
from httk.atomistic.symmetry.shapes import _averaged_metric, _site_shapes
from httk.atomistic.symmetry.torsions import _bond_torsions, _edge, _site_pair_torsions

FIXTURES = Path(__file__).parent / "fixtures" / "cod"
ONE = AffineOperation.identity()
NEGATE = AffineOperation(((-1, 0, 0), (0, -1, 0), (0, 0, -1)))
C = Species("C", ("C",), (1,))


# ---------------------------------------------------------------------------------------------
# the fixtures
# ---------------------------------------------------------------------------------------------
@cache
def _fixture(name: str) -> Any:
    """A COD fixture with default distance bonds, as ``Protonetstructure.from_asu`` reads it."""
    return _protonet_asu(distance_bonds(httk.core.load(str(FIXTURES / f"{name}.cif"))))


def _symbols(structure: Any) -> list[tuple[tuple[str, ...], tuple[str, ...], float, str, float]]:
    """Per bond orbit with a symbol: elements, shapes, length in Angstrom, symbol, |torsion|."""
    shapes = _site_shapes(structure)
    elements = _standard_elements(structure.spacegroup)
    metric = _averaged_metric(structure, elements.tables)
    species = {value.name: value for value in structure.species}
    refs = [_apply(elements.to_standard, _reference(structure.spacegroup, site)) for site in structure.wyckoff_sites]
    result = []
    for bond, (letter, tau, _reason) in zip(structure.bonds.pairs, _bond_torsions(structure, shapes), strict=True):
        if letter is None:
            continue
        end = elements.tables.apply(elements.element(_op(bond.operation)), refs[bond.partner])
        v = [q - p for p, q in zip(refs[bond.site], end, strict=True)]
        length = math.sqrt(float(sum(v[i] * metric[i][j] * v[j] for i in range(3) for j in range(3))))
        pair = (bond.site, bond.partner)
        assert tau is not None
        result.append(
            (
                tuple(sorted(species[structure.wyckoff_sites[s].species].chemical_symbols[0] for s in pair)),
                tuple(sorted(str(shapes[s]) for s in pair)),
                length,
                letter,
                tau,
            )
        )
    return sorted(result, key=lambda row: (row[0], row[1], row[3], row[4]))


# (elements, shapes, length, symbol, angle) per bond orbit with a symbol, under the final edges of Part A
CHECK = {
    # the 1.424 C-N bond (39 degrees under follow-up 10): its C end is planar (TP3) and its two ring
    # neighbours tie by count
    "4131686": [(("C", "N"), ("TP3", "TP3"), 1.353, "t", 174)],
    # the brief lists one orbit per (elements, shapes, length): two orbits share 1.524 (168 and 177 degrees)
    "4134228": [
        (("C", "C"), ("T4", "T4"), 1.524, "t", 168),
        (("C", "C"), ("T4", "T4"), 1.523, "t", 175),
        (("C", "C"), ("T4", "T4"), 1.524, "t", 177),
        (("C", "C"), ("T4", "T4"), 1.521, "t", 179),
    ],
    "4326344": [],  # the C-N bond: the N end's two methyl carbons tie
    "7217003": [],  # the C-C bond: a phenyl ring's two ortho carbons tie
    "7217521": [],
    # the 1.365 C-O bond (176 degrees under follow-up 10, by a refinement at S1): its C end is planar
    # (TP3) and its two ring neighbours tie by count; the 1.405 bond: a phenyl ring's two ortho
    # carbons tie
    "7217843": [],
    "7218683": [],
    "7219077": [],
    "7219102": [(("C", "N"), ("T4", "T4"), 1.480, "t", 166)],
    "7219855": [
        (("C", "C"), ("T4", "T4"), 1.521, "t", 175),
        (("C", "C"), ("T4", "T4"), 1.513, "t", 180),
        (("C", "N"), ("A2", "TP3"), 1.278, "t", 180),
        (("N", "N"), ("A2", "A2"), 1.414, "t", 180),
    ],
}


@pytest.mark.parametrize("name", sorted(CHECK))
def test_the_fixture_symbols(name: str) -> None:
    got = _symbols(_fixture(name))
    assert len(got) == len(CHECK[name])
    for (elements, shapes, length, letter, tau), expected in zip(got, CHECK[name], strict=True):
        assert (elements, shapes, letter) == expected[:2] + expected[3:4]
        assert abs(length - expected[2]) < 0.001 and abs(tau - expected[4]) < 0.5


def test_site_pair_letters_key_the_qualifying_orbits() -> None:
    structure = _fixture("7219855")
    letters = _site_pair_torsions(structure, _site_shapes(structure))
    assert sorted(letters.values()) == ["t", "t", "t", "t"]
    assert all(i <= j for i, j in letters)


@pytest.mark.parametrize("name", ["4326344", "7219102", "7219855"])  # P2_1, P2_12_12_1 (Sohncke), P2_1/c
def test_the_mirror_image_has_the_same_symbols(name: str) -> None:
    source = _fixture(name)
    mirror = _redescribed(source, NEGATE, basis=source.cell.basis, seed=3)

    def symbols(structure: Any) -> list[tuple[str | None, float | None]]:
        return sorted((x[0], x[1]) for x in _bond_torsions(structure, _site_shapes(structure)) if x[0])

    assert symbols(mirror) == symbols(_protonet_asu(mirror)) == symbols(source)  # bit-identical angles


# ---------------------------------------------------------------------------------------------
# synthetic molecules
# ---------------------------------------------------------------------------------------------
ORTHOGONAL = [[4, 0, 0], [0, 5, 0], [0, 0, 6]]
ROOT3 = SurdVector.sqrt_of(3)
HEXAGONAL = SurdVector([[4, 0, 0], [-2, 0, 0], [0, 0, 5]]) + SurdVector([[0, 0, 0], [0, 2, 0], [0, 0, 0]]) * ROOT3
Q = F(1, 4)
# X-A-B-C-D-Y in P1 with b0 = B - A = a/4 and b1 = C - B = c/4; b2 = D - C is b/4 (ORTHOGONAL, 90
# degrees) or (a + b)/4 (HEXAGONAL, cos = -1/2 exactly: 120 degrees)
CHAIN = [(Q, 2 * Q, Q), (Q, 3 * Q, Q), (2 * Q, 2 * Q, Q), (2 * Q, 2 * Q, 2 * Q)]  # A X B C
X, A, B, C_, D, Y = 0, 1, 2, 3, 4, 5
SHAPES = (None, "A2", "T4", "T4", "A2", None)
SPECIES = [f"C{i}" for i in range(6)]


def _chain(
    basis: Any = ORTHOGONAL,
    extra: tuple[tuple[F, F, F], ...] = (),
    bonds: tuple[tuple[int, int, tuple[int, int, int]], ...] = (),
    species: dict[int, Species] | None = None,
    *,
    without_x: bool = False,
    a: tuple[F, F, F] | None = None,
) -> ASUStructure:
    """The chain X-A-B-C-D-Y (sites 0-5) plus ``extra`` sites and ``bonds``; site ``i`` is carbon ``Ci`` unless given."""
    a, x, b, c = CHAIN if a is None else (a, (a[0], a[1] + Q, a[2]), *CHAIN[2:])
    d = (c[0] + Q, c[1] + Q, c[2]) if basis is HEXAGONAL else (c[0], c[1] + Q, c[2])
    points = [x, a, b, c, d, (d[0], d[1], d[2] + Q), *extra]
    rows = [(X, A, (0, 0, 0)), (A, B, (0, 0, 0)), (B, C_, (0, 0, 0)), (C_, D, (0, 0, 0)), (D, Y, (0, 0, 0)), *bonds]
    if without_x:
        rows = rows[1:]
    named = {i: Species(f"C{i}", ("C",), (1,)) for i in range(len(points))} | (species or {})
    sites = [WyckoffSite("a", point, named[i].name) for i, point in enumerate(points)]
    pairs = [SiteBond(i, j, AffineOperation(ONE.matrix, t)) for i, j, t in rows]
    return ASUStructure(basis, 1, sites, list(dict.fromkeys(named.values())), bonds=pairs)


def _results(structure: ASUStructure, shapes: Any = SHAPES) -> dict[tuple[int, int], tuple[Any, ...]]:
    """Per bonded site pair of a P1 structure, its ``(letter, |torsion|, reason)``."""
    full = tuple(shapes) + (None,) * (len(structure.wyckoff_sites) - len(shapes))
    assert structure.bonds is not None
    return {
        (min(bond.site, bond.partner), max(bond.site, bond.partner)): value
        for bond, value in zip(structure.bonds.pairs, _bond_torsions(structure, full), strict=True)
    }


def test_the_chain_has_one_symbol_and_why_the_others_have_none() -> None:
    results = _results(_chain())
    assert results.pop((B, C_))[:2] == ("g", 90.0)
    assert results == {
        (X, A): (None, None, "terminal atom"),
        (A, B): (None, None, "end group"),
        (C_, D): (None, None, "end group"),
        (D, Y): (None, None, "terminal atom"),
    }
    assert _site_pair_torsions(_chain(), SHAPES) == {(B, C_): "g"}


H1, H2, H3 = (2 * Q, F(3, 8), Q), (F(5, 8), 2 * Q, F(1, 8)), (2 * Q, 2 * Q, F(1, 8))


@pytest.mark.parametrize(
    ("structure", "shapes", "reason"),
    [
        (_chain(bonds=((Y, X, (1, 0, 0)),)), SHAPES, "periodic piece"),  # an infinite chain
        (_chain(extra=(H1, H2, H3), bonds=tuple((B, h, (0, 0, 0)) for h in (6, 7, 8))), SHAPES, "more than four bonds"),
        (_chain(bonds=((X, Y, (0, 0, 0)),)), SHAPES, "ring bond"),  # a six-membered ring
        (_chain(without_x=True), SHAPES, "end group"),  # B-C only turns A
        (_chain(), (None, "A2", "TP3", "T4", "A2", None), "shape pair without a letter"),
        (_chain(), (None, "A2", "L2", "T4", "A2", None), "shape pair without a letter"),
        # A-B-C 2.94 degrees from straight (tan = 4 / 78): skipped at 3 degrees, not by the reference's 0.05 (2.87)
        (_chain(a=(F(51, 104), 2 * Q, F(1, 8))), SHAPES, "collinear"),
    ],
)
def test_each_rule_alone_removes_the_symbol(structure: ASUStructure, shapes: Any, reason: str) -> None:
    assert _results(structure, shapes)[(B, C_)] == (None, None, reason)
    assert (B, C_) not in _site_pair_torsions(structure, tuple(shapes) + (None,) * 3)


@pytest.mark.parametrize(
    "species",
    [
        {A: Species("C0.5", ("C",), (F(1, 2),))},
        {B: Species("HF", ("H", "F"), (F(1, 2), F(1, 2)))},
        {C_: Species("Cv", ("C", "vacancy"), (F(3, 4), F(1, 4)))},
        {D: Species("CN", ("C", "N"), (F(1, 2), F(1, 2)))},
    ],
)
def test_a_partial_or_mixed_site_keeps_the_letter(species: dict[int, Species]) -> None:
    # the letter never depends on the species: a partial, mixed or vacant B, C, A or D changes nothing
    assert _results(_chain(species=species))[(B, C_)] == _results(_chain())[(B, C_)]
    assert _site_pair_torsions(_chain(species=species), SHAPES) == {(B, C_): "g"}


# P, P2, P3, P4: a second branch on B, P where the torsion P-B-C-D is 180 degrees (A-B-C-D is 90)
P, P2, P3, P4 = 6, 7, 8, 9
BRANCH = (H1, (F(5, 8), F(3, 8), Q), (F(5, 8), F(3, 8), F(1, 8)), (F(5, 8), Q, F(1, 8)))
BRANCHED = (None, "A2", "TPY3", "T4", "A2", None)  # B has three bonds: refined, as a T4 end is


def _branched(depth: int, *, longer_x: bool = False) -> ASUStructure:
    """The chain with the branch B-P-P2-... of ``depth`` atoms; ``longer_x`` bonds one more atom to X."""
    bonds = ((B, P, (0, 0, 0)), *((P + k, P + k + 1, (0, 0, 0)) for k in range(depth - 1)))
    extra = BRANCH[:depth]
    if longer_x:
        extra, bonds = (*extra, (Q, 3 * Q, 2 * Q)), (*bonds, (X, P + depth, (0, 0, 0)))
    return _chain(extra=extra, bonds=bonds)


def test_a_unique_most_bonded_neighbour_is_the_reference() -> None:
    # P has one bond, A two: A wins by count
    assert _results(_branched(1), BRANCHED)[(B, C_)][:2] == ("g", 90.0)


def test_a_tie_by_count_is_broken_at_the_first_shell() -> None:
    # A and P have two bonds; S1 is (1,) for A (X) and (2,) for P (P2): P wins
    assert _results(_branched(3), BRANCHED)[(B, C_)][:2] == ("t", 180.0)


def test_a_tie_by_count_and_first_shell_is_broken_at_the_second_shell() -> None:
    # S1 is (2,) for both (X, P2); S2 is (1,) for A (the atom after X) and (2,) for P (P3): P wins
    assert _results(_branched(4, longer_x=True), BRANCHED)[(B, C_)][:2] == ("t", 180.0)


def test_two_equal_branches_tie_and_give_no_letter() -> None:
    # an isopropyl-like end: A-X and P-P2 have the same signature ((1,), (), ())
    assert _results(_branched(2), BRANCHED)[(B, C_)] == (None, None, "tied reference neighbour")


@pytest.mark.parametrize(
    ("shapes", "letter"),
    [(("TP3", "TP3"), None), (("TP3", "A2"), None), (("TPY3", "T4"), "t"), (("TPY3", "TPY3"), "t")],
)
def test_a_tie_by_count_is_final_at_a_planar_end_outside_a_ring(shapes: tuple[str, str], letter: str | None) -> None:
    # A and P tie by count at B and P wins at S1 (as in the first-shell test): a substituted double
    # bond (TP3 at B) keeps the tie, a TPY3 end is refined
    result = _results(_branched(3), (None, "A2", *shapes, "A2", None))[(B, C_)]
    if letter is None:
        assert result == (None, None, "tied reference neighbour")
    else:
        assert result[:2] == (letter, 180.0)


# the ring B-A-X-R-P-B, with Z on X only: A and P tie by count, and A wins at S1 ((3,) against (2,))
R, Z = 7, 8
RING = _chain(
    extra=(H1, (F(3, 8), F(5, 8), F(1, 8)), (F(1, 8), 3 * Q, Q)),
    bonds=((B, P, (0, 0, 0)), (P, R, (0, 0, 0)), (R, X, (0, 0, 0)), (X, Z, (0, 0, 0))),
)


def test_a_planar_ring_end_tied_by_count_gives_no_letter() -> None:
    # follow-up 10 gave t here (A-B-C-D is 90 degrees, on the planar edge)
    assert _results(RING, (None, "A2", "TP3", "A2", "A2", None))[(B, C_)] == (None, None, "tied reference neighbour")


def test_a_tetrahedral_ring_atom_keeps_its_refined_letter() -> None:
    assert _results(RING, BRANCHED)[(B, C_)][:2] == ("g", 90.0)


@pytest.mark.parametrize(
    ("shapes", "letter"),
    [
        (("T4", "T4"), "g"),
        (("T4", "TPY3"), "g"),
        (("TPY3", "TPY3"), "g"),
        (("A2", "T4"), None),  # the COD measurement found no clean edge
        (("T4", "A2"), None),
        (("A2", "A2"), "g"),
        (("TP3", "TP3"), "t"),  # 90 degrees: below the tetrahedral edges, on the planar ones
        (("A2", "TP3"), "t"),
        (("TP3", "A2"), "t"),
        (("A2", "TPY3"), None),  # dropped from the table
        (("TPY3", "A2"), None),
        (("T4", "TP3"), None),
        (("TPY3", "TP3"), None),
        (("A2", "L2"), None),
    ],
)
def test_the_table_is_keyed_by_the_sorted_shape_pair(shapes: tuple[str, str], letter: str | None) -> None:
    result = _results(_chain(), (None, "A2", *shapes, "A2", None))[(B, C_)]
    assert result[0] == letter
    assert letter or result == (None, None, "shape pair without a letter")
    assert _edge(*shapes) == _edge(*reversed(shapes))


def _descriptions(source: ASUStructure) -> list[Any]:
    """The source, a cell change with an origin shift, and the mirror image."""
    cell = AffineOperation(((1, 1, 0), (0, 1, 0), (0, 0, 1)), (F(1, 3), 0, F(1, 5)))
    return [source, _redescribed(source, cell, seed=1), _redescribed(source, NEGATE, basis=source.cell.basis, seed=2)]


@pytest.mark.parametrize(
    ("basis", "shapes", "angle"),
    [(HEXAGONAL, SHAPES, 120.0), (ORTHOGONAL, (None, "A2", "TP3", "TP3", "A2", None), 90.0)],
)
def test_a_torsion_exactly_on_the_edge_takes_the_upper_class_in_every_description(
    basis: Any, shapes: Any, angle: float
) -> None:
    source = _chain(basis)
    seen = 0
    for described in _descriptions(source):
        # the sites are shuffled: carry each site's shape by its species
        moved = [dict(zip(SPECIES, shapes, strict=True))[site.species] for site in described.wyckoff_sites]
        values = [value for value in _bond_torsions(described, moved) if value[0]]
        assert [(letter, round(tau, 6)) for letter, tau, _ in values] == [("t", angle)]
        seen += 1
    assert seen == 3


def test_a_bridged_dimer_has_two_orbits_between_one_site_pair_and_no_letter() -> None:
    # P-1: B-C and B-(inverted C) are two orbits of the pair (B, C), closing the ring B-C-B'-C'
    inversion = AffineOperation(((-1, 0, 0), (0, -1, 0), (0, 0, -1)))
    points = {
        "X": (F(25, 100), F(15, 100), F(3, 100)),
        "A": (F(2, 10), F(5, 100), F(3, 100)),
        "B": (F(8, 100), F(2, 100), F(1, 100)),
        "C": (F(-2, 100), F(8, 100), F(-1, 100)),
        "D": (F(-5, 100), F(2, 10), F(-2, 100)),
        "Y": (F(-15, 100), F(25, 100), 0),
    }
    index = {name: i for i, name in enumerate(points)}
    sites = [WyckoffSite("i", point, "C") for point in points.values()]
    pairs = [SiteBond(index[i], index[j], ONE) for i, j in ("XA", "AB", "BC", "CD", "DY")]
    pairs.append(SiteBond(index["B"], index["C"], inversion))
    structure = ASUStructure([[10, 0, 0], [0, 10, 0], [0, 0, 10]], 2, sites, [C], bonds=pairs)
    shapes = (None, "A2", "TPY3", "TPY3", "A2", None)
    assert structure.bonds is not None
    between = [
        value
        for bond, value in zip(structure.bonds.pairs, _bond_torsions(structure, shapes), strict=True)
        if {bond.site, bond.partner} == {index["B"], index["C"]}
    ]
    assert between == [(None, None, "ring bond")] * 2
    assert _site_pair_torsions(structure, shapes) == {}


def test_a_structure_without_a_network_has_no_torsions() -> None:
    plain = ASUStructure(ORTHOGONAL, 1, [WyckoffSite("a", (0, 0, 0), "C")], [C])
    assert _bond_torsions(plain, (None,)) == [] and _site_pair_torsions(plain, (None,)) == {}
