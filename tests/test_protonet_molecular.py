"""Molecular crystals through ``Protonetstructure.from_asu``: pinned labels and description independence.

Every fixture is an ordered, H-complete COD entry in its standard setting whose covalent net is
molecular (finite components only). The bonds come from the element-pair cutoff table below, never
from one global cutoff, and every description checks that each cutoff lies well inside the gap
between the longest bond and the shortest non-bonded contact of its pair.
"""

import random
from fractions import Fraction as F
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any

import httk.core
import pytest
from _protonet_support import random_shapes, raw_rows, reference_point
from httk.core import FracVector
from test_protonet_cells import _p1_gram

from httk.atomistic import (
    ASUStructure,
    BondNetwork,
    Cell,
    ProtonetstructureView,
    Protonettype,
    ProtonettypeView,
    SiteBond,
    Species,
    UnitcellStructureView,
    WyckoffSite,
    canonical_asu_protostructure,
)
from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
from httk.atomistic.models.prototype.notation import parse_protonetstructure_label, parse_protonettype_label
from httk.atomistic.symmetry import _protonet_cells as cells
from httk.atomistic.symmetry._wyckoff_actions import compile_wyckoff_action
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.canonical_classification import _signature, canonical_protonetstructure
from httk.atomistic.symmetry.canonical_protostructure import _sohncke
from httk.atomistic.symmetry.limits import _budget_scope
from httk.atomistic.symmetry.protonet import _tables
from httk.atomistic.symmetry.spacegroup import Spacegroup

FIXTURES = Path(__file__).parent / "fixtures" / "cod"
UNIT = ((1, 0), (0, 1))
GUARD = 30.0  # cooperative seconds per from_asu or canonicalization; the default budget is 120 s
MARGIN = 0.25  # Å: no contact of a listed pair lies within this of its cutoff

type Matrix = tuple[tuple[int, int], tuple[int, int]]
type Bond = tuple[int, int, int, int, int, int]

# Å, per element pair; unlisted pairs (H-H, O-O, Cu-Br, ...) are never bonded
COVALENT = {
    frozenset(pair): cutoff
    for pair, cutoff in {
        ("C", "C"): 1.85,
        ("C", "H"): 1.3,
        ("C", "N"): 1.75,
        ("C", "O"): 1.75,
        ("C", "F"): 1.7,
        ("C", "S"): 2.1,
        ("C", "Br"): 2.3,
        ("H", "N"): 1.3,
        ("H", "O"): 1.3,
        ("N", "N"): 1.75,
        ("N", "O"): 1.75,
        ("N", "S"): 2.0,
        ("O", "S"): 1.8,
        # 7218683: square-planar CuN2O2 (Cu-N 2.008, Cu-O 1.896 Å; the nearest Cu-Br is 3.48 Å)
        ("Cu", "N"): 2.4,
        ("Cu", "O"): 2.4,
    }.items()
}

# COD entry -> (rows, pinned canonical label). ``rows`` counts the bond orbits from the chemistry:
# E for a molecule of V atoms and E = V - 1 + rings bonds on a general position, (E + c) / 2 for one
# on an inversion centre or 2-fold axis, c bonds being mapped onto themselves
MOLECULAR = {
    # P2_1/c, Z' = 1/2: a tricyclic tetrazine C8H16N4O2 on an inversion centre (V 30, E 32, c 0);
    # 35 s before the search fix
    "7217521": (
        16,
        "A8B4C2D_mP60_14_8e_4e_2e_e__9~10.14.1.2_10~15.3.4_11~12.15.5.6_12~13.14.7_13TPY3~14@2.8_14TPY3:H-C-N-O",
    ),
    # P2_1/c, Z' = 1: a fluorophenyl naphthalimide C20H14FNO3 (V 39, 4 rings, E 42)
    "7217843": (
        42,
        "A20B14C3DE_mP156_14_20e_14e_3e_e_e__19~20.30.31.32_20~38.33.34_1~2.3.4_2~5.6_3~7.8_4~9.10_5~11.21_6~12.35_7~11.22_8~38.36_9~12.23_10~38.37_11~24_12~25_13~14.15.26_14~16.27_15~17.35_16~18.39_17~18.28_18~29:C-H-O-N-F",
    ),
    # P2_1/c: Cu(C9H9BrNO)2, Cu on 2b with the complex on that centre; two phenyl and two chelate
    # rings (V 43, E 46, c 0)
    "7218683": (
        23,
        "A18B18C2D2E2F_mP86_14_9e_9e_e_e_e_a__8~9.20.15.16_9~17.18.21_22SP4~20.19_1~2.3.4_2~5.10_3~6.19_4~20.11_5~7.12_6~7.13_7~14:C-H-O-N-Br-Cu",
    ),
    # P2_1/c: two molecules on two inversion centres, C12H10N4 (V 26, E 27, the N-N bond through
    # its centre, c 1) and adipic acid C6H10O4 (V 20, E 19, the middle C-C bond, c 1)
    "7219855": (
        24,
        "A10B9C2D2_mP92_14_10e_9e_2e_2e__11~11@2t.12t.1.2_12~18.3.4_13~14.19.5_14~22.6_15~16.19.7_16~22.8_17~19.23t.9_18~20.21_20~10_23~23@2@455t:H-C-O-N",
    ),
    # P2_1/c, Z' = 1: a flexible C9H17N3O with one ring (V 30, E 30); 140 s to over 450 s
    # before the search fix
    "4134228": (
        30,
        "A17B9C3D_mP120_14_17e_9e_3e_e__18~19.1.2.3_19~20t.4.5_20~21t.6.7_21~22t.8.9_22~23t.10.11_23~28.12.13_24~25.28.14.15_26~27.28.29_27~16.17_25~29.30:H-C-N-O",
    ),
    # C2/c, Z' = 1/2: C18H18N2S2 on a 2-fold axis with two atoms on it (V 40, 3 rings, E 42, c 2:
    # the axial C-C bond and the ring CH2-CH2 bond across the axis)
    "7219077": (
        22,
        "A9B9CD_mC160_15_2e8f_9f_f_f__3~3@3.20.11.12_7~4.20.15.16_1~2.20_2~21_5~8.4.13_6~9.4.14_8~10.17_9~10.18_10~19:C-H-N-S",
    ),
    # P2_1: 2-[(N,N-dimethylamino)methyl]aniline C9H14N2 (V 25, 1 ring, E 25); 134 s to over
    # 400 s before the search fix
    "4326344": (
        25,
        "A14B9C2_mP50_4_14a_9a_2a__15~25.1.2.3_16~25.4.5.6_17~22.25.7.8_18~19.20.9_19~21.10_20~22.11_21~23.12_23~24.22_24~13.14_25TPY3:H-C-N",
    ),
    # P-1, Z = 1: C20H16N2O2 on an inversion centre (V 40, 2 rings, E 41, the middle bond, c 1)
    "4131686": (
        21,
        "A10B8CD_aP40_2_10i_8i_i_i__9~10.15.16.17_3~4.5.2_4~6.11_5~7.12_6~8.13_7~8.19_8~14_19~10t.18_10~20_1L2~1@2.2_2L2:C-H-N-O",
    ),
    # P-1, Z' = 1: a triazolo-benzothiadiazine dioxide C16H14N4O2S (V 37, 4 rings, E 40); 113 s before
    # the search fix
    "7217003": (
        40,
        "A16B14C4D2E_aP74_2_16i_14i_4i_2i_i__12~9.24.25.26_16~33.28.29.30_37~15.33.35.36_1~2.3.4_2~5.17_3~6.18_4~7.31_5~8.19_6~8.20_7~32.33_8~21_10~13.9.22_11~14.9.23_13~15.27_14~32.15_32~34_31~34:C-H-N-O-S",
    ),
    # P2_12_12_1: sarcosine C3H7NO2 (V 13, acyclic, E 12)
    "7219102": (12, "A7B3C2D_oP52_19_7a_3a_2a_a__8~13.1.2.3_9~13t.10.4.5_13~6.7_10~11.12:H-C-O-N"),
}

# fixtures whose input bare description is not the canonical one: from_asu keeps it (Cu on 2b),
# canonical_protonetstructure moves the metal to 2a
KEEPS_BARE = {"7218683"}


@cache
def _source(entry: str) -> tuple[int, list[tuple[str, str, tuple[F, ...]]], list[list[float]]]:
    """``(it, [(element, letter, free parameters)], lattice rows)`` of a fixture as written."""
    cif = httk.core.load(str(FIXTURES / f"{entry}.cif"))
    element = {species.name: species.chemical_symbols[0] for species in cif.species}
    sites = [(element[ws.species], ws.wyckoff, tuple(ws.free_params.to_fractions())) for ws in cif.wyckoff_sites]
    return cif.space_group_it_number, sites, [[float(x) for x in row] for row in cif.lattice_vectors]


def _action(X: Matrix) -> AffineOperation:
    (a, b), (c, d) = X
    return AffineOperation(((a, 0, b), (0, 1, 0), (c, 0, d)))


@cache
def _signature_trivial_cells(it: int) -> list[Matrix]:
    """The monoclinic cell changes with entries up to 2 that keep every letter and operation.

    In a Sohncke group only the proper ones: a change of determinant -1 describes the mirror image there.
    """
    group = Spacegroup.standard(it)
    identity = _signature(group, _action(UNIT))
    dets = (1,) if _sohncke(group) else (1, -1)
    return [
        X
        for X in (((a, b), (c, d)) for a, b, c, d in product(range(-2, 3), repeat=4) if a * d - b * c in dets)
        if X != UNIT and _signature(group, _action(X)) == identity
    ]


@cache
def _described(
    entry: str, seed: str | None = None, *, mirror: bool = False
) -> tuple[ASUStructure, list[Bond], list[Any]]:
    """The fixture as ``(asu, 1-based bonds, 0-based rows)``, re-described when ``seed`` is given.

    With ``mirror`` the coordinates are negated in the same cell first, which describes the mirror
    image (the same crystal in a centrosymmetric group, where the inversion is a symmetry).

    A re-description shuffles the sites, in monoclinic groups applies a random signature-trivial
    cell change, and takes another atom of every general orbit as its reference atom: a random
    operation, then the translation into the unit cube, so that neighbours stay within the
    translations ``raw_rows`` looks at. The bonds are found again from the moved geometry, and no
    contact lies within ``MARGIN`` of its pair's cutoff.
    """
    it, sites, lattice = _source(entry)
    tables = _tables(it)
    if mirror:
        group, negate = Spacegroup.standard(it), AffineOperation(((-1, 0, 0), (0, -1, 0), (0, 0, -1)))
        sites = [
            (element, compiled.target_letter, tuple(compiled.apply(p).to_fractions()))
            for element, letter, p in sites
            for compiled in (compile_wyckoff_action(group, negate, letter),)
        ]
    if seed is not None:
        rng = random.Random(seed)
        sites = list(sites)
        rng.shuffle(sites)
        if 3 <= it <= 15:
            X = rng.choice(_signature_trivial_cells(it))
            group, action = Spacegroup.standard(it), _action(X)
            changed = []
            for element, letter, p in sites:
                compiled = compile_wyckoff_action(group, action, letter)
                assert compiled.target_letter == letter
                changed.append((element, letter, tuple(compiled.apply(p).to_fractions())))
            sites = changed
            (a, b), (c, d) = cells._inverse(X)  # rows of the new lattice: (P^-1)ᵀ times the old rows
            lattice = [
                [a * lattice[0][k] + c * lattice[2][k] for k in range(3)],
                list(lattice[1]),
                [b * lattice[0][k] + d * lattice[2][k] for k in range(3)],
            ]
        moved = []
        for element, letter, p in sites:
            if tables.letter(letter).general:
                k = rng.randrange(tables.nops)
                W, w = tables.W[k], tables.w[k]
                p = tuple((sum(W[r][c] * p[c] for c in range(3)) + w[r]) % 1 for r in range(3))
            moved.append((element, letter, p))
        sites = moved
    refs = [reference_point(it, letter, p) for _, letter, p in sites]
    wide = {pair: cutoff + MARGIN for pair, cutoff in COVALENT.items()}
    rows, found = raw_rows(tables, [(e, letter) for e, letter, _ in sites], refs, lattice, wide)
    for (i, j, _), d in found.items():  # every contact within the widened cutoffs is a clear bond
        assert d < COVALENT[frozenset((sites[i][0], sites[j][0]))] - MARGIN, (entry, sites[i][0], sites[j][0], d)
    asu = ASUStructure(
        Cell(lattice),
        it,
        [WyckoffSite(letter, FracVector(p), element) for element, letter, p in sites],
        [Species(name=e, chemical_symbols=(e,), concentration=(1,)) for e in dict.fromkeys(e for e, _, _ in sites)],
    )
    return asu, [(i + 1, j + 1, g[0] + 1, g[1], g[2], g[3]) for i, j, g in rows], rows


@cache
def _value(entry: str) -> Protonetstructure:
    asu, bonds, _ = _described(entry)
    with _budget_scope(GUARD):
        return Protonetstructure.from_asu(asu, bonds)


@pytest.mark.parametrize("entry", list(MOLECULAR))
def test_molecular_nets_have_their_pinned_labels(entry: str) -> None:
    rows, label = MOLECULAR[entry]
    _, _, raw = _described(entry)
    it, sites, _ = _source(entry)
    # molecular: the bond Gram form of the barycentric P1 placement vanishes
    gram = _p1_gram(_tables(it), [(e, letter) for e, letter, _ in sites], raw)
    assert gram == [[0] * 3 for _ in range(3)]
    value = _value(entry)
    assert len(value.net) == rows
    assert Protonetstructure(value.spacegroup, value.occupations, value.net, value.shapes, value.torsions) == value
    with _budget_scope(GUARD):
        canonical = canonical_protonetstructure(value, timeout=None)
    assert (canonical == value) is (entry not in KEEPS_BARE)
    assert str(canonical.label) == label
    assert parse_protonetstructure_label(label) == canonical
    groups = label.partition("__")[2].rpartition(":")[0].split("_")
    assert sum(len(group.partition("~")[2].split(".")) for group in groups if "~" in group) == rows


@pytest.mark.parametrize(
    "entry,seed",
    [
        *((entry, 0) for entry in MOLECULAR),
        *(pytest.param(entry, k, marks=pytest.mark.extended) for entry in MOLECULAR for k in range(1, 4)),
    ],
)
def test_molecular_values_do_not_depend_on_the_description(entry: str, seed: int) -> None:
    asu, bonds, _ = _described(entry, f"molecular-{entry}-{seed}")
    assert len(bonds) == MOLECULAR[entry][0]  # the same bonds found again
    with _budget_scope(GUARD):
        assert Protonetstructure.from_asu(asu, bonds) == _value(entry)


@pytest.mark.parametrize(
    "entry,seed",
    [
        *((entry, 0) for entry in ("7217521", "4326344", "4131686", "7219102")),
        *(pytest.param(entry, k, marks=pytest.mark.extended) for entry in MOLECULAR for k in range(1, 3)),
    ],
)
def test_supplied_shapes_travel_with_their_sites(entry: str, seed: int) -> None:
    # random valid shapes on the fixture's own description, carried to a shuffled re-description
    asu, bonds, rows = _described(entry)
    it, sites, _ = _source(entry)
    rng = random.Random(f"shapes-{entry}-{seed}")
    shapes = random_shapes(_tables(it), [(e, letter) for e, letter, _ in sites], rows, rng)
    tag = f"molecular-{entry}-{seed}"
    order = list(range(len(sites)))
    random.Random(tag).shuffle(order)  # _described shuffles the sites first with this seed
    moved, moved_bonds, _ = _described(entry, tag)
    with _budget_scope(GUARD):
        value = Protonetstructure.from_asu(asu, bonds, shapes=shapes)
        assert Protonetstructure.from_asu(moved, moved_bonds, shapes=[shapes[old] for old in order]) == value
    assert sorted(map(str, filter(None, value.shapes))) == sorted(map(str, filter(None, shapes)))


# ---------------------------------------------------------------------------------------------
# torsion letters (protonet follow-up 10)
# ---------------------------------------------------------------------------------------------
def _networked(entry: str) -> ASUStructure:
    """The fixture with its bonds as the structure's own network."""
    from httk.atomistic.models.structure.asu import _affine, _standard_elements

    asu, bonds, _ = _described(entry)
    standard = _standard_elements(asu.spacegroup)
    pairs = (SiteBond(i - 1, j - 1, _affine(standard.operation((k - 1, *n)))) for i, j, k, *n in bonds)
    return asu._with_bonds(BondNetwork(tuple(pairs)))


def _letters(value: Any) -> list[str]:
    return sorted(filter(None, value.torsions))


QUICK = {"7219102", "4326344", "4134228"}  # both Sohncke fixtures and the four trans C-C bonds


@pytest.mark.parametrize(
    "entry,form",
    [
        pytest.param(entry, form, marks=[] if entry in QUICK else [pytest.mark.extended])
        for entry in MOLECULAR
        for form in ("mirror", "unit cell", "canonical asu")
    ],
)
def test_molecular_labels_do_not_depend_on_the_mirror_image_unit_cell_or_canonical_asu(entry: str, form: str) -> None:
    expected = _value(entry)
    with _budget_scope(GUARD):
        if form == "mirror":
            asu, bonds, _ = _described(entry, mirror=True)
            assert len(bonds) == MOLECULAR[entry][0]
            value = Protonetstructure.from_asu(asu, bonds)
        elif form == "unit cell":
            value = ProtonetstructureView(UnitcellStructureView(_networked(entry))).unview()
        else:
            value = Protonetstructure.from_asu(canonical_asu_protostructure(_networked(entry)))
        assert _letters(value) == _letters(expected)
        # a Sohncke group's mirror image is the other hand: one label for both hands only without the hand
        keep = not (form == "mirror" and _sohncke(expected.spacegroup))
        label = canonical_protonetstructure(value, preserve_chirality=keep, timeout=None).label
        assert str(label) == str(canonical_protonetstructure(expected, preserve_chirality=keep, timeout=None).label)
    if keep:
        assert str(label) == MOLECULAR[entry][1]


def test_letters_round_trip_through_the_labels_on_targets_with_and_without_operation_and_translation() -> None:
    text = MOLECULAR["7219855"][1]
    targets = [target for group in _net(text).split("_") for target in group.partition("~")[2].split(".")]
    assert {"11@2t", "12t", "23@2@455t"} <= set(targets)
    value = parse_protonetstructure_label(text)
    assert str(value.label) == text and _letters(value) == ["t", "t", "t", "t"]
    anonymous = value.protonettype
    assert anonymous.torsions == value.torsions
    assert parse_protonettype_label(str(anonymous.label)) == anonymous
    assert str(anonymous.label) == text.rpartition(":")[0]


def _net(label: str) -> str:
    return label.partition("__")[2].rpartition(":")[0]


SARCOSINE = MOLECULAR["7219102"][1]  # ..._9~13t.10.4.5_...: site 8 is a CH3 carbon, 1 an H, 13 the N


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("8~13.1.2.3", "8~13.1t.2.3", "give no torsion letter"),  # a hydrogen has no shape
        ("9~13t", "9~13c", "is not one of"),  # T4-T4 takes g or t
        ("8~13.1", "8~13g.1", "end group"),  # the CH3 end: every other neighbour is a hydrogen
        ("9~13t", "9~13tt", "invalid protonet destination"),
        ("9~13t", "9~13x", "invalid protonet destination"),
    ],
)
def test_the_parser_rejects_letters_the_net_and_shapes_do_not_allow(old: str, new: str, message: str) -> None:
    assert parse_protonetstructure_label(SARCOSINE).torsions.count("t") == 1
    text = SARCOSINE.replace(old, new, 1)
    assert text != SARCOSINE
    with pytest.raises(ValueError, match=message):
        parse_protonetstructure_label(text)


BENZAMIDE = MOLECULAR["4131686"][1]  # ..._4~6.11_..._7~8.19_...: C4-C6 a phenyl bond, C8-N19 bonds a phenyl


@pytest.mark.parametrize(
    "text,old,new,message",
    [
        ("A_cF8_227_a__1~1@14@654:C", "654", "654t", "periodic piece"),  # diamond
        (BENZAMIDE, "4~6.11", "4~6c.11", "a ring bond has"),
        # C8 is planar and its two ring neighbours tie by count: no refinement
        (BENZAMIDE, "7~8.19", "7~8.19c", "tied reference neighbour"),
        # the dimethylamino N-C bond: the N end's two methyl carbons tie
        (MOLECULAR["4326344"][1], "17~22.25.7.8", "17~22.25t.7.8", "tied reference neighbour"),
    ],
)
def test_the_parser_rejects_letters_on_framework_ring_and_tied_bonds(
    text: str, old: str, new: str, message: str
) -> None:
    changed = text.replace(old, new, 1)
    assert changed != text
    with pytest.raises(ValueError, match=message):
        parse_protonetstructure_label(changed)


def test_a_site_pair_with_two_orbits_or_two_letters_is_rejected() -> None:
    # P1: two atoms joined by two bond orbits (a chain along a), every atom bent with two bonds
    occupations, net, shapes = [("a", "A"), ("a", "A")], [(1, 2, 1, 0, 0, 0), (1, 2, 1, 1, 0, 0)], ("A2", "A2")
    text = str(Protonettype(1, occupations, net, shapes).label)
    assert text.partition("__")[2] == "1~2.2@1@455"
    for letters in ("t", "t"), ("t", "g"):
        with pytest.raises(ValueError, match="more than one bond orbit"):
            Protonettype(1, occupations, net, shapes, letters)
    for spelled in ("1~2t.2@1@455t", "1~2t.2@1@455g"):
        with pytest.raises(ValueError, match="more than one bond orbit"):
            parse_protonettype_label(text.replace("1~2.2@1@455", spelled))
    # one orbit spelled from both ends with two letters
    value = parse_protonetstructure_label(SARCOSINE)
    row = value.net[value.torsions.index("t")]
    i, j, k, *n = row
    back = Protonetstructure(value.spacegroup, value.occupations, [*value.net, (j, i, k, *n)], value.shapes)
    assert back == Protonetstructure(value.spacegroup, value.occupations, value.net, value.shapes)
    args = (value.spacegroup, value.occupations, [*value.net, (j, i, k, *n)], value.shapes)
    with pytest.raises(ValueError, match="different torsion letters"):
        Protonetstructure(*args, [*value.torsions, "g"])


@pytest.mark.parametrize("entry", [0, False, b"t"])
def test_a_torsion_entry_that_is_neither_str_nor_none_is_rejected(entry: object) -> None:
    value = parse_protonetstructure_label(SARCOSINE)
    for torsions in ([entry] * len(value.net), [*value.torsions[:-1], entry]):
        with pytest.raises(TypeError, match="str or None"):
            Protonetstructure(value.spacegroup, value.occupations, value.net, value.shapes, torsions)


def _arms(order: list[int]) -> tuple[ASUStructure, list[Bond], list[str | None]]:
    """P1, Z' = 1: H-C-C-C-C-C-H whose two halves are swapped by the net, one arm trans and one gauche.

    The torsion about C1-B (D1-C1-B-C2) is 180 degrees and about B-C2 (C1-B-C2-D2) 65 degrees, so
    the two net-equivalent bonds get ``t`` and ``g``; ``order`` lists the atoms as sites.
    """
    import math

    length, angle = 1.5, math.radians(112.0)
    dihedrals = [None, None, None, 180.0, 180.0, 65.0, 180.0]  # dihedral ending at atom k
    points = [(0.0, 0.0, 0.0), (length, 0.0, 0.0)]
    points.append((length - length * math.cos(angle), length * math.sin(angle), 0.0))
    for k in range(3, 7):  # place atom k from k-3, k-2, k-1 (natural extension reference frame)
        a, b, c = points[k - 3], points[k - 2], points[k - 1]
        bc = [c[x] - b[x] for x in range(3)]
        norm = math.sqrt(sum(v * v for v in bc))
        bc = [v / norm for v in bc]
        ab = [b[x] - a[x] for x in range(3)]
        n = [ab[1] * bc[2] - ab[2] * bc[1], ab[2] * bc[0] - ab[0] * bc[2], ab[0] * bc[1] - ab[1] * bc[0]]
        norm = math.sqrt(sum(v * v for v in n))
        n = [v / norm for v in n]
        m = [n[1] * bc[2] - n[2] * bc[1], n[2] * bc[0] - n[0] * bc[2], n[0] * bc[1] - n[1] * bc[0]]
        tau = math.radians(dihedrals[k])  # type: ignore[arg-type]
        d = (
            -length * math.cos(angle),
            length * math.sin(angle) * math.cos(tau),
            length * math.sin(angle) * math.sin(tau),
        )
        points.append(tuple(c[x] + d[0] * bc[x] + d[1] * m[x] + d[2] * n[x] for x in range(3)))
    elements = ["H", "C", "C", "C", "C", "C", "H"]
    shapes = [None, "A2", "A2", "A2", "A2", "A2", None]
    position = {atom: site for site, atom in enumerate(order)}
    sites = [
        WyckoffSite(
            "a", FracVector(tuple(F(x / 20 + 0.5).limit_denominator(10**6) for x in points[atom])), elements[atom]
        )
        for atom in order
    ]
    species = [Species(name=e, chemical_symbols=(e,), concentration=(1,)) for e in ("H", "C")]
    bonds = [(position[atom] + 1, position[atom + 1] + 1, 1, 0, 0, 0) for atom in range(6)]
    return (
        ASUStructure(Cell([[20, 0, 0], [0, 20, 0], [0, 0, 20]]), 1, sites, species),
        bonds,
        [shapes[a] for a in order],
    )


def test_two_net_equivalent_bonds_with_different_letters_give_one_label_from_both_numberings() -> None:
    values = [
        Protonetstructure.from_asu(asu, bonds, shapes=shapes)
        for asu, bonds, shapes in map(_arms, ([0, 1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1, 0], [3, 6, 1, 4, 0, 2, 5]))
    ]
    assert values[0] == values[1] == values[2]
    assert _letters(values[0]) == ["g", "t"]
    text = str(values[0].label)
    assert parse_protonetstructure_label(text) == values[0]
    # the trans letter comes first in the listing: the smallest sequence of letters
    assert [letter for letter in values[0].torsions if letter] == ["g", "t"]


def test_the_protonettype_of_a_structure_keeps_its_letters() -> None:
    value = _value("7219855")
    asu, bonds, _ = _described("7219855")
    with _budget_scope(GUARD):
        anonymous = Protonettype.from_asu(asu, bonds)
    assert anonymous.torsions == value.torsions and _letters(anonymous) == ["t", "t", "t", "t"]
    assert ProtonettypeView(value).unview() == anonymous
    text = str(anonymous.label)
    assert parse_protonettype_label(text) == anonymous and str(parse_protonettype_label(text).label) == text
    assert eval(repr(anonymous), {"Protonettype": Protonettype}) == anonymous
    assert Protonettype(anonymous.spacegroup, anonymous.occupations, anonymous.net, anonymous.shapes) != anonymous
