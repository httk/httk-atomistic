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

from httk.atomistic import ASUStructure, Cell, Species, WyckoffSite
from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
from httk.atomistic.models.prototype.notation import parse_protonetstructure_label
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
        "A10B9C2D2_mP92_14_10e_9e_2e_2e__11~11@2.12.1.2_12~18.3.4_13~14.19.5_14~22.6_15~16.19.7_16~22.8_17~19.23.9_18~20.21_20~10_23~23@2@455:H-C-O-N",
    ),
    # P2_1/c, Z' = 1: a flexible C9H17N3O with one ring (V 30, E 30); 140 s to over 450 s
    # before the search fix
    "4134228": (
        30,
        "A17B9C3D_mP120_14_17e_9e_3e_e__18~19.1.2.3_19~20.4.5_20~21.6.7_21~22.8.9_22~23.10.11_23~28.12.13_24~25.28.14.15_26~27.28.29_27~16.17_25~29.30:H-C-N-O",
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
        "A10B8CD_aP40_2_10i_8i_i_i__9~10.15.16.17_3~4.5.2_4~6.11_5~7.12_6~8.13_7~8.19_8~14_19~10.18_10~20_1L2~1@2.2_2L2:C-H-N-O",
    ),
    # P-1, Z' = 1: a triazolo-benzothiadiazine dioxide C16H14N4O2S (V 37, 4 rings, E 40); 113 s before
    # the search fix
    "7217003": (
        40,
        "A16B14C4D2E_aP74_2_16i_14i_4i_2i_i__12~9.24.25.26_16~33.28.29.30_37~15.33.35.36_1~2.3.4_2~5.17_3~6.18_4~7.31_5~8.19_6~8.20_7~32.33_8~21_10~13.9.22_11~14.9.23_13~15.27_14~32.15_32~34_31~34:C-H-N-O-S",
    ),
    # P2_12_12_1: sarcosine C3H7NO2 (V 13, acyclic, E 12)
    "7219102": (12, "A7B3C2D_oP52_19_7a_3a_2a_a__8~13.1.2.3_9~13.10.4.5_13~6.7_10~11.12:H-C-O-N"),
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
def _described(entry: str, seed: str | None = None) -> tuple[ASUStructure, list[Bond], list[Any]]:
    """The fixture as ``(asu, 1-based bonds, 0-based rows)``, re-described when ``seed`` is given.

    A re-description shuffles the sites, in monoclinic groups applies a random signature-trivial
    cell change, and takes another atom of every general orbit as its reference atom: a random
    operation, then the translation into the unit cube, so that neighbours stay within the
    translations ``raw_rows`` looks at. The bonds are found again from the moved geometry, and no
    contact lies within ``MARGIN`` of its pair's cutoff.
    """
    it, sites, lattice = _source(entry)
    tables = _tables(it)
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
    assert Protonetstructure(value.spacegroup, value.occupations, value.net, value.shapes) == value
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
