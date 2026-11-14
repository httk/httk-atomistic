"""Shapes in the net segment of protonet labels (follow-up 5, part B): grammar, values, canonical form."""

import random
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import httk.core
import protonet_reference as oracle
import pytest
from _protonet_support import random_net, random_shapes, scramble_shaped
from test_protonet_canonical import _cell_changes, _oracle_net, _random_cell_change
from test_structure_shapes import _b8

from httk.atomistic import ASUStructure, BondNetwork, Protonetstructure, Protonettype, SiteBond, Species, WyckoffSite
from httk.atomistic.models.prototype.notation import parse_protonetstructure_label, parse_protonettype_label
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.canonical_classification import canonical_protonetstructure
from httk.atomistic.symmetry.protonet import _canonical_net, _tables
from httk.atomistic.symmetry.shapes import _bond_counts

K2AGSBS4 = "A4B2CD_oP32_34_4c_abc_ab_c"
FRAME = "8I4~1.2_9SS4~3.4_10~1.2.3.4@4"  # S A2 (the default), Ag I4 and SS4, Sb T4
FRAME_SHAPES = ("A2",) * 4 + (None,) * 3 + ("I4", "SS4", "T4")
ONE = AffineOperation.identity()


def _key(result: Any) -> tuple[Any, ...]:
    return result.it_number, result.groups, result.slots, result.codes, result.shapes


# ---------------------------------------------------------------------------------------------
# grammar
# ---------------------------------------------------------------------------------------------
def test_parse_fills_the_defaults_and_rendering_omits_them() -> None:
    value = parse_protonettype_label(f"{K2AGSBS4}__{FRAME}")
    assert value.shapes == FRAME_SHAPES
    assert str(value.label) == f"{K2AGSBS4}__{FRAME}"
    assert parse_protonettype_label(str(Protonettype(34, value.occupations, value.net, value.shapes).label)) == value
    named = parse_protonetstructure_label(f"{K2AGSBS4}__{FRAME}:S-K-Ag-Sb")
    assert named.shapes == FRAME_SHAPES and named.protonettype == value


@pytest.mark.parametrize(
    ("net", "reason"),
    [
        ("8I4~1.2_9SS4~3.4_10T4~1.2.3.4@4", "not a canonical"),  # a written default
        ("8I4~1.2_9XX4~3.4_10~1.2.3.4@4", "unknown shape"),
        ("8I4~1.2_9SS4~3.4_10TBPY5~1.2.3.4@4", "is for 5 bonds, but the site has 4"),
        ("8I4~1.2_9SS4~3.4_10SPY4~1.2.3.4@4", "SPY4 is not in the shape table; its class is SP4"),
        ("8I4~1.2_9SS4~3.4_10OC5~1.2.3.4@4", "unknown shape"),
        ("8I4~1.2_9SS4~3.4_10~1.2.3.4@4_5A2", "site with 0 bond"),
        ("8I4~1.2_9SS4~3.4_10~1.2.3_4L2", "site with 1 bond"),
        ("8I4~1.2_9SS4~3.4_10~1.2.3.4@4_3", "a site number, then a shape"),
        ("8I4~1.2_9SS4~3.4_10~1.2.3_10~4@4", "opens a second group"),
        ("8I4~1.2_8~3.4_9SS4~3.4_10~1.2.3.4@4", "opens a second group"),
        ("8I4~1.2_9SS4~3.4_10~", "invalid protonet destination"),
        ("8I4~1.2_9SS4~3.4_10~1.2.3.4@4_08A2", "a site number, then a shape"),
        ("8I4~1.2_9SS4~3.4_10~1.2.3.4@4_11A2", "outside"),
        ("8i4~1.2_9SS4~3.4_10~1.2.3.4@4", "a site number, then a shape"),
    ],
)
def test_malformed_shapes_are_rejected(net: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_protonettype_label(f"{K2AGSBS4}__{net}")


def _star(symbol: str | None) -> Protonettype:
    """P-1: the inversion centre (site 2) bonded to five lattice translates of the pair on i, ten bonds."""
    rows = [(2, 1, 1, *n) for n in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0))]
    return Protonettype(2, [("i", "A"), ("a", "B")], rows, ("TBPY5", symbol))


def test_a_count_without_a_default_writes_its_shape() -> None:
    value = _star("I10")
    # site 1's TBPY5 is not the five-bond default (SPY5), so it opens a group of its own
    net = "2I10~1.1@1@455.1@1@545.1@1@554.1@1@445_1TBPY5"
    assert str(value.label) == f"A2B_aP3_2_i_a__{net}"
    assert parse_protonettype_label(str(value.label)) == value
    with pytest.raises(ValueError, match="no default shape"):
        parse_protonettype_label(f"A2B_aP3_2_i_a__{net.replace('I10', '')}")
    with pytest.raises(ValueError, match="needs a shape"):
        _star(None)


# ---------------------------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------------------------
LETTERS = ["c"] * 4 + ["a", "b", "c", "a", "b", "c"]
CLASSES = ["A"] * 4 + ["B"] * 3 + ["C"] * 2 + ["D"]
ROWS = [(i, j, 1, 0, 0, 0) for i, j in ((1, 8), (1, 10), (2, 8), (2, 10), (3, 9), (3, 10), (4, 9))] + [
    (4, 10, 3, 0, 0, 0)
]


def test_constructors_require_valid_shapes() -> None:
    occupations = list(zip(LETTERS, CLASSES, strict=True))
    value = Protonettype(34, occupations, ROWS, FRAME_SHAPES)
    assert str(value.label) == f"{K2AGSBS4}__{FRAME}"
    assert eval(repr(value), {"Protonettype": Protonettype}) == value
    reversed_rows = [(j, i, k, *n) for i, j, k, *n in ROWS[:-1]] + [ROWS[-1]]
    assert Protonettype(34, occupations, reversed_rows, FRAME_SHAPES) == value  # rows in either direction
    for shapes, error, match in (
        (None, ValueError, "shapes are missing"),
        (FRAME_SHAPES[:-1], ValueError, "one entry per site"),
        (FRAME_SHAPES[:-1] + (None,), ValueError, "site 10 has 4 bonds and needs a shape"),
        (FRAME_SHAPES[:-1] + ("OC6",), ValueError, "is for 6 bonds"),
        (FRAME_SHAPES[:-1] + ("XY4",), ValueError, "unknown shape"),
        (FRAME_SHAPES[:4] + ("A2",) + FRAME_SHAPES[5:], ValueError, "site with 0 bond"),
        (FRAME_SHAPES[:-1] + (4,), TypeError, "str or None"),
        ("A2", TypeError, "not a str"),
    ):
        with pytest.raises(error, match=match):
            Protonettype(34, occupations, ROWS, shapes)  # type: ignore[arg-type]
    assert value != Protonettype(34, occupations, ROWS, FRAME_SHAPES[:-3] + ("T4", "SS4", "T4"))
    undetermined = Protonettype.undetermined(f"{K2AGSBS4}__X")
    with pytest.raises(ValueError, match="shapes"):
        _ = undetermined.shapes


def test_two_net_equivalent_sites_with_different_shapes_give_one_label() -> None:
    # P2_1/c: two sites of one class on e, each bonded twice to a third site; the smaller symbol takes
    # the smaller number, and A2 is the two-bond default
    occupations = [("e", "A"), ("e", "A"), ("e", "B")]
    rows = [(1, 3, 1, 0, 0, 0), (1, 3, 2, 0, 0, 0), (2, 3, 1, 0, 0, 0), (2, 3, 2, 0, 0, 0)]
    first = Protonettype(14, occupations, rows, ("L2", "A2", "I4"))
    second = Protonettype(14, occupations, rows, ("A2", "L2", "I4"))
    assert first == second and first.shapes == ("A2", "L2", "I4")
    assert str(first.label) == "A2B_mP12_14_2e_e__3I4~1.1@2.2.2@2_2L2"
    assert first != Protonettype(14, occupations, rows, ("A2", "A2", "I4"))


# the reviewer's probes: (it, sites, rows, shapes, preserve_chirality)
TWINS: dict[str, tuple[Any, ...]] = {
    "p21c-l2-a2": (
        14,
        [("X", "e"), ("X", "e"), ("Y", "e")],
        [(0, 2, (0, 0, 0, 0)), (0, 2, (1, 0, 0, 0)), (1, 2, (0, 0, 0, 0)), (1, 2, (1, 0, 0, 0))],
        ["L2", "A2", "I4"],
        True,
    ),
    "pm-counts-differ": (
        6,
        [("X", "a"), ("X", "a"), ("Y", "a")],
        [(0, 2, (0, 0, 0, 0)), (1, 2, (0, 0, 1, 0))],
        [None, "L2", "TS3"],
        True,
    ),
    **{
        f"p21-preserve-{keep}": (
            4,
            [("X", "a"), ("X", "a"), ("Y", "a")],
            [(0, 2, (0, 0, 0, 0)), (0, 2, (1, 0, 0, 0)), (1, 2, (0, 0, 0, 0)), (1, 2, (1, 0, 0, 1))],
            ["L2", "A2", "I4"],
            keep,
        )
        for keep in (True, False)
    },
}


@pytest.mark.parametrize("name", list(TWINS))
def test_twin_probes_do_not_depend_on_the_numbering(name: str) -> None:
    it, sites, rows, shapes, keep = TWINS[name]
    tables, rng = _tables(it), random.Random(name)
    expected = _canonical_net(it, sites, rows, shapes=shapes, preserve_chirality=keep)
    for _ in range(20):
        moved = scramble_shaped(tables, sites, rows, shapes, rng, keep_hand=keep)
        assert _key(_canonical_net(it, moved[0], moved[1], shapes=moved[2], preserve_chirality=keep)) == _key(expected)
    if keep:
        _assert_oracle(it, sites, rows, shapes, expected)


def _assert_oracle(it: int, sites: Any, rows: Any, shapes: Any, result: Any) -> None:
    expected = _oracle_net(it, sites, rows, shapes)
    assert (list(result.slots), list(result.codes), list(result.shapes)) == (
        list(expected["slots"]),
        list(expected["codes"]),
        list(expected["shapes"]),
    )
    assert oracle.render(result.slots, result.codes, result.shapes) == expected["net"]


COUNT_GROUPS = (2, 6, 10, 12, 14, 15, 47, 63, 123, 139, 166, 191, 194, 221, 225, 229)


@pytest.mark.parametrize("it", COUNT_GROUPS)
def test_bond_counts_equal_the_exact_neighbour_points(it: int) -> None:
    # the oracle's count (distinct (site, point) neighbours of each reference atom), on production tables:
    # the oracle's own tables take minutes to build for the cubic groups
    tables, rng = _tables(it), random.Random(f"counts-{it}")
    for _ in range(10):
        sites, rows = random_net(tables, rng)
        mine = _bond_counts(tables, [tables.letter(letter).H for _, letter in sites], rows)
        letters = {letter: {"ref": tables.letter(letter).ref, "H": tables.letter(letter).H} for _, letter in sites}
        assert list(mine) == oracle.bond_counts(_Letters(tables, letters), sites, rows)


class _Letters:
    """Production tables with the oracle's dict-valued ``letter``."""

    def __init__(self, tables: Any, letters: dict[str, dict[str, Any]]) -> None:
        self.tables, self.letters = tables, letters

    def letter(self, letter: str) -> dict[str, Any]:
        return self.letters[letter]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tables, name)


# ---------------------------------------------------------------------------------------------
# description independence with random valid shapes carried with the sites
# ---------------------------------------------------------------------------------------------
def _shaped_net(tables: Any, rng: random.Random) -> tuple[list[Any], list[Any], list[str | None]]:
    for _ in range(100):
        sites, rows = random_net(tables, rng)
        if rows:
            return sites, rows, random_shapes(tables, sites, rows, rng)
    raise AssertionError("random_net drew 100 nets without rows")


# special positions (139, 166, 221, 62), monoclinic cells (12, 14, 15), P-1 (2, cell-keeping cosets)
SHAPE_GROUPS = (2, 4, 12, 14, 15, 34, 62, 139, 166, 221)


@pytest.mark.parametrize(
    ("it", "nets"),
    [*((it, 3) for it in SHAPE_GROUPS), *(pytest.param(it, 15, marks=pytest.mark.extended) for it in SHAPE_GROUPS)],
)
def test_random_nets_with_shapes_are_description_invariant(it: int, nets: int) -> None:
    tables, rng = _tables(it), random.Random(f"shapes-{it}-{nets}")
    for _ in range(nets):
        sites, rows, shapes = _shaped_net(tables, rng)
        expected = _key(_canonical_net(it, sites, rows, shapes=shapes))
        for _ in range(3):
            moved = scramble_shaped(tables, sites, rows, shapes, rng)
            assert _key(_canonical_net(it, moved[0], moved[1], shapes=moved[2])) == expected


@pytest.mark.parametrize("it", [4, 19, 80, 152])
def test_random_nets_with_shapes_identify_the_hands_without_preserve_chirality(it: int) -> None:
    tables, rng = _tables(it), random.Random(f"shaped-hands-{it}")
    for _ in range(3):
        sites, rows, shapes = _shaped_net(tables, rng)
        expected = _key(_canonical_net(it, sites, rows, shapes=shapes, preserve_chirality=False))
        for _ in range(3):
            moved = scramble_shaped(tables, sites, rows, shapes, rng, keep_hand=False)
            result = _canonical_net(it, moved[0], moved[1], shapes=moved[2], preserve_chirality=False)
            assert _key(result) == expected


@pytest.mark.parametrize("it", [2, 4, 12, 14, 15])
def test_shaped_molecules_do_not_depend_on_the_cell(it: int) -> None:
    # three atoms on the general position, one bonded to an image of itself (a dimer or a chain along b)
    tables, rng = _tables(it), random.Random(f"shaped-cells-{it}")
    general = max(tables.spacegroup.wyckoff, key=lambda w: (w.free_count, w.multiplicity)).letter
    sites = [("X", general), ("X", general), ("Y", general)]
    rows = [(0, 2, (0, 0, 0, 0)), (1, 2, (0, 0, 0, 0)), (0, 0, (tables.nops - 1, 1, 0, 0))]
    changes = _cell_changes(tables)
    for _ in range(3):
        shapes = random_shapes(tables, sites, rows, rng)
        expected = _key(_canonical_net(it, sites, rows, shapes=shapes))
        for _ in range(3):
            coset = _random_cell_change(tables, changes, rng)
            moved = scramble_shaped(tables, sites, rows, shapes, rng, coset=coset)
            assert _key(_canonical_net(it, moved[0], moved[1], shapes=moved[2])) == expected


@pytest.mark.parametrize("it", [4, 14, 34, 62])
def test_random_nets_with_shapes_match_the_oracle(it: int) -> None:
    tables, rng = _tables(it), random.Random(f"shaped-oracle-{it}")
    for _ in range(3):
        sites, rows, shapes = _shaped_net(tables, rng)
        _assert_oracle(it, sites, rows, shapes, _canonical_net(it, sites, rows, shapes=shapes))


# ---------------------------------------------------------------------------------------------
# B8: the prototype values of the brief (default A2), as implemented
# ---------------------------------------------------------------------------------------------
# name -> (canonical net, the brief's B8 net); the exact minimum differs from the brief's prototype,
# which kept one numbering and its first tied realisation, in two cases (design note section 8):
# K2AgSbS4's last code is @4 (not @4@554), and diopside's 6~ group takes @2@545 (and 4~ then 2@5).
# The brief's nets are written with the present defaults (diopside's DD8 Ca site, once written, is now the default).
B8: dict[str, tuple[str, str]] = {
    "AlCl3": ("A3B_mC16_12_ij_g__3~1.2.2@4:Cl-Al", "3~1.2.2@4"),
    "AgO": ("AB_mP8_14_ab_e__1SP4~3.3@3_3TPY3~2_2L2:Ag-O", "1SP4~3.3@3_3TPY3~2_2L2"),
    "CuCl2 2.6": ("A2B_mC6_12_i_a__2SP4~1@4:Cl-Cu", "2SP4~1@4"),
    "CuCl2 3.1": ("A2B_mC6_12_i_a__2~1.1@4_1I3:Cl-Cu", "2~1.1@4_1I3"),
    "K2AgSbS4": (f"{K2AGSBS4}__{FRAME}:S-K-Ag-Sb", "8I4~1.2_9SS4~3.4_10~1.2.3.4@4@554"),
    "diopside": (
        "A6B2CD_mC40_15_3f_f_e_e__5~1.1@2.2.3_6~2.2@2@545.3@2@545_4~1.1@6.2@5.3@2_1I4_3I3:O-Si-Ca-Mg",
        "5~1.1@2.2.3_6~2.2@2@565.3@2@565_4~1.1@6.2@5@545.3@2_1I4_3I3",
    ),
}
B8_DIFFERS = {"K2AgSbS4", "diopside"}
B8_MOLECULAR = {
    # the file's own bond loop
    "7219102": "8~13.1.2.3_9~13.10.4.5_13~6.7_10~11.12",
    # hydrogen-complete covalent bonds
    "7217521": "9~10.14.1.2_10~15.3.4_11~12.15.5.6_12~13.14.7_13TPY3~14@2.8_14TPY3",
}


def _net(label: str) -> str:
    return label.partition("__")[2].rpartition(":")[0]


@pytest.mark.parametrize("name", list(B8))
def test_b8_nets(name: str) -> None:
    label, brief = B8[name]
    value = canonical_protonetstructure(Protonetstructure.from_asu(_b8(name)), timeout=None)
    assert str(value.label) == label
    assert (_net(label) == brief) is (name not in B8_DIFFERS)
    assert parse_protonetstructure_label(label) == value


@pytest.mark.parametrize("name", ["AgO", "CuCl2 3.1", "K2AgSbS4"])
def test_b8_nets_match_the_oracle(name: str) -> None:
    value = Protonetstructure.from_asu(_b8(name))
    sites = [(o.species.name, o.wyckoff) for o in value.occupations]
    rows = [(i - 1, j - 1, (k - 1, *n)) for i, j, k, *n in value.net]
    it = value.spacegroup.it_number
    _assert_oracle(it, sites, rows, value.shapes, _canonical_net(it, sites, rows, shapes=value.shapes))


def test_b8_molecular_nets() -> None:
    from test_protonet_molecular import MOLECULAR

    path = str(Path(__file__).parent / "fixtures" / "cod" / "7219102.cif")
    loop = canonical_protonetstructure(httk.core.load(path), timeout=None)
    assert _net(str(loop.label)) == B8_MOLECULAR["7219102"]
    assert _net(MOLECULAR["7217521"][1]) == B8_MOLECULAR["7217521"]


# ---------------------------------------------------------------------------------------------
# from_asu with supplied shapes
# ---------------------------------------------------------------------------------------------
C = Species("C", ("C",), (1,))
C1, C2 = Species("C1", ("C",), (1,)), Species("C2", ("C",), (1,))
CHAIN = [(1, 2, 1, 0, 0, 0), (2, 1, 1, 1, 0, 0)]  # a chain along a through both sites


def _pair(species: tuple[Species, Species], shapes: Any = None, *, bonds: bool = True) -> ASUStructure:
    """P1, a doubled along a: one atom at 0 and one at 1/2; with ``bonds`` the chain as the network."""
    sites = [WyckoffSite("a", (0, 0, 0), species[0].name), WyckoffSite("a", (F(1, 2), 0, 0), species[1].name)]
    pairs = (SiteBond(0, 1, ONE), SiteBond(1, 0, AffineOperation(ONE.matrix, (1, 0, 0))))
    network = BondNetwork(pairs, shapes) if bonds else None
    unique = list(dict.fromkeys(species))
    return ASUStructure([[6, 0, 0], [0, 4, 0], [0, 0, 5]], 1, sites, unique, bonds=network)


def test_from_asu_uses_supplied_shapes_and_the_baseline_for_the_rest() -> None:
    baseline = Protonetstructure.from_asu(_pair((C, C)))
    assert baseline.shapes == ("L2", "L2")  # a straight chain
    supplied = Protonetstructure.from_asu(_pair((C, C), ("A2", None)))
    assert sorted(supplied.shapes) == ["A2", "L2"] and supplied != baseline
    plain = _pair((C, C), bonds=False)
    assert Protonetstructure.from_asu(plain, CHAIN, shapes=("A2", None)) == supplied
    assert Protonetstructure.from_asu(plain, CHAIN) == baseline
    # index bonds ignore the structure's own network and shapes
    assert Protonetstructure.from_asu(_pair((C, C), ("A2", "A2")), CHAIN) == baseline
    with pytest.raises(ValueError, match="needs bonds given as index tuples"):
        Protonetstructure.from_asu(_pair((C, C)), shapes=("A2", None))
    with pytest.raises(ValueError, match=r"shapes\[0\]: shape 'T4' is for 4 bonds, but the site has 2"):
        Protonetstructure.from_asu(plain, CHAIN, shapes=("T4", None))
    assert Protonettype.from_asu(plain, CHAIN, shapes=("A2", None)) == supplied.protonettype


def test_from_asu_refuses_merging_sites_with_different_supplied_shapes() -> None:
    # C1 and C2 are one class, whose structure has the halved cell: their sites merge
    merged = Protonetstructure.from_asu(_pair((C1, C2), ("A2", None)))
    assert merged.shapes == ("A2",)  # the site without a symbol gives way
    with pytest.raises(ValueError, match=r"different supplied shapes merge under the recognized symmetry IT \d+"):
        Protonetstructure.from_asu(_pair((C1, C2), ("A2", "L2")))
    with pytest.raises(ValueError, match="different supplied shapes merge"):
        Protonetstructure.from_asu(_pair((C1, C2), bonds=False), CHAIN, shapes=("A2", "L2"))


def test_a_bond_onto_the_atom_itself_through_its_site_symmetry_is_rejected() -> None:
    # P2_1/c: B on the inversion centre a; its row through the inversion maps the atom onto itself
    with pytest.raises(ValueError, match=r"row \(2, 2, 2, 0, 0, 0\) bonds an atom to itself"):
        Protonettype(14, [("e", "A"), ("a", "B")], [(2, 2, 2, 0, 0, 0), (1, 2, 1, 0, 0, 0)])


def test_from_asu_refuses_a_site_at_a_special_point_of_its_letter() -> None:
    # caller's site 2 is the general letter e placed on an inversion centre; it has no bonds
    sites = [WyckoffSite("e", (F(1, 5), F(1, 3), F(1, 7)), "C"), WyckoffSite("e", (0, 0, 0), "C")]
    plain = ASUStructure([[5, 0, 0], [0, 6, 0], [-1, 0, 7]], 14, sites, [C])
    message = r"site 2 \(C on Wyckoff letter 'e'\) lies at a special point of its letter"
    chain = [(1, 1, 1, 1, 0, 0)]
    for shapes in (None, ("A2", None)):
        with pytest.raises(ValueError, match=message):
            Protonetstructure.from_asu(plain, chain, shapes=shapes)
    networked = ASUStructure(plain.cell, 14, sites, [C], bonds=[SiteBond(0, 0, AffineOperation(ONE.matrix, (1, 0, 0)))])
    with pytest.raises(ValueError, match=message):
        Protonetstructure.from_asu(networked)
