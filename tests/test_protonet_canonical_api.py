"""Public protonet canonicalisers and ``Protonetstructure.from_asu``, end to end from structures."""

import dataclasses
import random
from fractions import Fraction as F
from functools import cache
from pathlib import Path
from typing import Any

import httk.core
import protonet_reference as oracle
import pytest
from _protonet_support import (
    default_shapes,
    handmade,
    lattice_from_parameters,
    load_structure,
    raw_rows,
    scramble_shaped,
)
from httk.core import FracVector
from test_protonet_canonical import QUARTZ, QUARTZ_CELL, SI_O, _enantiomorph

import httk.atomistic.symmetry.protonet as protonet_module
from httk.atomistic import ASUStructure, Cell, Species, WyckoffSite
from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
from httk.atomistic.models.protonettype.protonettype import Protonettype
from httk.atomistic.models.prototype.notation import (
    parse_protonetstructure_label,
    parse_protonettype_label,
    render_protonet,
)
from httk.atomistic.symmetry.canonical_classification import (
    canonical_bare_protostructure,
    canonical_bare_prototype,
    canonical_protonetstructure,
    canonical_protonettype,
)
from httk.atomistic.symmetry.limits import CanonicalizationLimitError
from httk.atomistic.symmetry.protonet import _canonical_net, _search, _tables
from httk.atomistic.symmetry.spacegroup import Spacegroup

type Bond = tuple[int, int, int, int, int, int]

K2AGSBS4 = str(Path(__file__).parent / "fixtures" / "cod" / "1509425.cif")
FRAME = {frozenset(("Sb", "S")): 3.0, frozenset(("Ag", "S")): 3.0}
FULL_NET = (
    "5~1.2.3.4_7~1.1@3.2.2@4@554.3@3.4.4@3@554_6I~1@3.2.3@2_1TPR6~8.10_2TPR6~8@1@556.10@1@556"
    "_3~9@2.10@3@545_4I~9@4.10@3@545_8I_9SS4"
)
FRAME_NET = "8I~1.2_9SS4~3.4_10~1.2.3.4@4"  # the B8 net of follow-up 5
CUBIC = (5.64, 5.64, 5.64, 90, 90, 90)
NACL = {frozenset(("Na", "Cl")): 3.0}
ELEMENT = {"A": "C", "B": "N"}  # the made-up P2_1 classes A and B; the label suffix names their classes C, N

# name -> (it number, spec or None for the CIF, cell, cutoffs)
CASES: dict[str, tuple[Any, ...]] = {
    "k2agsbs4-frame": (34, None, None, FRAME),
    "k2agsbs4-full": (34, None, None, {**FRAME, frozenset(("K", "S")): 3.75}),
    "nacl": (225, [("Na", "a", ()), ("Cl", "b", ())], CUBIC, NACL),
    "nacl-exchanged": (225, [("Cl", "a", ()), ("Na", "b", ())], CUBIC, NACL),
    "diamond": (227, [("C", "a", ())], (3.567, 3.567, 3.567, 90, 90, 90), {frozenset(("C",)): 1.7}),
    "wurtzite": (
        186,
        [("Zn", "b", (F(0),)), ("S", "b", (F(3, 8),))],
        (3.82, 3.82, 6.26, 90, 90, 120),
        {frozenset(("Zn", "S")): 2.6},
    ),
    "pdse2": (
        61,
        [("Pd", "a", ()), ("Se", "c", (F(1112, 10000), F(1179, 10000), F(4073, 10000)))],
        (5.7457, 5.8679, 7.6946, 90, 90, 90),
        {frozenset(("Pd", "Se")): 2.6, frozenset(("Se",)): 2.6},
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
    ),
}
LABELS = {
    "k2agsbs4-frame": f"A4B2CD_oP32_34_4c_abc_ab_c__{FRAME_NET}:S-K-Ag-Sb",
    "k2agsbs4-full": f"A4B2CD_oP32_34_4c_abc_ab_c__{FULL_NET}:S-K-Ag-Sb",
    "nacl": "AB_cF8_225_a_b__1~2:Cl-Na",
    "nacl-exchanged": "AB_cF8_225_a_b__1~2:Cl-Na",
    "diamond": "A_cF8_227_a__1~1@14@654:C",
    "wurtzite": "AB_hP4_186_b_b__1~2.2@2:S-Zn",
    "pdse2": "A2B_oP12_61_c_a__2SP4~1.1@3_1TPY3~1@2@545:Se-Pd",
    "p21-chain": "A2B_mP6_4_2a_a__1I~1@2.2.2@2.3_2I~3:N-C",  # renamed: species B and A are classes N and C
}


def _species(name: str) -> Species:
    return Species(name=name, chemical_symbols=(ELEMENT.get(name, name),), concentration=(1,))


def _asu(it: int, sites: list[tuple[str, str, tuple[F, ...]]], lattice: Any) -> ASUStructure:
    """An element-named ASU structure; ``from_asu`` reads only letters and species from it."""
    return ASUStructure(
        Cell(lattice),
        it,
        [WyckoffSite(letter, FracVector(tuple(params)), name) for name, letter, params in sites],
        [_species(name) for name in dict.fromkeys(name for name, _, _ in sites)],
    )


def _bonds(rows: list[Any]) -> list[Bond]:
    """0-based oracle rows to 1-based ``from_asu`` bonds."""
    return [(i + 1, j + 1, g[0] + 1, g[1], g[2], g[3]) for i, j, g in rows]


@cache
def _case(name: str) -> tuple[ASUStructure, tuple[Bond, ...], list[Any], list[Any]]:
    """Return ``(asu, bonds, sites, rows)``: bonds index ``asu.wyckoff_sites``; sites/rows are 0-based."""
    it, spec, cell, cutoffs = CASES[name]
    if spec is None:
        loaded = load_structure(K2AGSBS4)
        assert loaded is not None
        it, sites, refs, lattice = loaded
        cif = httk.core.load(K2AGSBS4)
        # the CIF names charged species (S-2, K+1, ...); rename them to their elements
        spec = [
            (cls, ws.wyckoff, tuple(ws.free_params.to_fractions())) for (cls, _), ws in zip(sites, cif.wyckoff_sites)
        ]
        asu = _asu(it, spec, cif.cell)
    else:
        sites, refs, lattice = handmade(it, spec, cell)
        asu = _asu(it, spec, lattice)
    rows, _ = raw_rows(_tables(it), sites, refs, lattice, cutoffs)
    return asu, tuple(_bonds(rows)), sites, rows


def _input_shapes(asu: ASUStructure, bonds: tuple[Bond, ...]) -> tuple[str | None, ...]:
    """The baseline shape of every site of ``asu`` with the index bonds as its network."""
    from httk.atomistic.models.structure.asu import _affine, _standard_elements
    from httk.atomistic.models.structure.network import BondNetwork, SiteBond
    from httk.atomistic.symmetry.shapes import _site_shapes

    standard = _standard_elements(asu.spacegroup)
    network = [SiteBond(i - 1, j - 1, _affine(standard.operation((k - 1, *n)))) for i, j, k, *n in bonds]
    return _site_shapes(asu._with_bonds(BondNetwork(tuple(network))))


@cache
def _value(name: str) -> Protonetstructure:
    asu, bonds, _, _ = _case(name)
    return Protonetstructure.from_asu(asu, bonds)


@cache
def _canonical(name: str) -> Protonetstructure:
    return canonical_protonetstructure(_value(name))


def _net_part(label: str) -> str:
    return label.partition("__")[2].rpartition(":")[0] if ":" in label else label.partition("__")[2]


# ---------------------------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(CASES))
def test_canonical_labels(name: str) -> None:
    label = str(_canonical(name).label)
    assert label == LABELS[name]
    assert str(canonical_protonettype(_value(name)).label) == label.rpartition(":")[0]


def test_k2agsbs4_levels_nest_as_in_the_brief() -> None:
    ps = _canonical("k2agsbs4-frame")
    nt = canonical_protonettype(ps)
    assert str(nt.anonymous_formula) == "A4B2CD"
    assert str(canonical_bare_prototype(nt).label) == "A4B2CD_oP32_34_4c_abc_ab_c"
    assert str(nt.label) == f"A4B2CD_oP32_34_4c_abc_ab_c__{FRAME_NET}"
    # no classes tie here, so the assigned bare part equals the canonical bare protostructure (R5)
    assert str(canonical_bare_protostructure(ps).label) == "A4B2CD_oP32_34_4c_abc_ab_c:S-K-Ag-Sb"
    assert str(ps.label) == f"A4B2CD_oP32_34_4c_abc_ab_c__{FRAME_NET}:S-K-Ag-Sb"


# Species exchanges that keep the prototype label: NaCl (a and b swapped by an origin shift) and
# wurtzite (two classes on b). There the net orders the species (D3), so the cut label is not
# promised to be the canonical bare protostructure label; elsewhere it is.
TIED = frozenset({"nacl", "nacl-exchanged", "wurtzite"})


@pytest.mark.parametrize("name", list(CASES))
def test_levels_nest_as_prefixes(name: str) -> None:
    ps = _canonical(name)
    nt = canonical_protonettype(ps)
    formulatype, bare, label = str(nt.anonymous_formula), str(canonical_bare_prototype(nt).label), str(nt.label)
    assert bare.startswith(formulatype + "_") and label.startswith(bare + "__")
    assert str(ps.label).rpartition(":")[0] == label
    assert label.partition("__")[0] == bare
    assert ps.bare_protostructure == canonical_bare_protostructure(_value(name))
    if name not in TIED:
        bare_part, _, rest = str(ps.label).partition("__")
        assert f"{bare_part}:{rest.rpartition(':')[2]}" == str(ps.bare_protostructure.label)


def test_k2agsbs4_with_k_s_contacts() -> None:
    assert _net_part(str(_canonical("k2agsbs4-full").label)) == FULL_NET


def test_nacl_species_assignments_agree() -> None:
    # one net, one value: the constructor already exchanges the species to the D3 minimum
    assert _value("nacl") == _value("nacl-exchanged") == _canonical("nacl")
    assert _canonical("nacl") == _canonical("nacl-exchanged")
    assert str(_canonical("nacl").label) == "AB_cF8_225_a_b__1~2:Cl-Na"


def test_wurtzite_suffix_is_the_d3_minimum_of_the_oracle() -> None:
    _, _, sites, rows = _case("wurtzite")
    expected = oracle.canonical_net(186, sites, rows)
    names = min(tuple(r["classes"]) for r in expected["realisations"])
    label = str(_canonical("wurtzite").label)
    assert _net_part(label) == "1~2.2@2" and label.rpartition(":")[2] == "-".join(names)


@pytest.mark.parametrize("name", list(CASES))
def test_label_is_the_canonical_render(name: str) -> None:
    value = _value(name)
    sites = [(o.species.name, o.wyckoff) for o in value.occupations]
    rows = [(i - 1, j - 1, (k - 1, n1, n2, n3)) for i, j, k, n1, n2, n3 in value.net]
    result = _canonical_net(value.spacegroup.it_number, sites, rows, shapes=value.shapes)
    net = [(i, j, g[0] + 1, g[1], g[2], g[3]) for (i, j), g in zip(result.slots, result.codes, strict=True)]
    expected = render_protonet(Spacegroup.standard(result.it_number), result.groups, net, shapes=result.shapes)
    assert str(canonical_protonettype(value).label) == expected


@pytest.mark.parametrize("name", list(CASES))
def test_constructor_equals_the_canonicaliser_on_a_canonical_bare_part(name: str) -> None:
    value = _value(name)
    assert value.protonettype.bare_prototype == canonical_bare_prototype(value)
    assert value == _canonical(name) and value.protonettype == canonical_protonettype(value)
    anonymous = value.protonettype
    rows = _translated(anonymous, random.Random(name))
    assert Protonettype(anonymous.spacegroup, anonymous.occupations, rows, anonymous.shapes) == anonymous
    assert eval(repr(value), {"Protonetstructure": Protonetstructure, "Species": Species, "Fraction": F}) == value


@pytest.mark.parametrize("name", list(CASES))
def test_canonicalisation_is_idempotent_and_parses_back(name: str) -> None:
    ps = _canonical(name)
    nt = canonical_protonettype(ps)
    assert canonical_protonetstructure(ps) == ps and canonical_protonettype(nt) == nt
    assert canonical_protonettype(_value(name).protonettype) == nt
    assert parse_protonetstructure_label(str(ps.label)) == ps
    assert parse_protonettype_label(str(nt.label)) == nt


# a block of tied classes (equal site count and letters) in P2_1/c, some bonded and some not
TIED_BLOCKS: dict[str, tuple[list[str], list[Bond]]] = {
    "one-site-classes": (["e"] * 4, [(3, 3, 2, 0, 0, 0), (3, 4, 1, 0, 1, 0)]),
    "two-site-classes": (["a", "e"] * 3, [(5, 6, 1, 0, 0, 0), (6, 6, 2, 1, 0, 0)]),
}


@pytest.mark.parametrize("name", list(TIED_BLOCKS))
def test_bonded_tied_classes_take_the_lowest_site_numbers(name: str) -> None:
    letters, net = TIED_BLOCKS[name]
    occupations = [(letter, "ABCD"[s // len(set(letters))]) for s, letter in enumerate(letters)]
    shapes = default_shapes(14, letters, net)
    value = Protonettype(14, occupations, net, shapes)
    bonded = {s for row in value.net for s in row[:2]}
    assert bonded == set(range(1, len(bonded) + 1)) and max(bonded) < len(letters)
    assert value == canonical_protonettype(value)
    sites = [(label, letter) for letter, label in occupations]
    expected = oracle.canonical_net(14, sites, [(i - 1, j - 1, (k - 1, *n)) for i, j, k, *n in net], shapes=shapes)
    assert [((i, j), (k - 1, *n)) for i, j, k, *n in value.net] == [
        (tuple(slot), tuple(code)) for slot, code in zip(expected["slots"], expected["codes"], strict=True)
    ]


# ---------------------------------------------------------------------------------------------
# invariance over equivalent input values
# ---------------------------------------------------------------------------------------------
def _translated(value: Protonetstructure | Protonettype, rng: random.Random) -> list[Bond]:
    """The value's rows with the general-position reference atoms moved by large lattice translations."""
    tables = _tables(value.spacegroup.it_number)
    shift = {
        s + 1: (rng.randint(-40, 40), rng.randint(-40, 40), rng.randint(-40, 40))
        for s, o in enumerate(value.occupations)
        if tables.letter(o.wyckoff).general
    }
    rows = []
    for i, j, k, *n in value.net:
        ti, tj, W = shift.get(i, (0, 0, 0)), shift.get(j, (0, 0, 0)), tables.W[k - 1]
        rows.append((i, j, k, *(n[r] + ti[r] - sum(int(W[r][c]) * tj[c] for c in range(3)) for r in range(3))))
    return rows


def _unreduced(asu: ASUStructure, bonds: tuple[Bond, ...], rng: random.Random) -> tuple[ASUStructure, list[Bond]]:
    """The structure with free parameters moved by large integers, and its bonds relative to the moved atoms."""
    tables, sites, shifts = _tables(asu.spacegroup.it_number), [], []
    for site in asu.wyckoff_sites:
        position = asu.spacegroup.wyckoff_position(site.wyckoff)
        params = site.free_params + FracVector(tuple(rng.randint(-30, 30) for _ in range(position.free_count)))
        old, new = position.coordinates(site.free_params)[0], position.coordinates(params)[0]
        shifts.append(tuple(int(b - a) for a, b in zip(old.to_fractions(), new.to_fractions(), strict=True)))
        sites.append(WyckoffSite(site.wyckoff, params, site.species))
    moved = []
    for i, j, k, *n in bonds:
        ti, tj, W = shifts[i - 1], shifts[j - 1], tables.W[k - 1]
        moved.append((i, j, k, *(n[r] + ti[r] - sum(int(W[r][c]) * tj[c] for c in range(3)) for r in range(3))))
    return ASUStructure(asu.cell, asu.spacegroup, sites, asu.species), moved


def _variants(name: str, rng: random.Random, count: int) -> list[Protonetstructure]:
    """Valid, non-canonical spellings of one structure through the public API.

    Shuffled sites, reversed bonds, free parameters moved by large integers and reference atoms
    moved by large lattice translations keep the description, so they give the constructor's value
    itself; scrambled cosets and gauges also change the description.
    """
    asu, bonds, sites, rows = _case(name)
    tables, it = _tables(asu.spacegroup.it_number), asu.spacegroup.it_number
    out = []
    perm = list(range(len(asu.wyckoff_sites)))
    rng.shuffle(perm)
    new = {old: index for index, old in enumerate(perm)}
    shuffled = ASUStructure(asu.cell, it, [asu.wyckoff_sites[old] for old in perm], asu.species)
    out.append(Protonetstructure.from_asu(shuffled, [(new[i - 1] + 1, new[j - 1] + 1, *rest) for i, j, *rest in bonds]))
    reversed_bonds = []
    for i, j, k, *n in bonds:
        e = tables.inv((k - 1, n[0], n[1], n[2]))
        reversed_bonds.append((j, i, e[0] + 1, e[1], e[2], e[3]))
    out.append(Protonetstructure.from_asu(asu, reversed_bonds))
    out.append(Protonetstructure.from_asu(*_unreduced(asu, bonds, rng)))
    value = _value(name)
    out.append(Protonetstructure(value.spacegroup, value.occupations, _translated(value, rng), value.shapes))
    assert all(v == value for v in out)
    # the scrambled descriptions below have made-up coordinates: the real shapes travel with their sites
    shapes = _input_shapes(asu, bonds)
    for _ in range(count):
        sites2, rows2, shapes2 = scramble_shaped(tables, sites, rows, shapes, rng)
        spec = [
            (
                cls,
                letter,
                tuple(F(13 + n, 97) for n in range(Spacegroup.standard(it).wyckoff_position(letter).free_count)),
            )
            for cls, letter in sites2
        ]
        out.append(Protonetstructure.from_asu(_asu(it, spec, asu.cell), _bonds(rows2), shapes=shapes2))
    return out


@pytest.mark.parametrize(
    ("name", "trials"),
    [
        ("k2agsbs4-frame", 3),
        ("wurtzite", 3),
        ("pdse2", 20),
        pytest.param("k2agsbs4-frame", 20, marks=pytest.mark.extended),
        pytest.param("wurtzite", 20, marks=pytest.mark.extended),
    ],
)
def test_equivalent_values_canonicalise_identically(name: str, trials: int) -> None:
    reference, anonymous = _canonical(name), canonical_protonettype(_canonical(name))
    for value in _variants(name, random.Random(f"{name}-{trials}"), trials):
        assert canonical_protonetstructure(value) == reference
        assert canonical_protonettype(value) == anonymous
        if value.protonettype.bare_prototype == canonical_bare_prototype(value):
            assert value == reference


# ---------------------------------------------------------------------------------------------
# chirality, guard, limits
# ---------------------------------------------------------------------------------------------
def test_quartz_enantiomorphs_agree_only_without_preserved_chirality() -> None:
    values = []
    for it, spec in ((152, QUARTZ), (154, _enantiomorph(QUARTZ))):
        sites, refs, lattice = handmade(it, spec, QUARTZ_CELL)
        rows, _ = raw_rows(_tables(it), sites, refs, lattice, SI_O)
        values.append(Protonetstructure.from_asu(_asu(it, spec, lattice), _bonds(rows)))
    kept = [canonical_protonetstructure(v) for v in values]
    flipped = [canonical_protonetstructure(v, preserve_chirality=False) for v in values]
    assert [v.spacegroup.it_number for v in kept] == [152, 154] and kept[0] != kept[1]
    assert flipped[0] == flipped[1] and flipped[0].spacegroup.it_number == 152
    anonymous = [canonical_protonettype(v, preserve_chirality=False) for v in values]
    assert anonymous[0] == anonymous[1] != canonical_protonettype(values[1])


def test_guard_rejects_a_net_search_on_another_bare_prototype(monkeypatch: pytest.MonkeyPatch) -> None:
    value, real = _value("nacl"), protonet_module._canonical_net

    def wrong(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        moved = tuple(dataclasses.replace(r, letters=("c",) * len(r.letters)) for r in result.realisations)
        # shapes valid for the moved letters, so that only the bare prototype is wrong
        net = [(i, j, g[0] + 1, *g[1:]) for (i, j), g in zip(result.slots, result.codes, strict=True)]
        shapes = tuple(default_shapes(result.it_number, moved[0].letters, net))
        return dataclasses.replace(result, realisations=moved, shapes=shapes)

    monkeypatch.setattr(protonet_module, "_canonical_net", wrong)
    with pytest.raises(ValueError, match="bare prototype"):
        canonical_protonettype(value)
    with pytest.raises(ValueError, match="bare prototype"):
        canonical_protonetstructure(value)


def test_unwritable_canonical_translation_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    value, real = _value("nacl"), protonet_module._canonical_net

    def long(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        k, _, n2, n3 = result.codes[0]
        return dataclasses.replace(result, codes=((k, 6, n2, n3), *result.codes[1:]))

    monkeypatch.setattr(protonet_module, "_canonical_net", long)
    with pytest.raises(ValueError, match=r"cannot write this canonical net.*-5\.\.\+4"):
        canonical_protonettype(value)
    with pytest.raises(ValueError, match="cannot write this canonical net"):
        canonical_protonetstructure(value)


def test_tiny_timeout_raises_limit_error_with_on_limit_raise() -> None:
    label = str(_value("k2agsbs4-frame").protonettype.label)
    _search.cache_clear()
    with pytest.raises(CanonicalizationLimitError):  # the budget also covers parsing the label
        canonical_protonettype(label, timeout=1e-9, on_limit="raise")
    with pytest.raises(CanonicalizationLimitError):
        canonical_protonetstructure(_value("k2agsbs4-frame"), timeout=1e-9, on_limit="raise")
    with pytest.raises(CanonicalizationLimitError):
        canonical_protonettype(_value("k2agsbs4-frame"), timeout=1e-9, on_limit="raise")


def test_tiny_timeout_gives_undetermined_labels() -> None:
    # nothing is determined within 1e-9 s; a label string is read as its own description, without a search,
    # so the bare part is tried first and also runs out
    label = str(_value("k2agsbs4-frame").protonettype.label)
    _search.cache_clear()
    assert str(canonical_protonettype(label, timeout=1e-9).label) == "A4B2CD_oP32_34_X__X"
    assert str(canonical_protonettype(_value("k2agsbs4-frame"), timeout=1e-9).label) == "A4B2CD_oP32_34_X__X"
    value = canonical_protonetstructure(_value("k2agsbs4-frame"), timeout=1e-9)
    assert str(value.label) == "A4B2CD_oP32_34_X__X:S-K-Ag-Sb" and not value.determined


# ---------------------------------------------------------------------------------------------
# from_asu
# ---------------------------------------------------------------------------------------------
def test_from_asu_rejects_invalid_inputs() -> None:
    asu, bonds, _, _ = _case("nacl")
    rhombohedral = ASUStructure(
        Cell(lattice_from_parameters(4.5, 4.5, 11.8, 90, 90, 120)),
        Spacegroup.from_setting("166:R"),
        [WyckoffSite("a", FracVector(()), "C")],
        [_species("C")],
    )
    with pytest.raises(ValueError, match="standard setting"):
        Protonetstructure.from_asu(rhombohedral, [])
    retained = ASUStructure(
        asu.cell,
        225,
        [WyckoffSite("a", FracVector(()), "Na", representative=FracVector((0, 0, 0))), asu.wyckoff_sites[1]],
        asu.species,
    )
    with pytest.raises(ValueError, match="drop it first"):
        Protonetstructure.from_asu(retained, bonds)
    with pytest.raises(ValueError, match="site indices"):
        Protonetstructure.from_asu(asu, [(1, 3, 1, 0, 0, 0)])
    with pytest.raises(ValueError, match="operation number"):
        Protonetstructure.from_asu(asu, [(1, 2, 193, 0, 0, 0)])
    with pytest.raises(ValueError, match="itself"):
        Protonetstructure.from_asu(asu, [(1, 1, 1, 0, 0, 0)])
    with pytest.raises(ValueError, match="six integers"):
        Protonetstructure.from_asu(asu, [(1, 2, 1, 0, 0)])  # type: ignore[list-item]
    # the input takes any translation, but this long Na-Cl bond keeps it in the canonical net (D4)
    with pytest.raises(ValueError, match=r"\(D4\)"):
        Protonetstructure.from_asu(asu, [(1, 2, 1, 9, 0, 0)])


def test_from_asu_collapses_members_of_one_orbit() -> None:
    asu, bonds, sites, _ = _case("nacl")
    assert len(bonds) == 1
    tables = _tables(225)
    i, j, k, *n = bonds[0]
    g = (k - 1, n[0], n[1], n[2])
    h1, h2 = tables.letter(sites[0][1]).H[5], tables.letter(sites[1][1]).H[7]
    other = tables.mul(tables.mul(h1, g), h2)
    back = tables.inv(other)
    assert other != g
    members = [bonds[0], (i, j, other[0] + 1, *other[1:]), (j, i, back[0] + 1, *back[1:])]
    value = Protonetstructure.from_asu(asu, members)
    assert len(value.net) == 1 and value == _value("nacl")


def test_from_asu_orders_sites_and_keeps_the_input_unrelabelled() -> None:
    value = _value("k2agsbs4-frame")
    assert [o.species.name for o in value.occupations] == ["S"] * 4 + ["K"] * 3 + ["Ag"] * 2 + ["Sb"]
    assert [o.wyckoff for o in value.occupations] == list("ccccabcabc")
    assert isinstance(value.protonettype, Protonettype)


def test_b_only_periodic_net_does_not_depend_on_the_cell() -> None:
    # P2_1/c: a helix on the 2_1 axis at (0,y,1/4) (op 3) and a dimer across the origin centre (op 2).
    # In the cell x' = x + 2z the helix row becomes op 3 plus the translation (1,0,0); the net is the same.
    occupations = [("e", "A"), ("e", "B")]
    base = Protonettype(14, occupations, [(1, 1, 3, 0, 0, 0), (2, 2, 2, 0, 0, 0)], ("A2", None))
    moved = Protonettype(14, occupations, [(1, 1, 3, 1, 0, 0), (2, 2, 2, 0, 0, 0)], ("A2", None))
    assert base == moved


def test_from_asu_re_describes_a_left_handed_cell_in_a_sohncke_group() -> None:
    # P4_2 32, H on i and P on b, re-described by an improper normalizer action: the same crystal in a
    # left-handed cell, whose letters and bonds read the mirror image; from_asu re-describes it
    # right-handed with its bonds (follow-up 4, section 2.6) instead of refusing it
    from httk.atomistic.symmetry.canonical_classification import _actions
    from httk.atomistic.symmetry.canonical_protostructure import _apply_action

    species = [Species("H", ("H",), (1,)), Species("P", ("P",), (1,))]
    asu = ASUStructure(
        Cell([6, 6, 6, 90, 90, 90]), 208, [WyckoffSite("i", (F(1, 7),), "H"), WyckoffSite("b", (), "P")], species
    )
    assert str(Protonetstructure.from_asu(asu, [(1, 2, 1, 0, 0, 0)]).label) == "A3B_cP16_208_i_b__2I~1:H-P"
    improper = next(action for action in _actions(asu.spacegroup) if action.determinant() < 0)
    left = _apply_action(asu, improper)
    assert left.cell.basis.det().sign() < 0
    assert str(Protonetstructure.from_asu(left, [(1, 2, 1, 0, 0, 0)]).label) == "A3B_cP16_208_i_b__2I~1:H-P"
    # outside the Sohncke groups a left-handed cell is accepted: an improper operation keeps the crystal
    rocksalt = ASUStructure(
        Cell(((-5, 0, 0), (0, 5, 0), (0, 0, 5))),
        225,
        [WyckoffSite("a", (), "Na"), WyckoffSite("b", (), "Cl")],
        [Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))],
    )
    assert str(Protonetstructure.from_asu(rocksalt, [(1, 2, 1, 0, 0, 0)]).label).startswith("AB_cF8_225_")
