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
from _protonet_support import handmade, lattice_from_parameters, load_structure, raw_rows, scramble
from httk.core import FracVector
from test_protonet_canonical import NON_CLOSED, QUARTZ, QUARTZ_CELL, SI_O, _enantiomorph

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
    "1~5.6.7.7@4.8.10_2~5.6@4.7@4.7@2@556.8@1@556.10@1@556_3~5.6.7@1@556.9.10@4@455_4~5.7.7@3@445.9@4@454.10@4@455"
)
CUBIC = (5.64, 5.64, 5.64, 90, 90, 90)
NACL = {frozenset(("Na", "Cl")): 3.0}
ELEMENT = {"A": "C", "B": "N"}  # the made-up P2_1 classes, named A and B in the label suffix

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
    "k2agsbs4-frame": "A4B2CD_oP32_34_4c_abc_ab_c__1~8.10_2~8.10_3~9.10_4~9.10@3:S-K-Ag-Sb",
    "k2agsbs4-full": f"A4B2CD_oP32_34_4c_abc_ab_c__{FULL_NET}:S-K-Ag-Sb",
    "nacl": "AB_cF8_225_a_b__1~2:Cl-Na",
    "nacl-exchanged": "AB_cF8_225_a_b__1~2:Cl-Na",
    "diamond": "A_cF8_227_a__1~1@14@654:C",
    "wurtzite": "AB_hP4_186_b_b__1~2.2@2:S-Zn",
    "pdse2": "A2B_oP12_61_c_a__1~1@2@455.2.2@4@455:Se-Pd",
    "p21-chain": "A2B_mP6_4_2a_a__1~1@2.2.2@2.3_2~3:B-A",
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
    assert str(nt.label) == "A4B2CD_oP32_34_4c_abc_ab_c__1~8.10_2~8.10_3~9.10_4~9.10@3"
    # no classes tie here, so the assigned bare part equals the canonical bare protostructure (R5)
    assert str(canonical_bare_protostructure(ps).label) == "A4B2CD_oP32_34_4c_abc_ab_c:S-K-Ag-Sb"
    assert str(ps.label) == "A4B2CD_oP32_34_4c_abc_ab_c__1~8.10_2~8.10_3~9.10_4~9.10@3:S-K-Ag-Sb"


@pytest.mark.parametrize("name", list(CASES))
def test_levels_nest_as_prefixes(name: str) -> None:
    ps = _canonical(name)
    nt = canonical_protonettype(ps)
    formulatype, bare, label = str(nt.anonymous_formula), str(canonical_bare_prototype(nt).label), str(nt.label)
    assert bare.startswith(formulatype + "_") and label.startswith(bare + "__")
    bare_part, _, rest = str(ps.label).partition("__")
    assert bare_part == bare and f"{bare_part}:{rest.rpartition(':')[2]}" == str(ps.bare_protostructure.label)


def test_k2agsbs4_with_k_s_contacts() -> None:
    assert _net_part(str(_canonical("k2agsbs4-full").label)) == FULL_NET


def test_nacl_species_assignments_agree() -> None:
    assert _value("nacl") != _value("nacl-exchanged")
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
    result = _canonical_net(value.spacegroup.it_number, sites, rows)
    net = [(i, j, g[0] + 1, g[1], g[2], g[3]) for (i, j), g in zip(result.slots, result.codes, strict=True)]
    expected = render_protonet(Spacegroup.standard(result.it_number), result.groups, net)
    assert str(canonical_protonettype(value).label) == expected


@pytest.mark.parametrize("name", list(CASES))
def test_canonicalisation_is_idempotent_and_parses_back(name: str) -> None:
    ps = _canonical(name)
    nt = canonical_protonettype(ps)
    assert canonical_protonetstructure(ps) == ps and canonical_protonettype(nt) == nt
    assert canonical_protonettype(_value(name).protonettype) == nt
    if name != "p21-chain":  # the parser accepts element names only; the made-up chain is named A, B
        assert parse_protonetstructure_label(str(ps.label)) == ps
    assert parse_protonettype_label(str(nt.label)) == nt


# ---------------------------------------------------------------------------------------------
# invariance over equivalent input values
# ---------------------------------------------------------------------------------------------
def _variants(name: str, rng: random.Random, count: int) -> list[Protonetstructure]:
    """Valid, non-canonical values of one structure: shuffled sites, reversed bonds, scrambled cosets."""
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
    # scrambled gauges can need translations outside -5..+4, which no value holds (D4): those draws
    # are skipped, and the skip rate is bounded
    attempts = 0
    while len(out) < count + 2:
        attempts += 1
        assert attempts <= 10 * count
        sites2, rows2 = scramble(tables, sites, rows, rng, change_description=it not in NON_CLOSED)
        spec = [
            (
                cls,
                letter,
                tuple(F(13 + n, 97) for n in range(Spacegroup.standard(it).wyckoff_position(letter).free_count)),
            )
            for cls, letter in sites2
        ]
        try:
            out.append(Protonetstructure.from_asu(_asu(it, spec, asu.cell), _bonds(rows2)))
        except ValueError as error:
            assert "translation" in str(error)
    assert attempts - count < 0.9 * attempts
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
    real = protonet_module._canonical_net

    def wrong(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        moved = tuple(dataclasses.replace(r, letters=("c",) * len(r.letters)) for r in result.realisations)
        return dataclasses.replace(result, realisations=moved)

    monkeypatch.setattr(protonet_module, "_canonical_net", wrong)
    with pytest.raises(ValueError, match="bare prototype"):
        canonical_protonettype(_value("nacl"))
    with pytest.raises(ValueError, match="bare prototype"):
        canonical_protonetstructure(_value("nacl"))


def test_unwritable_canonical_translation_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    real = protonet_module._canonical_net

    def long(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        k, _, n2, n3 = result.codes[0]
        return dataclasses.replace(result, codes=((k, 6, n2, n3), *result.codes[1:]))

    monkeypatch.setattr(protonet_module, "_canonical_net", long)
    with pytest.raises(ValueError, match=r"cannot write this canonical net.*-5\.\.\+4"):
        canonical_protonettype(_value("nacl"))
    with pytest.raises(ValueError, match="cannot write this canonical net"):
        canonical_protonetstructure(_value("nacl"))


def test_tiny_timeout_raises_limit_error() -> None:
    _search.cache_clear()
    with pytest.raises(CanonicalizationLimitError):
        canonical_protonetstructure(_value("k2agsbs4-frame"), timeout=1e-9)
    with pytest.raises(CanonicalizationLimitError):
        canonical_protonettype(_value("k2agsbs4-frame"), timeout=1e-9)


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
    with pytest.raises(ValueError, match="translation"):
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
