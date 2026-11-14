"""Species classes: the class key and name grammar, and classes used alike at every label level."""

import re
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from _protonet_support import generic_cutoffs, raw_rows, reference_point
from httk.core import load

from httk.atomistic import (
    BareProtostructureView,
    BarePrototypeView,
    FormulatypeView,
    PrototypeView,
    Species,
    UnitcellStructure,
    WyckoffSite,
)
from httk.atomistic._loading import _structure_from_payload
from httk.atomistic.cif_structures import _read_cif_for_atomistic
from httk.atomistic.models.protonetstructure.protonetstructure import Protonetstructure
from httk.atomistic.models.protonettype.view import ProtonettypeView
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.prototype.notation import parse_bare_protostructure_label, parse_protonetstructure_label
from httk.atomistic.models.species.classes import _species_class, _species_from_class_name
from httk.atomistic.models.structure.asu import ASUStructure
from httk.atomistic.models.structure.asu_view import ASUStructureView
from httk.atomistic.symmetry.canonical import canonical_asu
from httk.atomistic.symmetry.canonical_classification import (
    canonical_bare_protostructure,
    canonical_bare_prototype,
    canonical_protonetstructure,
    canonical_protonettype,
    canonical_protostructure,
    canonical_prototype,
)
from httk.atomistic.symmetry.canonical_protostructure import (
    canonical_asu_protostructure,
    canonical_asu_protostructure_assignments,
)
from httk.atomistic.symmetry.protonet import _tables
from httk.atomistic.symmetry.subgroups import _standard_input

FIXTURES = Path(__file__).parent / "fixtures"
ALL = sorted(
    str(path.relative_to(FIXTURES))
    for folder in ("disorder", "cod", "structreading")
    for path in (FIXTURES / folder).glob("*.cif")
)
F = Fraction


def _profile(names: list[str], normal: set[str]) -> list[Any]:
    return [name if name in normal else pytest.param(name, marks=pytest.mark.extended) for name in names]


@cache
def _load(name: str) -> Any:
    return load(FIXTURES / name, repair=True)


def _used_species(structure: Any) -> list[Species]:
    by_name = {species.name: species for species in structure.species}
    return [by_name[name] for name in dict.fromkeys(site.species for site in structure.wyckoff_sites)]


def _ordered(species: Species) -> bool:
    return len(species.chemical_symbols) == 1 and species.concentration == (Fraction(1),)


DISORDERED = [name for name in ALL if not all(_ordered(species) for species in _used_species(_load(name)))]


# ---------------------------------------------------------------------------------------------
# the key and the name grammar
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("species", "name"),
    [
        (Species("Ti4+", ("Ti",), (1,), charges=(4,), labels=("Ti1",)), "Ti"),
        (Species("Bi1/Na1", ("Bi", "Na"), (F(1, 2), F(1, 2))), "Bi0.5Na0.5"),
        (Species("Li1", ("Li", "vacancy"), (F(19, 25), F(6, 25))), "Li0.76"),
        (Species("Li1", ("Li",), (F(19, 25),)), "Li0.76"),  # the vacancy is the remainder, stated or not
        (Species("x", ("Li",), (F(1, 3),)), "Li0.333"),  # rounded to three decimals, never "p/q"
        (Species("Ag1/Ga1", ("Ga", "Ag", "vacancy"), (F(561, 1000), F(317, 1000), F(61, 500))), "Ag0.317Ga0.561"),
        (Species("O1", ("O",), (1,), attached=("H",), nattached=(2,), charges=(-2,)), "OH2"),
        (Species("N1", ("N",), (1,), attached=("H",), nattached=(4,)), "NH4"),
        (Species("O9", ("O",), (1,), attached=("H",), nattached=(1,)), "OH1"),
        (Species("O9", ("O",), (1,), attached=("H",), nattached=(0,)), "O"),  # no attached atoms
        (Species("m", ("Na", "K"), (F(1, 2), F(1, 2)), attached=("H", "C"), nattached=(3, 1)), "K0.5Na0.5C1H3"),
        (Species("Fe", ("Fe", "Fe"), (F(1, 2), F(1, 2)), charges=(2, 3)), "Fe"),  # charges dropped, then summed
        (Species("H", ("H",), (1,), mass=(2.014,)), "H"),  # an isotope is its element (S1)
        (Species("O", ("O", "vacancy"), (F(303, 10000), F(9697, 10000))), "O0.03"),
        (Species("Cd1/Pb1", ("Cd", "Pb"), (F(1, 13), F(12, 13))), "Cd0.077Pb0.923"),
        (Species("F", ("F", "O"), (F(83335, 100004), F(16669, 100004))), "F0.833O0.167"),
        (Species("x", ("Li",), (F(2345, 10000),)), "Li0.235"),  # half up, not half even
        (Species("x", ("Li",), (F(5, 10000),)), "Li0.001"),
        (Species("x", ("Li", "Na"), (F(4, 10000), F(1, 2))), "Na0.5"),  # rounds to 0: dropped
        (Species("x", ("Fe", "vacancy"), (F(9996, 10000), F(4, 10000))), "Fe"),  # rounds to 1: the element
        (
            Species("Fe", ("Fe", "Fe"), (F(5002, 10000), F(5002, 10000)), charges=(2, 3)),
            "Fe",
        ),  # 1.0004 is rounding noise
        (Species("x", ("Fe", "Ni"), (F(9996, 10000), F(1, 100))), "Fe1.0Ni0.01"),
        (Species("x", ("Na", "K"), (F(4995, 10000), F(5005, 10000))), "K0.501Na0.5"),  # sum over 1: no vacancy
    ],
)
def test_class_names_and_round_trip(species: Species, name: str) -> None:
    reduced = _species_class(species)
    assert reduced.name == name
    assert _species_from_class_name(name) == reduced == _species_class(reduced)
    assert reduced.charges is None and reduced.labels is None and reduced.mass is None


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Na0.50",
        "Na1/2",
        "Na2/4",
        "O1",
        "H2",
        "OH0",
        "Na0.5Bi0.5",
        "Bi0.5Bi0.5",
        "OH1C1",
        "Xx",
        "X",
        "na",
        "Na-Cl",
        "Na0.5:Cl",
        "Na_Cl",
        "0.5Na",
        "Na.5",
        "Na0.5H",
        "Li1/0",
        "Li0/0",
        "Fe0.0",
        "Fe0.5Ni0.0",
        "Fe2.0",
        "Fe1/0",
        "Li1/3",
        "Cd1/13Pb12/13",
        "Li0.3333",
        "Na0.5000",
        "Na0.0004",
        "Fe1.0",
        "Fe0.0005",
    ],
)
def test_class_name_parser_rejects_non_canonical_names(name: str) -> None:
    with pytest.raises(ValueError):
        _species_from_class_name(name)


def test_unknown_and_vacancy_only_species() -> None:
    dummy = Species("A", ("X",), (1,), labels=("A",))
    assert _species_class(dummy) == dummy  # a dummy or unknown species is its own class
    with pytest.raises(ValueError, match="no constituent other than vacancy"):
        _species_class(Species("empty", ("vacancy",), (1,)))
    with pytest.raises(ValueError, match="summing to 3/2"):
        _species_class(Species("Fe", ("Fe", "Fe"), (F(3, 4), F(3, 4)), charges=(2, 3)))
    with pytest.raises(ValueError, match="summing to 1001/1000"):
        _species_class(Species("Fe", ("Fe", "Fe"), (F(5003, 10000), F(5003, 10000)), charges=(2, 3)))
    with pytest.raises(ValueError, match="no constituent other than vacancy"):
        _species_class(Species("trace", ("Li", "vacancy"), (F(4, 10000), F(9996, 10000))))
    # each constituent at most one, their sum not limited
    assert _species_from_class_name("K0.6Na0.6").concentration == (F(3, 5), F(3, 5))
    rounded = _species_class(Species("x", ("Na", "K"), (F(4995, 10000), F(5005, 10000))))
    assert rounded.chemical_symbols == ("K", "Na") and rounded.concentration == (F(501, 1000), F(1, 2))


def test_concentrations_equal_to_three_decimals_form_one_class() -> None:
    refined = _species_class(Species("a", ("Ga", "B"), (F(7002, 10000), F(2998, 10000))))
    assert refined == _species_class(Species("b", ("B", "Ga"), (F(3, 10), F(7, 10))))
    assert refined.name == "B0.3Ga0.7"
    assert _species_class(Species("c", ("B", "Ga"), (F(2821, 10000), F(7179, 10000)))).name == "B0.282Ga0.718"
    assert _species_class(Species("Fe", ("Fe", "vacancy"), (F(9996, 10000), F(4, 10000)))) == _species_class(
        Species("Fe", ("Fe",), (1,))
    )


def test_class_names_are_safe_inside_labels() -> None:
    mixed = _species_class(Species("m", ("Na", "K"), (F(1, 2), F(1, 2)), attached=("H", "C"), nattached=(3, 1)))
    third = _species_class(Species("t", ("Li",), (F(1, 3),)))
    long = _species_class(Species("f", ("F", "O"), (F(83335, 100004), F(16669, 100004))))
    for species in (mixed, third, long):
        assert re.fullmatch(r"[A-Za-z0-9.]+", species.name)
    # names with "." after the ":" of a protonet label, whose net uses "." "@" "~" "_"
    text = f"A2B_mP6_4_2a_a__1~1@2.2.2@2.3_2~3:{mixed.name}-{third.name}"
    value = parse_protonetstructure_label(text)
    assert str(value.label) == text
    assert [occupation.species for occupation in value.occupations] == [mixed, mixed, third]
    bare = parse_bare_protostructure_label(f"AB_cF8_225_a_b:{long.name}-{third.name}")
    assert {occupation.species for occupation in bare.occupations} == {long, third}


@pytest.mark.parametrize("name", _profile(ALL, {"disorder/100.cif", "disorder/133.cif", "cod/3000362.cif"}))
def test_every_fixture_species_round_trips(name: str) -> None:
    for species in _load(name).species:
        if "X" in species.chemical_symbols or set(species.chemical_symbols) == {"vacancy"}:
            continue
        reduced = _species_class(species)
        assert "/" not in reduced.name
        assert _species_from_class_name(reduced.name) == reduced
        assert reduced.name == _species_class(_species_from_class_name(reduced.name)).name


# ---------------------------------------------------------------------------------------------
# every level in terms of classes
# ---------------------------------------------------------------------------------------------
def _standard(structure: Any) -> ASUStructure:
    standard = _standard_input(structure)
    sites = [WyckoffSite(site.wyckoff, site.free_params, site.species) for site in standard.wyckoff_sites]
    return ASUStructure(standard.cell, standard.spacegroup, sites, standard.species)


def _net(structure: Any, *, bonded: bool) -> Protonetstructure:
    """The value from the standard ASU, bonded by a generic cutoff over the classes, or with no bonds."""
    asu = _standard(structure)
    bonds = []
    if bonded:
        by_name = {species.name: species for species in asu.species}
        it = asu.spacegroup.it_number
        tables = _tables(it)
        sites = [(_species_class(by_name[site.species]).name, site.wyckoff) for site in asu.wyckoff_sites]
        refs = [reference_point(it, site.wyckoff, site.free_params) for site in asu.wyckoff_sites]
        lattice = [[float(x) for x in row] for row in asu.lattice_vectors]
        rows, _ = raw_rows(tables, sites, refs, lattice, generic_cutoffs(tables, sites, refs, lattice))
        bonds = [(i + 1, j + 1, g[0] + 1, g[1], g[2], g[3]) for i, j, g in rows]
    return Protonetstructure.from_asu(asu, bonds)


def _levels(structure: Any) -> dict[str, str]:
    net = _net(structure, bonded=True)
    return {
        "formulatype": str(FormulatypeView(structure)),
        "bare prototype": str(canonical_bare_prototype(structure).label),
        "bare protostructure": str(canonical_bare_protostructure(structure).label),
        "refined": str(canonical_protostructure(Protostructure(representative=canonical_asu(structure))).label),
        "protonettype": str(canonical_protonettype(net).label),
        "protonetstructure": str(canonical_protonetstructure(net).label),
    }


def _rewrite_atom_sites(text: str, column: str, rewrite: Any) -> str:
    """Rewrite one column of the ``_atom_site_`` loop of a CIF text, row by row."""
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.strip() == "_atom_site_label")
    header = start
    while lines[header].strip().startswith("_atom_site_"):
        header += 1
    index = [line.strip() for line in lines[start:header]].index(column)
    row = header
    while row < len(lines) and lines[row].strip() and not lines[row].strip().startswith(("loop_", "_")):
        tokens = lines[row].split()
        tokens[index] = rewrite(row - header, tokens[index])
        lines[row] = " ".join(tokens) + "\n"
        row += 1
    return "".join(lines)


def _from_text(text: str) -> Any:
    return _structure_from_payload(_read_cif_for_atomistic(text.splitlines(keepends=True), repair=True))


@pytest.mark.parametrize("name", ["disorder/100.cif", "disorder/52.cif"])
def test_cif_site_labels_do_not_change_any_label(name: str) -> None:
    text = (FIXTURES / name).read_text()
    relabelled = _rewrite_atom_sites(
        text, "_atom_site_label", lambda row, label: re.match(r"[A-Za-z]+", label).group(0) + str(101 + row)
    )
    original, copy = _from_text(text), _from_text(relabelled)
    assert original == _load(name)  # the in-memory reading is the fixture's
    assert {species.name for species in copy.species} != {species.name for species in original.species}
    assert _levels(copy) == _levels(original)


def test_ago_without_oxidation_states_has_the_same_labels() -> None:
    text = (FIXTURES / "cod/1509688.cif").read_text()
    plain = _from_text(
        _rewrite_atom_sites(text, "_atom_site_type_symbol", lambda row, symbol: symbol.rstrip("+-0123456789"))
    )
    original = _from_text(text)
    assert all(species.charges is None for species in plain.species)
    assert any(species.charges is not None for species in original.species)
    levels = _levels(original)
    assert _levels(plain) == levels
    # merged: Ag(I) and Ag(III) are one class
    assert levels["formulatype"] == "AB"
    assert levels["bare protostructure"] == "AB_mP8_14_ab_e:Ag-O"


@pytest.mark.parametrize("name", _profile(DISORDERED, {"disorder/100.cif", "disorder/133.cif", "disorder/196.cif"}))
def test_disordered_fixtures_give_anonymous_levels_directly(name: str) -> None:
    structure = _load(name)
    assigned = canonical_bare_protostructure(structure)
    direct = canonical_bare_prototype(structure)
    assert direct == canonical_bare_prototype(BarePrototypeView(assigned))
    assert BarePrototypeView(structure).unview() == BarePrototypeView(BareProtostructureView(structure)).unview()
    assert parse_bare_protostructure_label(str(assigned.label)) == assigned
    assert "/" not in str(assigned.label)  # ordered fixtures name their classes by symbols alone
    net = _net(structure, bonded=False)
    protonettype = canonical_protonettype(net)
    protonetstructure = canonical_protonetstructure(net)
    assert protonettype == ProtonettypeView(protonetstructure).unview()
    assert str(protonettype.label).partition("__")[0] == str(direct.label)
    assert parse_protonetstructure_label(str(protonetstructure.label)) == protonetstructure
    assert "/" not in str(protonetstructure.label)


# ---------------------------------------------------------------------------------------------
# every entry point classifies the structure of species classes
# ---------------------------------------------------------------------------------------------
def _cscl(first: Species, second: Species) -> UnitcellStructure:
    return UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]], [[0, 0, 0], [F(1, 2)] * 3], [first, second], [first.name, second.name]
    )


@pytest.mark.parametrize(
    ("first", "second", "name"),
    [
        (Species("Fe2+", ("Fe",), (1,), charges=(2,)), Species("Fe3+", ("Fe",), (1,), charges=(3,)), "Fe"),
        (Species("Na1", ("Na",), (1,)), Species("Na2", ("Na",), (1,)), "Na"),
    ],
    ids=["charge-ordered", "split-names"],
)
def test_every_label_path_classifies_the_class_structure(first: Species, second: Species, name: str) -> None:
    # merged: two names of one chemistry on the CsCl sites are one class, so the class structure is bcc
    structure = _cscl(first, second)
    label = f"A_cI2_229_a:{name}"
    asu = canonical_asu_protostructure(structure)
    assert str(FormulatypeView(structure)) == "A"
    assert str(BareProtostructureView(structure).label) == label
    assert str(canonical_bare_protostructure(structure).label) == label
    assert str(structure.canonical_bare_protostructure().label) == label
    assert (
        str(canonical_bare_prototype(structure).label)
        == str(structure.canonical_bare_prototype().label)
        == label.partition(":")[0]
    )
    assert str(BareProtostructureView(asu).label) == label
    assert str(canonical_protostructure(Protostructure(representative=asu)).label) == label
    assert all(
        BareProtostructureView(member).spacegroup.it_number == 229
        for member in canonical_asu_protostructure_assignments(structure)
    )
    # the structure canonicalizer keeps the source species and their own (primitive cubic) symmetry
    source = canonical_asu(structure)
    assert source.spacegroup.it_number == 221
    assert {species.name for species in source.species} == {first.name, second.name}


@pytest.mark.parametrize("name", ["disorder/149.cif", "disorder/81.cif"])
def test_canonical_asu_chooses_the_description_by_class_names(name: str) -> None:
    # renamed: tied classes are told apart by their class names, as in the bare canonicalizer
    structure = _load(name)
    assert str(BareProtostructureView(canonical_asu(structure)).label) == str(
        canonical_bare_protostructure(structure).label
    )


@pytest.mark.parametrize("name", ["cod/1509688.cif", "disorder/66.cif"])
def test_a_declared_representative_is_stored_in_classes_and_erases(name: str) -> None:
    value = Protostructure(representative=canonical_asu(_load(name)))
    representative = value.representative
    assert representative is not None
    assert all(_species_class(species) == species for species in representative.species)
    assert {site.species for site in representative.wyckoff_sites} == {
        species.name for species in representative.species
    }
    canonical = canonical_protostructure(value)
    assert canonical_prototype(PrototypeView(value)) == canonical_prototype(PrototypeView(canonical))
    assert str(PrototypeView(value).label) == str(canonical_bare_prototype(BareProtostructureView(value)).label)


def test_molecular_structures_have_anonymous_bare_prototypes() -> None:
    # the anonymous erasure drops molecular metadata, as Structuretype does; the assigned level rejects it
    cell = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
    caesium = Species("Cs", ("Cs",), (1,))
    asu = ASUStructure(cell, 221, (WyckoffSite("a", (), "Cs", representative=(0, 0, 0)),), (caesium,), molecular=True)
    assert str(BarePrototypeView(asu).label) == "A_cP1_221_a"
    with pytest.raises(ValueError, match="molecular"):
        BareProtostructureView(asu).unview()
    sodium, chlorine = Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,), charges=(-1,))
    unitcell = UnitcellStructure(cell, [[0, 0, 0], [F(1, 2)] * 3], [sodium, chlorine], ["Na", "Cl"], molecular=True)
    label = "AB_cP2_221_a_b"
    assert str(BarePrototypeView(unitcell).label) == str(canonical_bare_prototype(unitcell).label) == label
    assert str(unitcell.canonical_bare_prototype().label) == label
    with pytest.raises(ValueError, match="molecular"):
        unitcell.canonical_bare_protostructure()


def test_a_declared_asu_whose_species_share_a_class_is_classified_as_the_class_structure() -> None:
    # merged: the declared P m -3 m describes Fe2+ and Fe3+; their one class is bcc
    fe2, fe3 = Species("Fe2+", ("Fe",), (1,), charges=(2,)), Species("Fe3+", ("Fe",), (1,), charges=(3,))
    cell = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    sites = (WyckoffSite("a", (), "Fe2+"), WyckoffSite("b", (), "Fe3+"))
    asu = ASUStructure(cell, 221, sites, (fe2, fe3))
    structure = _cscl(fe2, fe3)
    label = "A_cI2_229_a:Fe"
    for source in (asu, ASUStructureView(structure), canonical_asu(structure)):
        assert source.spacegroup.it_number == 221
        assert str(BareProtostructureView(source).label) == label
        assert str(canonical_bare_protostructure(source).label) == label
        assert str(BarePrototypeView(source).label) == str(canonical_bare_prototype(source).label) == "A_cI2_229_a"
        assert str(BareProtostructureView(canonical_asu_protostructure(source)).label) == label
        assert all(member.spacegroup.it_number == 229 for member in canonical_asu_protostructure_assignments(source))
        refined = Protostructure(representative=source)
        assert str(refined.label) == str(canonical_protostructure(refined).label) == label
        assert str(PrototypeView(refined).label) == "A_cI2_229_a"
    assert str(asu.canonical_bare_protostructure().label) == label
    assert str(FormulatypeView(asu)) == "A"
    # the bonds are re-expressed in the class symmetry when it keeps them (follow-up 4, section 2.6)
    assert str(Protonetstructure.from_asu(asu, []).label) == "A_cI2_229_a__:Fe"
    # bcc: eight bonds in a cube, written since DD8 is the eight-bond default
    assert str(Protonetstructure.from_asu(asu, [(1, 2, 1, 0, 0, 0)]).label) == "A_cI2_229_a__1CU8~1@10:Fe"
    with pytest.raises(ValueError, match="not invariant under the recognized symmetry IT 229"):
        Protonetstructure.from_asu(asu, [(1, 1, 1, 1, 0, 0)])
    # a merge that keeps the space group and cell (AgO) is accepted
    ago = _standard(_load("cod/1509688.cif"))
    # (from_asu keeps the declared description, Ag on a and d)
    assert str(Protonetstructure.from_asu(ago, []).label) == "AB_mP8_14_ad_e__:Ag-O"
    # with names that are their own classes the declared symmetry stays, as before (P1 CsCl)
    p1 = ASUStructure(
        cell,
        1,
        (WyckoffSite("a", (0, 0, 0), "Cs"), WyckoffSite("a", (F(1, 2),) * 3, "Cl")),
        (Species("Cs", ("Cs",), (1,)), Species("Cl", ("Cl",), (1,))),
    )
    assert str(canonical_bare_protostructure(p1).label) == "AB_aP2_1_a_a:Cl-Cs"


def test_structure_canonicalization_keeps_a_vacancy_only_species() -> None:
    # vacancy-only: no class, so the labels raise, but the structure canonicalizer works as before
    caesium, empty = Species("Cs", ("Cs",), (1,)), Species("V", ("vacancy",), (1,))
    structure = _cscl(caesium, empty)
    assert canonical_asu(structure).spacegroup.it_number == 221
    with pytest.raises(ValueError, match="no constituent other than vacancy"):
        canonical_bare_protostructure(structure)
