"""Focused contracts for exact structural-classification canonicalization."""

import random
from fractions import Fraction as F
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any

import httk.core
import pytest
from httk.core import FracVector

from httk.atomistic import Cell, Spacegroup, Species, WyckoffSite
from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure
from httk.atomistic.models.bareprotostructure.view import BareProtostructureView
from httk.atomistic.models.bareprototype.bareprototype import BarePrototype
from httk.atomistic.models.bareprototype.view import BarePrototypeView
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.prototype.prototype import Prototype
from httk.atomistic.models.prototype.view import PrototypeView
from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure
from httk.atomistic.models.structure.comparison import same_crystal
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.models.structuretype.anonymize import dummy_species
from httk.atomistic.models.structuretype.fundamental import FundamentalDomainTemplate
from httk.atomistic.symmetry import canonical_classification as classification
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.canonical import canonical_asu, canonicalize
from httk.atomistic.symmetry.canonical_protostructure import (
    _ac_reduced,
    _apply_action,
    _discrete_candidates,
    _kernel_residues,
    _mirrored,
    _sohncke,
    _with_basis,
    canonical_asu_protostructure_assignments,
)
from httk.atomistic.symmetry.lift import _canonical_sites, _site_key, _translation_normal_form
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.subgroups import _standard_input

CELL = Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7)))
CUBIC_CELL = Cell(((6, 0, 0), (0, 6, 0), (0, 0, 6)))


def _species(name: str) -> Species:
    return Species(name, (name,), (1,))


def _assigned() -> FundamentalDomainStructure:
    na, oxygen = _species("Na"), _species("O")
    return FundamentalDomainStructure(
        CELL,
        47,
        (WyckoffSite("a", (), "Na"), WyckoffSite("c", (), "O")),
        (na, oxygen),
    )


def _template(swapped: bool = False, *, unused: bool = False) -> FundamentalDomainTemplate:
    labels = ("B", "A") if swapped else ("A", "B")
    species = tuple(dummy_species(label) for label in labels)
    if unused:
        return FundamentalDomainTemplate(CELL, 47, (WyckoffSite("a", (), "A"),), species)
    return FundamentalDomainTemplate(
        CELL,
        47,
        (WyckoffSite("a", (), labels[0]), WyckoffSite("c", (), labels[1])),
        species,
    )


def test_bare_values_are_idempotent_and_carry_species_classes() -> None:
    decorated = Species("Na", ("Na",), (1,), charges=(F(1),), labels=("sodium",))
    assigned = BareProtostructure(47, [("a", decorated), ("c", _species("O"))])
    prototype = BarePrototype(47, [("a", "A"), ("c", "B")])

    assigned_result = classification.canonical_bare_protostructure(assigned)
    prototype_result = classification.canonical_bare_prototype(prototype)

    assert classification.canonical_bare_protostructure(assigned_result) == assigned_result
    assert classification.canonical_bare_prototype(prototype_result) == prototype_result
    # dropped charges: an occupation carries the species class, without charges and labels
    species = tuple(occupation.species for occupation in assigned_result.occupations)
    assert _species("Na") in species
    assert decorated not in species


def test_all_vendored_normalizer_actions_share_the_same_bare_minimum() -> None:
    source = BarePrototype(47, [("a", "A"), ("c", "B")])
    expected = classification.canonical_bare_prototype(source)
    group = Spacegroup.standard(47)
    for action in classification._actions(group):
        image = BarePrototype(group, classification._mapped_occupations(source, action, group))
        assert classification.canonical_bare_prototype(image) == expected


@pytest.mark.parametrize(
    ("x", "y", "label"),
    [
        ("a", "b", "AB_mP4_14_a_b"),
        ("a", "d", "AB_mP4_14_a_b"),
        ("b", "c", "AB_mP4_14_a_b"),
        ("a", "c", "AB_mP4_14_a_c"),
    ],
)
def test_p21c_labels_do_not_depend_on_the_cell(x: str, y: str, label: str) -> None:
    # task 4.1: "X on a, Y on d" is "X on a, Y on b" in another valid cell of P2_1/c
    assert str(classification.canonical_bare_prototype(BarePrototype(14, [(x, "A"), (y, "B")])).label) == label


def test_chirality_uses_the_partner_group_for_bare_values() -> None:
    source = BarePrototype(213, [("c", "A")])
    result = classification.canonical_bare_prototype(source, preserve_chirality=False)

    assert result.spacegroup.it_number == 212
    assert classification.canonical_bare_prototype(result, preserve_chirality=False) == result


def test_refined_values_preserve_discriminators_and_rebuild_matching_bases() -> None:
    assigned = Protostructure(representative=_assigned(), discriminator="001")
    anonymous = Prototype(representative=_template(), discriminator="001")

    assigned_result = classification.canonical_protostructure(assigned)
    anonymous_result = classification.canonical_prototype(anonymous)

    assert assigned_result.discriminator == anonymous_result.discriminator == "001"
    assert Protostructure(representative=assigned_result.representative).occupations == assigned_result.occupations
    assert Prototype(representative=anonymous_result.representative).occupations == anonymous_result.occupations
    assert classification.canonical_protostructure(assigned_result) == assigned_result
    assert classification.canonical_prototype(anonymous_result) == anonymous_result


def test_discriminator_only_uses_the_bare_convention() -> None:
    assigned = Protostructure(47, [("a", "Na"), ("c", "O")], discriminator="001")
    anonymous = Prototype(47, [("a", "A"), ("c", "B")], discriminator="001")

    assigned_result = classification.canonical_protostructure(assigned)
    anonymous_result = classification.canonical_prototype(anonymous)
    assert assigned_result.discriminator == anonymous_result.discriminator == "001"
    assert assigned_result.representative is anonymous_result.representative is None
    assigned_bare = classification.canonical_bare_protostructure(assigned)
    anonymous_bare = classification.canonical_bare_prototype(anonymous)
    assert (assigned_result.spacegroup, assigned_result.occupations) == (
        assigned_bare.spacegroup,
        assigned_bare.occupations,
    )
    assert (anonymous_result.spacegroup, anonymous_result.occupations) == (
        anonymous_bare.spacegroup,
        anonymous_bare.occupations,
    )


def test_template_dummy_label_swaps_and_unused_definitions_are_canonical() -> None:
    first = classification.canonical_prototype(Prototype(representative=_template()))
    second = classification.canonical_prototype(Prototype(representative=_template(swapped=True)))
    unused = classification.canonical_prototype(Prototype(representative=_template(unused=True)))

    assert first == second
    assert tuple(species.name for species in unused.representative.species) == ("A", "B")


def test_retained_representative_must_be_exact_or_is_dropped() -> None:
    sodium = _species("Na")
    exact = FundamentalDomainStructure(CELL, 47, (WyckoffSite("a", (), "Na", FracVector((0, 0, 0))),), (sodium,))
    imprecise = FundamentalDomainStructure(
        CELL,
        47,
        (WyckoffSite("a", (), "Na", FracVector((F(1, 100), 0, 0))),),
        (sodium,),
        coordinate_precision=F(1, 100),
    )

    assert all(
        site.representative is None
        for site in classification.canonical_protostructure(
            Protostructure(representative=exact)
        ).representative.wyckoff_sites
    )
    with pytest.raises(ValueError, match="imprecise retained"):
        classification.canonical_protostructure(Protostructure(representative=imprecise))


FIXTURES = Path(__file__).parent / "fixtures"
LOW_SYMMETRY = tuple(range(2, 16))
MONOCLINIC_CELL = Cell(((5, 0, 0), (0, 6, 0), (-1, 0, 7)))
TRICLINIC_CELL = Cell(((5, 0, 0), (1, 6, 0), (-1, 2, 7)))
GENERAL = (F(1, 7), F(2, 9), F(3, 11))


def _synthetic(it: int, sites: list[tuple[str, str]]) -> ASUStructure:
    group = Spacegroup.standard(it)
    return ASUStructure(
        TRICLINIC_CELL if it == 2 else MONOCLINIC_CELL,
        it,
        [WyckoffSite(letter, GENERAL[: group.wyckoff_position(letter).free_count], name) for name, letter in sites],
        [_species(name) for name in dict.fromkeys(name for name, _ in sites)],
    )


@cache
def _invariant_case(name: str) -> ASUStructure:
    if name.endswith(".cif"):
        loaded = httk.core.load(str(FIXTURES / name))
        sites = [WyckoffSite(site.wyckoff, site.free_params, site.species) for site in loaded.wyckoff_sites]
        return ASUStructure(loaded.cell, loaded.spacegroup, sites, loaded.species)
    it, sites = SYNTHETIC[name]
    return _synthetic(it, sites)


SYNTHETIC: dict[str, tuple[int, list[tuple[str, str]]]] = {
    "p21c-a-e": (14, [("Na", "a"), ("O", "e")]),  # design note §7, probe 1
    "p21c-a-d": (14, [("Na", "a"), ("Cl", "d")]),  # design note §7, probe 2
    "p21c-b-c-e": (14, [("Na", "b"), ("Cl", "c"), ("O", "e")]),
    "p-1-a-d": (2, [("Na", "a"), ("Cl", "d")]),
    "p-1-b-g-i": (2, [("Na", "b"), ("Cl", "g"), ("O", "i")]),
    "p2m-a-d": (10, [("Na", "a"), ("Cl", "d")]),
    "c2m-a-c": (12, [("Na", "a"), ("Cl", "c")]),
    # the species order differed before the refined key carried species names (refined :Cl-O-Na)
    "c2m-f-b-d": (12, [("Cl", "f"), ("Na", "b"), ("O", "d")]),
    "p-1-a-d-i": (2, [("Na", "a"), ("Cl", "d"), ("O", "i")]),
    "p2-b-d-e": (3, [("Na", "b"), ("Cl", "d"), ("O", "e")]),
}
INVARIANT_CASES = (
    *(f"structreading/{n}.cif" for n in LOW_SYMMETRY),
    "cod/1000008.cif",
    *SYNTHETIC,
)


def _labels(asu: ASUStructure) -> tuple[str, str, str, str]:
    """Return the anonymous refined and bare canonical labels, then the assigned ones."""
    value = Protostructure(representative=asu)
    refined = classification.canonical_protostructure(value)
    bare = classification.canonical_bare_prototype(BarePrototypeView(value).unview())
    assigned = classification.canonical_bare_protostructure(value)
    return str(BarePrototypeView(refined).unview().label), str(bare.label), str(refined.label), str(assigned.label)


@pytest.mark.parametrize("name", INVARIANT_CASES)
def test_refined_label_is_the_canonical_bare_one(name: str) -> None:
    # task 4.1b: the refined terminal reaches the bare canonicalizers' cell changes; this failed
    # for structreading/3.cif (a3e_e_d), p21c-a-d and p21c-b-c-e after 4.1a alone. The assigned
    # labels agree as strings since the refined discrete key carries the species names.
    refined, bare, refined_assigned, bare_assigned = _labels(_invariant_case(name))
    assert refined == bare
    assert refined_assigned == bare_assigned


def test_refined_result_is_unchanged_when_no_cell_change_is_needed() -> None:
    # C2/m: the identity cell change reaches the least key, so the refined output is the one of
    # the tabulated descriptions alone (label, sites and cell metric as before task 4.1b)
    result = classification.canonical_protostructure(
        Protostructure(representative=_invariant_case("structreading/12.cif"))
    )
    assert str(result.label) == "A_mC4_12_i:O"
    assert [(site.wyckoff, tuple(site.free_params.to_fractions())) for site in result.representative.wyckoff_sites] == [
        ("i", (F(3109, 50000), F(14667, 100000)))
    ]
    metric = result.representative.cell.metric().to_floats()
    expected = [
        [18.2468228569, 0.0, -7.717433262698566],
        [0.0, 18.29118931225367, 0.0],
        [-7.717433262698566, 0.0, 20.898429390363873],
    ]
    assert metric == [pytest.approx(row, abs=1e-9) for row in expected]


def _cell_change_generators(it: int) -> list[AffineOperation]:
    """Cell changes that map the group onto itself: elementary matrices (P-1), else ac-plane ones."""
    if it == 2:
        matrices = [
            tuple(tuple(int(r == c) + sign * int((r, c) == (i, j)) for c in range(3)) for r in range(3))
            for i in range(3)
            for j in range(3)
            if i != j
            for sign in (1, -1)
        ]
    else:
        matrices = [
            ((a, 0, b), (0, 1, 0), (c, 0, d))
            for a, b, c, d in product(range(-2, 3), repeat=4)
            if a * d - b * c in (1, -1)
        ]
    group = Spacegroup.standard(it)
    return [
        AffineOperation(matrix)
        for matrix in matrices
        if classification._signature(group, AffineOperation(matrix)) is not None
    ]


def _representative(asu: ASUStructure) -> tuple[Any, ...]:
    """Return the refined representative's exact Gram matrix and sites."""
    representative = classification.canonical_protostructure(Protostructure(representative=asu)).representative
    assert representative is not None
    metric = representative.cell.metric()
    return (
        tuple(metric._element((row, column)) for row in range(3) for column in range(3)),
        tuple((s.species, s.wyckoff, tuple(s.free_params.to_fractions())) for s in representative.wyckoff_sites),
    )


@pytest.mark.extended
@pytest.mark.parametrize("it", LOW_SYMMETRY)
def test_refined_label_is_the_canonical_bare_label_in_every_cell(it: int) -> None:
    # random occupations, each also described in two random other cells: the refined label is the
    # bare one as a string, and the refined value (cell and sites included) does not depend on the cell
    rng = random.Random(f"refined-bare-{it}")
    group = Spacegroup.standard(it)
    generators = _cell_change_generators(it)
    fixed = {w.letter for w in group.wyckoff if w.free_count == 0}
    for _ in range(4):
        sites: list[tuple[str, str]] = []
        for _ in range(rng.randint(2, 4)):
            letter = rng.choice(group.wyckoff).letter
            if not (letter in fixed and any(letter == x for _, x in sites)):
                sites.append((rng.choice(("Na", "Cl", "O")), letter))
        asu = ASUStructure(
            TRICLINIC_CELL if it == 2 else MONOCLINIC_CELL,
            it,
            [
                WyckoffSite(
                    letter,
                    tuple(F(rng.randint(1, 96), 97) for _ in range(group.wyckoff_position(letter).free_count)),
                    name,
                )
                for name, letter in sites
            ],
            [_species(name) for name in dict.fromkeys(name for name, _ in sites)],
        )
        labels = _labels(asu)
        assert labels[0] == labels[1] and labels[2] == labels[3]
        representative = _representative(asu)
        for _ in range(2):
            action = AffineOperation.identity()
            for _ in range(rng.randint(1, 3)):
                action = rng.choice(generators) * action
            assert _labels(_apply_action(asu, action)) == labels
            assert _representative(_apply_action(asu, action)) == representative


def _refined_value(asu: ASUStructure) -> tuple[Any, ...]:
    """Return the refined label, the bare label and the representative's Gram matrix, basis and sites."""
    value = Protostructure(representative=asu)
    refined = classification.canonical_protostructure(value)
    representative = refined.representative
    assert representative is not None and representative.cell.basis.det().sign() > 0
    metric, basis = representative.cell.metric(), representative.cell.basis
    return (
        str(refined.label),
        str(classification.canonical_bare_protostructure(value).label),
        tuple(metric._element((row, column)) for row in range(3) for column in range(3)),
        tuple(basis._element((row, column)) for row in range(3) for column in range(3)),
        tuple((s.species, s.wyckoff, tuple(s.free_params.to_fractions())) for s in representative.wyckoff_sites),
    )


@cache
def _redescriptions(it: int) -> tuple[AffineOperation, ...]:
    """Same-group re-descriptions: cell changes in groups 2 to 15, and the tabulated normalizer actions."""
    changes = tuple(_cell_change_generators(it)) if 2 <= it <= 15 else ()
    return changes + classification._tabulated_actions(Spacegroup.standard(it))


def _assert_cell_independent(asu: ASUStructure, rng: random.Random, cells: int) -> None:
    first = _refined_value(asu)
    assert first[0] == first[1]
    for _ in range(cells):
        action = AffineOperation.identity()
        for _ in range(rng.randint(1, 3)):
            action = rng.choice(_redescriptions(asu.spacegroup.it_number)) * action
        assert _refined_value(_apply_action(asu, action)) == first


@pytest.mark.parametrize("name", SYNTHETIC)
def test_named_synthetic_cases_give_one_refined_value_in_every_cell(name: str) -> None:
    # the refined label is the bare one as a string, and label, cell and sites do not depend on
    # the input cell (design note §7 probes; the P-1, P2 and C2/m cases also differed in species order)
    _assert_cell_independent(_invariant_case(name), random.Random(name), 6)


def _fixture_names() -> list[str]:
    numbered = sorted((FIXTURES / "structreading").glob("*.cif"), key=lambda path: int(path.stem))
    return [f"structreading/{path.name}" for path in numbered] + [
        f"cod/{path.name}" for path in sorted((FIXTURES / "cod").glob("*.cif"))
    ]


def _fixture(name: str) -> ASUStructure:
    try:
        loaded = httk.core.load(str(FIXTURES / name))
    except ValueError:  # an unrecognised declared setting or occupancies needing repair
        loaded = httk.core.load(str(FIXTURES / name), repair=True)
    assert isinstance(loaded, ASUStructure)
    standard = _standard_input(loaded)
    sites = [WyckoffSite(site.wyckoff, site.free_params, site.species) for site in standard.wyckoff_sites]
    return ASUStructure(standard.cell, standard.spacegroup, sites, standard.species)


@pytest.mark.extended
@pytest.mark.parametrize("name", _fixture_names())
def test_fixture_refined_label_is_bare_label_in_random_cells(name: str) -> None:
    _assert_cell_independent(_fixture(name), random.Random(name), 2)


def _enantiomer(asu: ASUStructure) -> ASUStructure:
    """Return the mirror image: the coordinates negated in the same basis."""
    inversion = AffineOperation(FracVector(((-1, 0, 0), (0, -1, 0), (0, 0, -1))), (0, 0, 0))
    return _with_basis(_apply_action(asu, inversion), asu.cell.basis)


@pytest.mark.parametrize(
    "number,labels,identified",
    (
        (208, ("A3B_cP16_208_i_c:H-P", "A3B_cP16_208_i_b:H-P"), "A3B_cP16_208_i_b:H-P"),
        (214, ("A3B2C_cI48_214_f_e_a:Ag-Te-Au", "A3B2C_cI48_214_f_e_b:Ag-Te-Au"), "A3B2C_cI48_214_f_e_a:Ag-Te-Au"),
    ),
)
def test_both_hands_of_an_own_enantiomorph_sohncke_structure(
    number: int, labels: tuple[str, str], identified: str
) -> None:
    # P4_2 32 (b <-> c) and I4_1 32 (a <-> b): the mirror image exchanges letters. The kept hand uses
    # proper descriptions only, so the hands get different labels and right-handed representatives
    # that are the crystals themselves; preserve_chirality=False identifies them
    hands = (_fixture(f"structreading/{number}.cif"), _enantiomer(_fixture(f"structreading/{number}.cif")))
    values = [Protostructure(representative=hand) for hand in hands]
    refined = [classification.canonical_protostructure(value) for value in values]
    assert tuple(str(result.label) for result in refined) == labels
    assert tuple(str(classification.canonical_bare_protostructure(value).label) for value in values) == labels
    for value in values:
        assert str(classification.canonical_prototype(PrototypeView(value).unview()).label) == str(
            classification.canonical_bare_prototype(BarePrototypeView(value).unview()).label
        )
    for result, label in zip(refined, labels, strict=True):
        representative = result.representative
        assert representative is not None and representative.cell.basis.det().sign() > 0
        # the representative read in its own right-handed basis is the hand it came from
        assert (
            str(classification.canonical_bare_protostructure(Protostructure(representative=representative)).label)
            == label
        )
    flipped = [classification.canonical_protostructure(value, preserve_chirality=False) for value in values]
    assert flipped[0] == flipped[1]
    assert str(flipped[0].label) == identified
    assert {
        str(classification.canonical_bare_protostructure(value, preserve_chirality=False).label) for value in values
    } == {identified}
    representative = flipped[0].representative
    assert representative is not None and representative.cell.basis.det().sign() > 0


CHIRALITY_TABLE: dict[str, tuple[Cell, int, list[tuple[str, tuple[F, ...], str]], str, str]] = {
    # follow-up 3 §1: the input, its default label and its preserve_chirality=False label
    "154": (
        Cell([5, 5, 6, 90, 90, 120]),
        154,
        [("a", (F(47, 100),), "Si"), ("c", (F(41, 100), F(27, 100), F(1, 5)), "O")],
        "A2B_hP9_154_c_a:O-Si",
        "A2B_hP9_152_c_a:O-Si",
    ),
    "78": (
        Cell([5, 5, 7, 90, 90, 90]),
        78,
        [("a", (F(1, 7), F(2, 9), F(3, 11)), "Se")],
        "A_tP4_78_a:Se",
        "A_tP4_76_a:Se",
    ),
    "208-b": (
        CUBIC_CELL,
        208,
        [("i", (F(1, 7),), "H"), ("b", (), "P")],
        "A3B_cP16_208_i_b:H-P",
        "A3B_cP16_208_i_b:H-P",
    ),
    "208-c": (
        CUBIC_CELL,
        208,
        [("i", (F(1, 7),), "H"), ("c", (), "P")],
        "A3B_cP16_208_i_c:H-P",
        "A3B_cP16_208_i_b:H-P",
    ),
}


def _labels_of(asu: ASUStructure, preserve_chirality: bool) -> set[str]:
    """Return the bare, refined and (with names) prototype labels of one ASU."""
    value = Protostructure(representative=asu)
    refined = classification.canonical_protostructure(value, preserve_chirality=preserve_chirality)
    bare = classification.canonical_bare_protostructure(value, preserve_chirality=preserve_chirality)
    prototype = classification.canonical_prototype(PrototypeView(value).unview(), preserve_chirality=preserve_chirality)
    names = ":" + str(bare.label).split(":")[1]
    return {str(refined.label), str(bare.label), str(prototype.label) + names}


@pytest.mark.parametrize("name", CHIRALITY_TABLE)
def test_chirality_table(name: str) -> None:
    cell, it, sites, kept, identified = CHIRALITY_TABLE[name]
    asu = ASUStructure(
        cell, it, [WyckoffSite(*site) for site in sites], [_species(n) for n in dict.fromkeys(s[2] for s in sites)]
    )
    assert _labels_of(asu, True) == {kept}
    assert _labels_of(asu, False) == {identified}
    # the mirror image: the other 208 input, or the enantiomorphic partner
    mirrored = {"208-b": "A3B_cP16_208_i_c:H-P", "208-c": "A3B_cP16_208_i_b:H-P"}.get(name, identified)
    assert _labels_of(_mirrored(asu), True) == {mirrored}
    assert _labels_of(_mirrored(asu), False) == {identified}


def _p4232(letter: str, basis: Any = ((6, 0, 0), (0, 6, 0), (0, 0, 6)), **kwargs: Any) -> ASUStructure:
    """P4_2 32 with H on i and P on ``letter``, in ``basis``."""
    sites = [WyckoffSite("i", (F(1, 7),), "H"), WyckoffSite(letter, (), "P")]
    return ASUStructure(Cell(basis), 208, sites, [_species("H"), _species("P")], **kwargs)


def test_handedness_is_that_of_the_standard_basis_after_the_setting_transform() -> None:
    # an identity cell with a det -1 setting transform: the standard basis is left-handed, so P on b
    # describes the mirror image, the P-on-c crystal, at every entry point
    swapped = _p4232("b", transform=SettingTransform(((0, 1, 0), (1, 0, 0), (0, 0, 1))))
    label = "A3B_cP16_208_i_c:H-P"
    assert str(BareProtostructureView(canonicalize(swapped, symmetry="declared")).unview().label) == label
    assert str(classification.canonical_bare_protostructure(swapped).label) == label
    assert str(classification.canonical_bare_protostructure(BareProtostructureView(swapped).unview()).label) == label


def test_left_handed_declared_values_are_read_as_the_crystal() -> None:
    # the P-on-b crystal written in the negated basis is the P-on-c crystal: the recognized bare
    # value, the representative of a refined value and a template all read it so
    left = _p4232("b", ((-6, 0, 0), (0, -6, 0), (0, 0, -6)))
    assert str(classification.canonical_bare_protostructure(BareProtostructureView(left).unview()).label) == (
        "A3B_cP16_208_i_c:H-P"
    )
    representative = Protostructure(representative=left).representative
    assert representative is not None and representative.cell.basis.det().sign() > 0
    template = FundamentalDomainTemplate(
        Cell(((-6, 0, 0), (0, -6, 0), (0, 0, -6))),
        208,
        (WyckoffSite("i", (F(1, 7),), "A"), WyckoffSite("b", (), "B")),
        (dummy_species("A"), dummy_species("B")),
    )
    assert str(classification.canonical_bare_prototype(BarePrototypeView(template).unview()).label) == (
        "A3B_cP16_208_i_c"
    )
    prototype = Prototype(representative=template)
    assert prototype.representative is not None and prototype.representative.cell.basis.det().sign() > 0
    assert str(classification.canonical_bare_prototype(prototype).label) == "A3B_cP16_208_i_c"
    # a right-handed template keeps its own reading
    right = FundamentalDomainTemplate(
        Cell(((6, 0, 0), (0, 6, 0), (0, 0, 6))), 208, template.wyckoff_sites, template.species
    )
    assert str(classification.canonical_bare_prototype(BarePrototypeView(right).unview()).label) == "A3B_cP16_208_i_b"


def test_detect_mode_chooses_species_by_the_bare_key() -> None:
    # structreading/82: the anonymous family orders O-P-Al by geometry; the canonical ASU follows the
    # bare key (O-Al-P) in detect mode as in declared mode, so it is not a member of the family
    source = UnitcellStructureView(httk.core.load(str(FIXTURES / "structreading/82.cif")))
    result = canonical_asu(source)
    family = canonical_asu_protostructure_assignments(source)
    declared = classification.canonical_bare_protostructure(
        Protostructure(representative=_fixture("structreading/82.cif"))
    )
    assert str(BareProtostructureView(result).unview().label) == str(declared.label) == "A4BC_tI12_82_g_a_c:O-Al-P"
    assert [str(BareProtostructureView(member).unview().label) for member in family] == ["A4BC_tI12_82_g_a_c:O-P-Al"]
    assert result not in family


def test_lattice_translated_descriptions_give_the_same_terminal_result() -> None:
    # the terminal evaluates one candidate per (basis change, wrapped operation)
    structure = _invariant_case("p21c-b-c-e")
    for candidate in _discrete_candidates(structure)[:12]:
        shifted = AffineOperation(FracVector.eye((3, 3)), (1, -2, 3)) * candidate.operation
        assert shifted.wrapped() == candidate.operation.wrapped()
        results = [
            _canonical_sites(_translation_normal_form(_apply_action(structure, operation)))
            for operation in (candidate.operation, shifted)
        ]
        assert _site_key(results[0]) == _site_key(results[1])


def _hand_value(asu: ASUStructure) -> tuple[Any, ...]:
    """Return the refined value with ``preserve_chirality=False``: label, Gram, handedness and sites."""
    result = classification.canonical_protostructure(Protostructure(representative=asu), preserve_chirality=False)
    representative = result.representative
    assert representative is not None
    metric = representative.cell.metric()
    return (
        str(result.label),
        tuple(metric._element((row, column)) for row in range(3) for column in range(3)),
        representative.cell.basis.det().sign(),
        tuple((s.species, s.wyckoff, tuple(s.free_params.to_fractions())) for s in representative.wyckoff_sites),
    )


HAND_CASES: dict[str, tuple[Cell, int, list[tuple[str, tuple[F, ...], str]]]] = {
    # the mirror image is anonymously equal to the crystal with the two species classes exchanged, so
    # the hands are related only by an improper map, which the anonymous P1 frame does not enumerate
    "p1-two-species": (
        TRICLINIC_CELL,
        1,
        [("a", (F(1, 7), F(2, 9), F(3, 11)), "Na"), ("a", (F(4, 7), F(5, 9), F(1, 11)), "Cl")],
    ),
    "p2-swap": (
        MONOCLINIC_CELL,
        3,
        [("e", (F(1, 7), F(2, 9), F(3, 11)), "Na"), ("e", (F(1, 7), F(5, 9), F(3, 11)), "Cl")],
    ),
    "p222-swap": (CELL, 16, [("u", (F(1, 7), F(2, 9), F(3, 11)), "Na"), ("u", (F(-1, 7), F(-2, 9), F(-3, 11)), "Cl")]),
}


@pytest.mark.parametrize("name", HAND_CASES)
def test_preserve_chirality_false_identifies_the_hands(name: str) -> None:
    cell, it, sites = HAND_CASES[name]
    asu = ASUStructure(
        cell, it, [WyckoffSite(*site) for site in sites], [_species(n) for n in dict.fromkeys(s[2] for s in sites)]
    )
    assert _hand_value(asu) == _hand_value(_enantiomer(asu))


def _sohncke_fixtures() -> list[str]:
    names = []
    for name in _fixture_names():
        number = int(name.split("/")[1].split(".")[0]) if name.startswith("structreading") else None
        if number is not None and _sohncke(Spacegroup.standard(number)):
            names.append(name)
    # the COD Sohncke fixtures: P1, P2_1, P2_1 2_1 2_1
    return [*names, "cod/1001844.cif", "cod/4326344.cif", "cod/7219102.cif"]


@pytest.mark.extended
@pytest.mark.parametrize("name", _sohncke_fixtures())
def test_preserve_chirality_false_identifies_the_hands_of_every_sohncke_fixture(name: str) -> None:
    # every Sohncke group among the structreading fixtures (numbered by group), including I4_1,
    # I4_122 and F4_132, where the normalizing inversion needs an origin shift; the mirror image is
    # the same fractions in the negated basis (inverting the coordinates through the origin would
    # leave the group there). Bare and refined labels agree, and both representatives are right-handed
    asu = _fixture(name)
    hands = (asu, _mirrored(asu))
    values = [_hand_value(hand) for hand in hands]
    assert values[0] == values[1]
    # the representatives themselves, site order included (sorted, also with tied species
    # assignments: 169, C and N on 4a); the recorded cell precision follows the description path
    first, second = (
        classification.canonical_protostructure(
            Protostructure(representative=hand), preserve_chirality=False
        ).representative
        for hand in hands
    )
    assert first is not None and second is not None
    assert first.wyckoff_sites == second.wyckoff_sites and same_crystal(first, second)
    assert values[0][2] > 0
    assert {
        str(
            classification.canonical_bare_protostructure(
                Protostructure(representative=hand), preserve_chirality=False
            ).label
        )
        for hand in hands
    } == {values[0][0]}


def test_declared_letter_more_general_than_its_point_is_the_documented_exception() -> None:
    # Na declared on the general position i of F-43m at a point of h: the refined terminal demotes
    # the site, the bare canonicalizer keeps the declared letter, so the labels differ
    asu = ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        216,
        [WyckoffSite("i", (F(1, 101), F(27, 101), F(100, 101)), "Na"), WyckoffSite("f", (F(28, 97),), "Cl")],
        [_species("Na"), _species("Cl")],
    )
    value = Protostructure(representative=asu)
    assert str(classification.canonical_protostructure(value).label) == "A2B_cF72_216_h_f:Na-Cl"
    assert str(classification.canonical_bare_protostructure(value).label) == "A4B_cF120_216_i_f:Na-Cl"


def _brute_reduced(
    form: tuple[int, int, int], residues: frozenset[Any], matrices: list[tuple[int, ...]]
) -> tuple[tuple[int, ...], set[Any]]:
    aa, ac, cc = form

    def q(u: tuple[int, int], v: tuple[int, int]) -> int:
        return aa * u[0] * v[0] + ac * (u[0] * v[1] + u[1] * v[0]) + cc * u[1] * v[1]

    best: tuple[int, ...] | None = None
    minimisers: set[Any] = set()
    for a, b, c, d in matrices:
        if ((a % 4, b % 4), (c % 4, d % 4)) not in residues:
            continue
        key = (q((a, b), (a, b)), q((c, d), (c, d)), q((a, b), (c, d)))
        if best is None or key < best:
            best, minimisers = key, {((a, b), (c, d))}
        elif key == best:
            minimisers.add(((a, b), (c, d)))
    assert best is not None
    return best, minimisers


def _unimodular(bound: int) -> list[tuple[int, ...]]:
    return [m for m in product(range(-bound, bound + 1), repeat=4) if m[0] * m[3] - m[1] * m[2] in (1, -1)]


def _check_ac_reduced(forms: list[tuple[int, int, int]], shear: int, bound: int, seed: str) -> None:
    rng = random.Random(seed)
    matrices = _unimodular(bound)
    shears = _unimodular(shear)
    for it in range(3, 16):
        residues = _kernel_residues(Spacegroup.standard(it))
        for a, b, c in forms:
            # the reduced form moved by a random unimodular P: P G P^T
            p00, p01, p10, p11 = rng.choice(shears)
            moved = (
                a * p00 * p00 + 2 * b * p00 * p01 + c * p01 * p01,
                a * p00 * p10 + b * (p00 * p11 + p01 * p10) + c * p01 * p11,
                a * p10 * p10 + 2 * b * p10 * p11 + c * p11 * p11,
            )
            key, changes = _ac_reduced(moved, residues)
            assert (key, set(changes)) == _brute_reduced((moved[0], moved[1], moved[2]), residues, matrices), (
                it,
                moved,
            )


def test_kernel_residues_lift_within_three() -> None:
    # every unimodular residue class modulo 4 has a lift with entries in -3..3, so _kernel_residues
    # decides every class
    assert len({tuple(v % 4 for v in m) for m in _unimodular(3)}) == 96


def test_ac_reduced_matches_brute_force() -> None:
    # random, hexagonal, square and C >> A forms in every group 3 to 15
    forms = [(6, -3, 6), (5, 0, 5), (2, -1, 30), (8, 3, 11), (10, -4, 13)]
    _check_ac_reduced(forms, 1, 7, "ac-reduced")


@pytest.mark.extended
def test_ac_reduced_matches_brute_force_extended() -> None:
    rng = random.Random("ac-reduced-extended")
    forms = [(6, 3, 6), (4, -2, 4), (3, 0, 3), (5, 0, 9), (18, 9, 40), (1, 0, 60)]
    for _ in range(20):
        a = 2 * rng.randint(1, 20)
        forms.append((a, rng.randint(-a // 2, a // 2), a + 2 * rng.randint(0, 20)))
    _check_ac_reduced(forms, 2, 12, "ac-reduced-extended")
