"""Focused contracts for exact structural-classification canonicalization."""

import random
from fractions import Fraction as F
from functools import cache
from itertools import product
from pathlib import Path

import httk.core
import pytest
from httk.core import FracVector

from httk.atomistic import Cell, Spacegroup, Species, WyckoffSite
from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure
from httk.atomistic.models.bareprototype.bareprototype import BarePrototype
from httk.atomistic.models.bareprototype.view import BarePrototypeView
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.prototype.prototype import Prototype
from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure
from httk.atomistic.models.structuretype.anonymize import dummy_species
from httk.atomistic.models.structuretype.fundamental import FundamentalDomainTemplate
from httk.atomistic.symmetry import canonical_classification as classification
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.canonical_protostructure import _apply_action

CELL = Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7)))


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


def test_bare_values_are_idempotent_and_preserve_full_species() -> None:
    decorated = Species("Na", ("Na",), (1,), charges=(F(1),), labels=("sodium",))
    assigned = BareProtostructure(47, [("a", decorated), ("c", _species("O"))])
    prototype = BarePrototype(47, [("a", "A"), ("c", "B")])

    assigned_result = classification.canonical_bare_protostructure(assigned)
    prototype_result = classification.canonical_bare_prototype(prototype)

    assert classification.canonical_bare_protostructure(assigned_result) == assigned_result
    assert classification.canonical_bare_prototype(prototype_result) == prototype_result
    assert decorated in tuple(occupation.species for occupation in assigned_result.occupations)


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
    "p21c-a-d": (14, [("Na", "a"), ("Cl", "d")]),
    "p21c-b-c-e": (14, [("Na", "b"), ("Cl", "c"), ("O", "e")]),
    "p-1-a-d": (2, [("Na", "a"), ("Cl", "d")]),
    "p-1-b-g-i": (2, [("Na", "b"), ("Cl", "g"), ("O", "i")]),
    "p2m-a-d": (10, [("Na", "a"), ("Cl", "d")]),
    "c2m-a-c": (12, [("Na", "a"), ("Cl", "c")]),
    # only the anonymous labels agree: the refined path picks the species order by geometry
    # (refined :Cl-O-Na, bare :Cl-Na-O)
    "c2m-f-b-d": (12, [("Cl", "f"), ("Na", "b"), ("O", "d")]),
}
INVARIANT_CASES = (
    *(f"structreading/{n}.cif" for n in LOW_SYMMETRY),
    "cod/1000008.cif",
    *SYNTHETIC,
)


def _anonymous_labels(asu: ASUStructure) -> tuple[str, str, str]:
    """Return the anonymous refined and bare canonical labels, and the assigned refined label."""
    value = Protostructure(representative=asu)
    refined = classification.canonical_protostructure(value)
    bare = classification.canonical_bare_prototype(BarePrototypeView(value).unview())
    return str(BarePrototypeView(refined).unview().label), str(bare.label), str(refined.label)


@pytest.mark.parametrize("name", INVARIANT_CASES)
def test_refined_anonymous_label_is_the_canonical_bare_one(name: str) -> None:
    # task 4.1b: the refined terminal reaches the bare canonicalizers' cell changes; this failed
    # for structreading/3.cif (a3e_e_d), p21c-a-d and p21c-b-c-e after 4.1a alone. The assigned
    # labels can list species in another order (c2m-f-b-d), so only the anonymous ones are compared.
    refined, bare, _ = _anonymous_labels(_invariant_case(name))
    assert refined == bare


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


@pytest.mark.extended
@pytest.mark.parametrize("it", LOW_SYMMETRY)
def test_refined_label_is_the_canonical_bare_label_in_every_cell(it: int) -> None:
    # random occupations, each also described in two random other cells. The anonymous labels agree;
    # the species assignment of the refined value is chosen by geometry after the anonymous
    # description (also before task 4.1), so only the refined assigned label is compared across cells
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
        labels = _anonymous_labels(asu)
        assert labels[0] == labels[1]
        for _ in range(2):
            action = AffineOperation.identity()
            for _ in range(rng.randint(1, 3)):
                action = rng.choice(generators) * action
            assert _anonymous_labels(_apply_action(asu, action)) == labels
