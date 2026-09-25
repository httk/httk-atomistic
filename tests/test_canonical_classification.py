"""Focused contracts for exact structural-classification canonicalization."""

from fractions import Fraction as F

import pytest
from httk.core import FracVector

from httk.atomistic import Cell, Spacegroup, Species, WyckoffSite
from httk.atomistic.models.bareprotostructure.bareprotostructure import BareProtostructure
from httk.atomistic.models.bareprototype.bareprototype import BarePrototype
from httk.atomistic.models.protostructure.protostructure import Protostructure
from httk.atomistic.models.prototype.prototype import Prototype
from httk.atomistic.models.structure.asu import FundamentalDomainStructure
from httk.atomistic.models.structuretype.anonymize import dummy_species
from httk.atomistic.models.structuretype.fundamental import FundamentalDomainTemplate
from httk.atomistic.symmetry import canonical_classification as classification

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
