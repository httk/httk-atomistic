"""Tests for the anonymous unit-cell family."""

from fractions import Fraction

import pytest

from httk.atomistic import (
    AnonymousStructure,
    AnonymousStructureView,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
)
from httk.atomistic.models.prototype.anonymize import (
    canonical_dummy_assignment,
    dummy_species,
    is_dummy_species,
)

CELL = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
SITES = [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]]


def test_dummy_species_round_trip_and_decorations() -> None:
    assert is_dummy_species(dummy_species("A"))
    assert not is_dummy_species(Species("A", ("X",), (1,), labels=("other",)))
    assert not is_dummy_species(Species("A", ("Na", "Cl"), (1, 1)))


def test_anonymous_structure_infers_species_and_rejects_bad_labels() -> None:
    value = AnonymousStructure(CELL, SITES, species_at_sites=("A", "B"))
    assert tuple(species.name for species in value.species) == ("A", "B")
    with pytest.raises(ValueError, match="consecutive"):
        AnonymousStructure(CELL, [[0, 0, 0]], species_at_sites=("B",), species=(dummy_species("B"),))
    with pytest.raises(ValueError, match="unknown"):
        AnonymousStructure(CELL, [[0, 0, 0]], species=(dummy_species("A"),), species_at_sites=("B",))
    with pytest.raises(ValueError, match="dummy"):
        AnonymousStructure(CELL, [[0, 0, 0]], species=(Species("A", ("Na",), (1,)),), species_at_sites=("A",))


def test_canonical_assignment_and_canonicality() -> None:
    assert canonical_dummy_assignment((("Na", Fraction(1)), ("Cl", Fraction(1)))) == {"Cl": "A", "Na": "B"}
    assert canonical_dummy_assignment((("Na", Fraction(2)), ("Cl", Fraction(1)))) == {"Na": "A", "Cl": "B"}
    canonical = AnonymousStructure(CELL, [[0, 0, 0], [0, 0, 0], [0, 0, 0]], species_at_sites=("A", "A", "B"))
    swapped = AnonymousStructure(CELL, [[0, 0, 0], [0, 0, 0], [0, 0, 0]], species_at_sites=("A", "B", "B"))
    assert canonical.is_canonical
    assert not swapped.is_canonical


def test_anonymous_view_is_canonical_lazy_and_preserves_formula() -> None:
    species = (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,)))
    structure = UnitcellStructure(CELL, SITES, species, ("Na", "Cl"))
    view = AnonymousStructureView(structure)
    assert "_derived" not in view._backend.__dict__
    assert view.is_canonical
    assert view.anonymous_formula == UnitcellStructureView(structure).chemical_formula_anonymous
    assert view.unwrap() is structure


def test_anonymization_rejects_unsupported_structure_features_at_construction() -> None:
    cases = [
        (Species("mixed", ("Na", "Cl"), (Fraction(1, 2), Fraction(1, 2))), "single real element"),
        (Species("unknown", ("X",), (1,)), "single real element"),
        (Species("vacancy", ("vacancy",), (1,)), "single real element"),
        (Species("attached", ("Na",), (1,), attached=("H",), nattached=(1,)), "single real element"),
    ]
    for species, message in cases:
        structure = UnitcellStructure(CELL, [[0, 0, 0]], (species,), (species.name,))
        with pytest.raises(ValueError, match=message):
            AnonymousStructureView(structure)


def test_anonymous_structures_are_not_structure_like() -> None:
    value = AnonymousStructure(CELL, [[0, 0, 0]], species_at_sites=("A",))
    with pytest.raises(TypeError):
        UnitcellStructureView(value)


def test_anonymization_rejects_unused_species_and_contradictory_kind() -> None:
    structure = UnitcellStructure(
        CELL,
        [[0, 0, 0]],
        (Species("Na1", ("Na",), (1,)), Species("Na2", ("Na",), (1,))),
        ("Na1",),
    )
    with pytest.raises(ValueError, match="Na2"):
        AnonymousStructureView(structure)
    with pytest.raises(TypeError):
        AnonymousStructureView(structure, kind="bogus")


def test_matched_numeric_structure_re_raises_species_backend_errors() -> None:
    pytest.importorskip("numpy")
    from httk.atomistic.models.cell.numeric import NumericCell
    from httk.atomistic.models.sites.numeric import NumericSites

    class NumericObject:
        cell = NumericCell(CELL)
        sites = NumericSites([[0, 0, 0]])
        species = ({"not": "a species"},)
        species_at_sites = ("invalid",)

    with pytest.raises(TypeError, match="SpeciesBackend"):
        AnonymousStructureView(NumericObject())
