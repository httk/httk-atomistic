from fractions import Fraction

import pytest

from httk.atomistic.models.formula.composition import Composition


def test_composition_fields_shadow_api_defaults() -> None:
    composition = Composition({"Al": 2, "O": 3})
    assert composition.amounts == (("Al", Fraction(2)), ("O", Fraction(3)))
    assert composition.complete is True
    assert composition.uncertainties == (("Al", None), ("O", None))
    assert composition.exact is True


def test_composition_accepts_mappings_and_pairs_and_sorts_alphabetically() -> None:
    mapping = Composition({"O": 3, "Al": 2})
    pairs = Composition((("O", 3), ("Al", 2)))
    assert mapping == pairs
    assert mapping.amounts == (("Al", Fraction(2)), ("O", Fraction(3)))
    assert Composition({}) == Composition(())

    with pytest.raises(ValueError, match="real element"):
        Composition({"not-an-element": 1})
    with pytest.raises(ValueError, match="positive"):
        Composition({"Al": 0})
    with pytest.raises(ValueError, match="positive"):
        Composition({"Al": -1})


def test_composition_positional_projection_shape_round_trips() -> None:
    source = Composition(
        (("Ge", Fraction(5, 8)), ("Si", Fraction(3, 8))),
        (("Ge", None), ("Si", None)),
        True,
        True,
        True,
        "exact",
        (),
    )
    copied = Composition(
        source.amounts,
        source.uncertainties,
        source.complete,
        source.exact,
        source.normalized,
        source.normalization_status,
        source.diagnostics,
    )
    assert copied == source


def test_composition_formula_rendering_matches_existing_strings() -> None:
    composition = Composition({"Ge": 5, "Si": 3})
    assert composition.chemical_formula_reduced == "Ge5Si3"
    assert composition.chemical_formula_anonymous == "A5B3"


def test_composition_equality_and_hash_include_all_seven_fields() -> None:
    first = Composition({"Al": 2, "O": 3})
    second = Composition({"O": 3, "Al": 2})
    differing = Composition({"Al": 2, "O": 3}, complete=False)

    assert first == second
    assert hash(first) == hash(second)
    assert first != differing

    class ChildComposition(Composition):
        pass

    child = ChildComposition({"Al": 2, "O": 3})
    assert first == child
    assert child == first
