from fractions import Fraction

import pytest

from httk.atomistic import PlainSpecies, PlainSpeciesView, Species, SpeciesView
from httk.atomistic.entries._payloads import species_payload


def test_decorations_normalize_and_canonicalize() -> None:
    species = Species(
        "Fe",
        ("Fe", "O"),
        (1, 1),
        charges=("5/2", None),
        spins=(1, "-1/2"),
        labels=("host", None),
    )

    assert species.charges == (Fraction(5, 2), None)
    assert species.spins == (Fraction(1), Fraction(-1, 2))
    assert Species("Fe", ("Fe",), (1,), charges=(None,)).charges is None
    assert Species("Fe", ("Fe",), (1,), spins=(None,)).spins is None
    assert Species("Fe", ("Fe",), (1,), labels=(None,)).labels is None


def test_without_charges_drops_only_charges() -> None:
    species = Species("Fe", ("Fe",), (1,), charges=(2,), spins=("1/2",), labels=("host",))

    projected = species.without_charges()

    assert projected.charges is None
    assert projected.spins == (Fraction(1, 2),)
    assert projected.labels == ("host",)
    plain = Species("Fe", ("Fe",), (1,))
    assert plain.without_charges() is plain


@pytest.mark.parametrize("field", ["charges", "spins", "labels"])
def test_decorations_must_align(field: str) -> None:
    with pytest.raises(ValueError, match=f"Species {field} must have the same length"):
        Species("Fe", ("Fe", "O"), (1, 1), **{field: (None,)})


def test_repeated_symbols_require_distinct_decorations() -> None:
    decorated = Species("Fe", ("Fe", "Fe"), (1, 1), charges=(2, 3))
    assert decorated.charges == (Fraction(2), Fraction(3))

    with pytest.raises(ValueError, match="decorations.*mass/charge/spin/label"):
        Species("Fe", ("Fe", "Fe"), (1, 1))
    with pytest.raises(ValueError, match="exact duplicate constituents"):
        Species("Fe", ("Fe", "Fe"), (1, 1), charges=(2, 2))


def test_decoration_fields_participate_in_identity() -> None:
    plain = Species("Fe", ("Fe",), (1,))
    charged = Species("Fe", ("Fe",), (1,), charges=(0,))
    same = Species("Fe", ("Fe",), (1,), charges=("0",))

    assert plain != charged
    assert charged == same
    assert hash(charged) == hash(same)


def test_exact_fraction_decorations_participate_in_identity() -> None:
    first = Species("Fe", ("Fe", "O"), (1, 1), charges=("1/3", None))
    same = Species("Fe", ("Fe", "O"), (1, 1), charges=(Fraction(1, 3), None))

    assert first.charges == (Fraction(1, 3), None)
    assert first == same
    assert hash(first) == hash(same)


def test_plain_species_canonicalizes_all_none_decorations() -> None:
    raw = {
        "name": "Fe",
        "chemical_symbols": ["Fe"],
        "concentration": [1],
        "_httk_charges": [None],
        "_httk_spins": [None],
        "_httk_labels": [None],
    }
    plain = PlainSpecies(raw)

    assert plain.charges is None
    assert plain.spins is None
    assert plain.labels is None
    assert "_httk_charges" not in PlainSpeciesView(plain)
    assert "_httk_spins" not in PlainSpeciesView(plain)
    assert "_httk_labels" not in PlainSpeciesView(plain)


def test_plain_species_keeps_misaligned_all_none_evidence() -> None:
    raw = {
        "name": "SiO",
        "chemical_symbols": ["Si", "O"],
        "concentration": [1, 1],
        "_httk_charges": [None],
    }
    plain = PlainSpecies(raw)

    assert plain.charges == (None,)
    with pytest.raises(ValueError, match="charges must have the same length"):
        SpeciesView(plain)


def test_dict_and_payload_round_trip_decorations() -> None:
    raw = {
        "name": "Fe",
        "chemical_symbols": ["Fe", "Fe"],
        "concentration": [1, 1],
        "_httk_charges": [2.5, None],
        "_httk_spins": [0, None],
        "_httk_labels": ["site-a", None],
    }
    species = Species.from_object(raw)

    assert species.charges == (Fraction(5, 2), None)
    assert species.spins == (Fraction(0), None)
    assert species.labels == ("site-a", None)
    assert SpeciesView(raw).charges == species.charges
    assert Species.from_object(PlainSpeciesView(species)).charges == species.charges
    payload = species_payload(species)
    assert payload["_httk_charges"] == [2.5, None]
    assert payload["_httk_spins"] == [0.0, None]
    assert Species.from_object(payload).charges == species.charges
    assert Species.from_object({**raw, "_httk_charges": [0, None]}).charges == (Fraction(0), None)
