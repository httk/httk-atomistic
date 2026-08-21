"""Tests for the anonymous geometrical-class prototype family."""

import pickle

import pytest
from httk.core import FracVector

from httk.atomistic import (
    ASUStructure,
    FundamentalDomainPattern,
    Protochroma,
    Prototype,
    PrototypeView,
    Species,
    WyckoffSite,
)

CELL = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
EMPTY = FracVector(())


def _dummy(label: str) -> Species:
    return Species(label, ("X",), (1,), labels=(label,))


def _rocksalt_pattern() -> FundamentalDomainPattern:
    return FundamentalDomainPattern(
        CELL,
        225,
        (WyckoffSite("a", EMPTY, "A"), WyckoffSite("b", EMPTY, "B")),
        (_dummy("A"), _dummy("B")),
    )


def _rocksalt_asu() -> ASUStructure:
    return ASUStructure(
        CELL,
        225,
        (WyckoffSite("a", EMPTY, "Na"), WyckoffSite("b", EMPTY, "Cl")),
        (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
    )


def test_requires_at_least_one_of_representative_or_discriminator() -> None:
    pattern = _rocksalt_pattern().protochroma
    with pytest.raises(ValueError, match="at least one of representative or discriminator"):
        Prototype(pattern)


def test_protochroma_only_is_rejected() -> None:
    pattern = Protochroma(225, [("a", "A"), ("b", "B")])
    with pytest.raises(ValueError, match="at least one of representative or discriminator"):
        Prototype(pattern)


def test_discriminator_only_requires_a_protochroma() -> None:
    with pytest.raises(ValueError, match="needs a protochroma"):
        Prototype(discriminator="001")


def test_discriminator_must_be_non_empty_string() -> None:
    pattern = Protochroma(225, [("a", "A"), ("b", "B")])
    with pytest.raises(ValueError, match="non-empty string"):
        Prototype(pattern, discriminator="")


def test_representative_only_derives_the_protochroma() -> None:
    representative = _rocksalt_pattern()
    prototype = Prototype(representative=representative)
    assert prototype.representative == representative
    assert prototype.discriminator is None
    assert prototype.protochroma == representative.protochroma


def test_both_given_agreement_is_enforced() -> None:
    representative = _rocksalt_pattern()
    # A single-class pattern cannot describe the two-class rocksalt representative.
    mismatched = Protochroma(221, [("a", "A")])
    with pytest.raises(ValueError, match="disagrees with its representative"):
        Prototype(mismatched, representative=representative)
    # The agreeing protochroma is accepted.
    prototype = Prototype(representative.protochroma, representative=representative)
    assert prototype.representative == representative


def test_representative_only_is_never_equal_to_discriminator_only() -> None:
    representative = _rocksalt_pattern()
    pattern = representative.protochroma
    representative_only = Prototype(representative=representative)
    discriminator_only = Prototype(pattern, discriminator="001")
    assert representative_only != discriminator_only


def test_equality_and_inequality_across_the_triple() -> None:
    representative = _rocksalt_pattern()
    pattern = representative.protochroma
    assert Prototype(representative=representative) == Prototype(representative=representative)
    assert Prototype(pattern, discriminator="001") == Prototype(pattern, discriminator="001")
    assert Prototype(pattern, discriminator="001") != Prototype(pattern, discriminator="002")
    assert Prototype(representative=representative, discriminator="001") != Prototype(representative=representative)
    assert Prototype(representative=representative) != object()


def test_prototype_is_unhashable() -> None:
    with pytest.raises(TypeError):
        hash(Prototype(representative=_rocksalt_pattern()))


def test_view_recognizes_a_fundamental_domain_pattern_carrying_a_representative() -> None:
    representative = _rocksalt_pattern()
    prototype = PrototypeView(representative).unview()
    assert isinstance(prototype, Prototype)
    assert prototype.representative == representative
    assert prototype.discriminator is None
    assert prototype.protochroma == representative.protochroma


def test_view_recognizes_an_exact_asu_without_spglib() -> None:
    # The build-cod pass-2 path: PrototypeView(canonical_asu).unview() must not need spglib.
    prototype = PrototypeView(_rocksalt_asu()).unview()
    assert prototype.representative is not None
    assert prototype.discriminator is None
    assert prototype.label == "AB_cF8_225_a_b"


def test_label_pearson_and_anonymous_formula_delegate_to_the_protochroma() -> None:
    prototype = Prototype(representative=_rocksalt_pattern())
    pattern = prototype.protochroma
    assert prototype.label == pattern.label
    assert prototype.pearson_symbol == pattern.pearson_symbol == "cF8"
    assert prototype.anonymous_formula == pattern.anonymous_formula == "AB"
    assert prototype.nsites_conventional == pattern.nsites_conventional
    assert prototype.spacegroup == pattern.spacegroup


def test_view_of_a_native_value_unviews_to_the_same_identity() -> None:
    native = Prototype(Protochroma(225, [("a", "A"), ("b", "B")]), discriminator="001")
    view = PrototypeView(native)
    assert view.unview() is native
    assert view.unwrap() is native


def test_view_rewrap_rejects_recognition_arguments() -> None:
    view = PrototypeView(_rocksalt_asu())
    with pytest.raises(ValueError):
        PrototypeView(view, tolerance=0.1)


def test_bare_protochroma_is_not_a_prototype_source() -> None:
    with pytest.raises(TypeError):
        PrototypeView(Protochroma(225, [("a", "A"), ("b", "B")]))


def test_prototype_value_pickle_round_trip() -> None:
    representative_only = Prototype(representative=_rocksalt_pattern())
    discriminator_only = Prototype(Protochroma(225, [("a", "A"), ("b", "B")]), discriminator="001")
    for value in (representative_only, discriminator_only):
        restored = pickle.loads(pickle.dumps(value))
        assert restored == value


def test_prototype_view_pickle_preserves_resolved_value() -> None:
    view = PrototypeView(_rocksalt_asu())
    _ = view.protochroma  # resolve
    restored = pickle.loads(pickle.dumps(view))
    assert restored.unview() == view.unview()


def test_prototype_view_unresolved_pickle_stays_lazy() -> None:
    view = PrototypeView(_rocksalt_asu())  # no field access -> unresolved
    restored = pickle.loads(pickle.dumps(view))
    assert restored._resolved_prototype is None
    assert restored.unview() == view.unview()
