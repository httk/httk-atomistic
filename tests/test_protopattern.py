"""Tests for the geometry-free, element-free protopattern family."""

import pytest
from httk.core import FracVector

from httk.atomistic import (
    ASUStructure,
    FundamentalDomainPattern,
    Protopattern,
    ProtopatternLabel,
    ProtopatternOccupation,
    ProtopatternView,
    Protostructure,
    Spacegroup,
    Species,
    WyckoffSite,
)

CELL = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
EMPTY = FracVector(())


def _rocksalt_pattern() -> Protopattern:
    return Protopattern(225, [("a", "A"), ("b", "B")])


def test_construction_validates_setting_letters_and_labels() -> None:
    with pytest.raises(ValueError, match="standard setting"):
        Protopattern(Spacegroup.from_setting("15:c1"), [("e", "A")])
    with pytest.raises(ValueError, match="non-empty"):
        Protopattern(225, [])
    with pytest.raises(ValueError, match="no Wyckoff letter"):
        Protopattern(225, [("zz", "A")])
    with pytest.raises(ValueError, match="consecutive anonymous symbols"):
        Protopattern(225, [("a", "A"), ("b", "C")])


def test_canonical_relabeling_is_permutation_invariant() -> None:
    first = Protopattern(225, [("a", "A"), ("b", "B")])
    permuted = Protopattern(Spacegroup.standard(225), [("a", "B"), ("b", "A")])
    assert first == permuted
    assert str(first.label) == str(permuted.label) == "AB_cF8_225_a_b"
    assert first.occupations == permuted.occupations


def test_interchangeable_classes_tie_to_the_same_label() -> None:
    first = Protopattern(225, [("e", "A"), ("e", "B")])
    swapped = Protopattern(225, [("e", "B"), ("e", "A")])
    assert first == swapped
    assert str(first.label) == str(swapped.label)


def test_hashable_and_usable_as_dict_key() -> None:
    first = _rocksalt_pattern()
    same = Protopattern(225, [("b", "B"), ("a", "A")])
    assert {first: "value"}[same] == "value"
    assert hash(first) == hash(same)


def test_repeated_letter_renders_with_count() -> None:
    # One class occupying the twofold position i twice: a doubled single letter renders "2i".
    doubled = Protopattern(47, [("i", "A"), ("i", "A")])
    assert str(doubled.label) == "A_oP4_47_2i"


def test_erasure_from_protostructure() -> None:
    proto = Protostructure(225, [("a", "Na"), ("b", "Cl")])
    view = ProtopatternView(proto)
    assert str(view.label) == "AB_cF8_225_a_b"
    assert view.unwrap() is proto


def test_erasure_from_fundamental_domain_pattern_needs_no_spglib() -> None:
    fdp = FundamentalDomainPattern(CELL, 225, (WyckoffSite("a", EMPTY, "A"), WyckoffSite("b", EMPTY, "B")))
    assert str(ProtopatternView(fdp).label) == "AB_cF8_225_a_b"


def test_erasure_from_structure_via_recognition() -> None:
    pytest.importorskip("spglib")
    sodium = Species("Na", ("Na",), (1,))
    chlorine = Species("Cl", ("Cl",), (1,))
    asu = ASUStructure(CELL, 225, (WyckoffSite("a", EMPTY, "Na"), WyckoffSite("b", EMPTY, "Cl")), (sodium, chlorine))
    assert str(ProtopatternView(asu).label) == "AB_cF8_225_a_b"


def test_label_string_dispatch() -> None:
    view = ProtopatternView("AB_cF8_225_a_b")
    assert view.unview() == _rocksalt_pattern()
    assert str(ProtopatternLabel("AB_cF8_225_a_b")) == "AB_cF8_225_a_b"


def test_view_is_lazy_and_unwrap_is_identity() -> None:
    proto = Protostructure(225, [("a", "Na"), ("b", "Cl")])
    view = ProtopatternView(proto)
    assert view._resolved_protopattern is None
    assert view.unwrap() is proto
    assert view._resolved_protopattern is None
    _ = view.spacegroup
    assert view._resolved_protopattern is not None
    assert view.unview() == _rocksalt_pattern()


def test_native_protopattern_view_returns_value_identity() -> None:
    pattern = _rocksalt_pattern()
    assert ProtopatternView(pattern).unview() is pattern
    assert ProtopatternView(ProtopatternView(pattern)) is not None


def test_occupation_str_coerces_fields() -> None:
    occupation = ProtopatternOccupation(0, 0)
    assert occupation.wyckoff == "0" and occupation.label == "0"
