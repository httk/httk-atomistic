"""Golden and round-trip tests for httk prototype/protostructure label notation."""

import pytest

from httk.atomistic import Prototype, Spacegroup
from httk.atomistic.models.prototype.notation import (
    parse_protostructure_label,
    parse_prototype_label,
    pearson_symbol,
    render_aflow_label,
    render_protostructure_label,
    render_prototype_label,
)


def test_rocksalt_prototype_label() -> None:
    template = Prototype(225, [("a", "A"), ("b", "B")])
    assert str(template.label) == "AB_cF8_225_a_b"


def test_calcite_prototype_label() -> None:
    template = Prototype(167, [("a", "A"), ("b", "B"), ("e", "C")])
    assert str(template.label) == "ABC3_hR10_167_a_b_e"


def test_rhombohedral_divides_conventional_count_by_three() -> None:
    assert pearson_symbol(Spacegroup.standard(167), 30) == "hR10"


def test_a_centred_group_uses_centring_letter_c() -> None:
    # Group 38 is A-centred orthorhombic; the base-centred variant folds to Pearson letter C.
    assert pearson_symbol(Spacegroup.standard(38), 4).startswith("oC")


def test_special_27th_letter_round_trips_as_uppercase_a() -> None:
    template = Prototype(47, [("α", "A")])
    assert str(template.label) == "A_oP8_47_A"
    assert parse_prototype_label("A_oP8_47_A") == template


def test_repeated_letter_group_pins() -> None:
    template = Prototype(47, [("i", "A"), ("i", "A")])
    assert str(template.label) == "A_oP4_47_2i"
    assert parse_prototype_label("A_oP4_47_2i") == template


def test_repeated_special_letter_group_pins() -> None:
    # The special 27th letter α occupied twice by one class renders "2A" (multiplicity 8 each).
    template = Prototype(47, [("α", "A"), ("α", "A")])
    assert str(template.label) == "A_oP16_47_2A"
    assert parse_prototype_label("A_oP16_47_2A") == template


@pytest.mark.parametrize(
    ("it_number", "occupations", "expected"),
    [
        # Trigonal-P and hexagonal-P: the count is NOT divided by three (that is R-only).
        (156, [("a", "A"), ("b", "B")], "AB_hP2_156_a_b"),
        (194, [("a", "A"), ("b", "A"), ("f", "B"), ("f", "B")], "AB2_hP12_194_ab_2f"),
    ],
)
def test_non_rhombohedral_pearson_is_not_divided(it_number: int, occupations: list, expected: str) -> None:
    template = Prototype(it_number, occupations)
    assert str(template.label) == expected
    assert parse_prototype_label(expected) == template


def test_protostructure_label_and_aflow_divergence() -> None:
    sg = Spacegroup.standard(167)
    occupations = [("a", "Ca"), ("b", "C"), ("e", "O")]
    assert render_protostructure_label(sg, occupations) == "ABC3_hR10_167_a_b_e:Ca-C-O"
    # AFLOW orders classes by element symbol, so the unsuffixed prefix reorders too.
    assert render_aflow_label(sg, occupations) == "ABC3_hR10_167_b_a_e:C-Ca-O"


@pytest.mark.parametrize(
    "text",
    [
        "AA_cF8_225_a_b",  # bad anonymous label sequence
        "AB_cP8_225_a_b",  # wrong Pearson centring
        "AB2_cF8_225_a_b",  # wrong anonymous counts
        "AB_cF8_225_a_zz",  # unknown Wyckoff letter
        "AB_cF8_225_b_a",  # non-canonical group order
        "A_cF8_225_1a",  # explicit count 1 is invalid
    ],
)
def test_strict_parser_rejects_non_canonical(text: str) -> None:
    with pytest.raises(ValueError):
        parse_prototype_label(text)


def test_protostructure_parser_rejects_unknown_element() -> None:
    with pytest.raises(ValueError, match="element symbol"):
        parse_protostructure_label("AB_cF8_225_a_b:Xx-Cl")


def test_render_parse_render_loop_over_valid_templates() -> None:
    cases = [
        Prototype(225, [("a", "A"), ("b", "B")]),
        Prototype(167, [("a", "A"), ("b", "B"), ("e", "C")]),
        Prototype(47, [("α", "A")]),
        Prototype(47, [("i", "A"), ("i", "A")]),
        Prototype(1, [("a", "A")]),
        Prototype(225, [("a", "A"), ("b", "A"), ("c", "B")]),
    ]
    for template in cases:
        text = render_prototype_label(template.spacegroup, [(o.wyckoff, o.label) for o in template.occupations])
        assert parse_prototype_label(text) == template
        again = parse_prototype_label(text)
        assert render_prototype_label(again.spacegroup, [(o.wyckoff, o.label) for o in again.occupations]) == text
