"""Golden and round-trip tests for httk protopattern/protostructure label notation."""

import pytest

from httk.atomistic import Protopattern, Spacegroup
from httk.atomistic.models.protopattern.notation import (
    parse_protopattern_label,
    parse_protostructure_label,
    pearson_symbol,
    render_aflow_label,
    render_protopattern_label,
    render_protostructure_label,
)


def test_rocksalt_protopattern_label() -> None:
    pattern = Protopattern(225, [("a", "A"), ("b", "B")])
    assert str(pattern.label) == "AB_cF8_225_a_b"


def test_calcite_protopattern_label() -> None:
    pattern = Protopattern(167, [("a", "A"), ("b", "B"), ("e", "C")])
    assert str(pattern.label) == "ABC3_hR10_167_a_b_e"


def test_rhombohedral_divides_conventional_count_by_three() -> None:
    assert pearson_symbol(Spacegroup.standard(167), 30) == "hR10"


def test_a_centred_group_uses_centring_letter_c() -> None:
    # Group 38 is A-centred orthorhombic; the base-centred variant folds to Pearson letter C.
    assert pearson_symbol(Spacegroup.standard(38), 4).startswith("oC")


def test_special_27th_letter_round_trips_as_uppercase_a() -> None:
    pattern = Protopattern(47, [("α", "A")])
    assert str(pattern.label) == "A_oP8_47_A"
    assert parse_protopattern_label("A_oP8_47_A") == pattern


def test_repeated_letter_group_pins() -> None:
    pattern = Protopattern(47, [("i", "A"), ("i", "A")])
    assert str(pattern.label) == "A_oP4_47_2i"
    assert parse_protopattern_label("A_oP4_47_2i") == pattern


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
        parse_protopattern_label(text)


def test_protostructure_parser_rejects_unknown_element() -> None:
    with pytest.raises(ValueError, match="element symbol"):
        parse_protostructure_label("AB_cF8_225_a_b:Xx-Cl")


def test_render_parse_render_loop_over_valid_patterns() -> None:
    cases = [
        Protopattern(225, [("a", "A"), ("b", "B")]),
        Protopattern(167, [("a", "A"), ("b", "B"), ("e", "C")]),
        Protopattern(47, [("α", "A")]),
        Protopattern(47, [("i", "A"), ("i", "A")]),
        Protopattern(1, [("a", "A")]),
        Protopattern(225, [("a", "A"), ("b", "A"), ("c", "B")]),
    ]
    for pattern in cases:
        text = render_protopattern_label(
            pattern.spacegroup, [(o.wyckoff, o.label) for o in pattern.occupations]
        )
        assert parse_protopattern_label(text) == pattern
        again = parse_protopattern_label(text)
        assert render_protopattern_label(again.spacegroup, [(o.wyckoff, o.label) for o in again.occupations]) == text
