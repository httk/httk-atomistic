"""Lossless CIF reading and writing: ``read_cif(..., lossless=True)`` and ``write_cif`` of blocks.

The ``cif_lossless`` fixtures are small synthetic files written for these tests.
"""

import dataclasses
import io
import logging
import re
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import httk.core
import pytest
from httk.core import DecimalGap, DecimalScalar, DecimalVector

from httk.atomistic.io.cif import (
    CIFBlock,
    CIFComment,
    CIFDataItem,
    CIFLoop,
    CIFSaveFrame,
    CIFTable,
    CIFTextField,
    read_cif,
)
from httk.atomistic.io.cif.cif_items import _cif_mode, _is_cif2_header
from httk.atomistic.io.cif.cif_reader import _legacy_block
from httk.atomistic.io.cif.cif_writer import write_cif

FIXTURES = Path(__file__).parent / "fixtures"
LOSSLESS = FIXTURES / "cif_lossless"
ALL_FIXTURES = sorted(path for path in FIXTURES.rglob("*") if path.suffix in (".cif", ".mcif"))
MALFORMED = FIXTURES / "malformed_auxiliary_loop.cif"
# A synthetic probe of a documented fast-reader divergence (D8): an mCIF quote directly before
# ']'. The lossless model is correct; the fast reader is not.
FAST_READER_FAULTS = {LOSSLESS / "mcif_brackets.mcif"}
CIF2 = "#\\#CIF_2.0\n"
C = CIFComment


def _read(source: Any, mcif: bool = False, **options: Any) -> list[CIFBlock]:
    if isinstance(source, str):
        source = io.StringIO(source)
    return read_cif(source, allow_cif2=mcif, lossless=True, **options)


def _write(blocks: CIFBlock | list[CIFBlock]) -> str:
    out = io.StringIO()
    write_cif(out, blocks)
    return out.getvalue()


def _assert_round_trip(blocks: list[CIFBlock], mcif: bool) -> str:
    text = _write(blocks)
    again = _read(text, mcif)
    assert again == blocks
    assert _write(again) == text
    return text


def _ids(path: Path) -> str:
    return str(path.relative_to(FIXTURES))


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=_ids)
def test_every_fixture_round_trips(path: Path) -> None:
    mcif = path.suffix == ".mcif"
    if path == MALFORMED:
        with pytest.raises(ValueError, match="has 3 values for 2 data names"):
            read_cif(path, lossless=True, repair=True)  # D3: repair never drops a loop here
        return
    blocks = read_cif(path, allow_cif2=mcif, lossless=True)
    assert blocks and all(block.dialect == ("mcif" if mcif else "cif") for block in blocks)
    _assert_round_trip(blocks, mcif)


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=_ids)
def test_every_fixture_renders_the_fast_readers_dictionary(path: Path, caplog: pytest.LogCaptureFixture) -> None:
    mcif = path.suffix == ".mcif"
    if path == MALFORMED or path in FAST_READER_FAULTS:
        return
    blocks = read_cif(path, allow_cif2=mcif, lossless=True)
    if _cif_mode(blocks[0]) == "cif2":
        return  # the fast reader has no CIF2 syntax
    # As conversion uses it: structural for CIF, everything for mCIF.
    with caplog.at_level(logging.CRITICAL):
        fast = read_cif(path, allow_cif2=mcif, structural_only=not mcif)[0]
    assert [(block.name.lower(), _legacy_block(block, structural_only=not mcif)) for block in blocks] == fast
    if not mcif:  # and every item, including the loops structure reading skips
        with caplog.at_level(logging.CRITICAL):
            full = read_cif(path)[0]
        assert [(block.name.lower(), _legacy_block(block, structural_only=False)) for block in blocks] == full


BONDS_CIF = """\
data_BONDED
_cell_length_a 5.0
_journal_name_full 'Not retained'
loop_
_atom_site_label
_atom_site_fract_x
C1 0.1
O1 0.2
loop_
_geom_bond_atom_site_label_1
_geom_bond_atom_site_label_2
_geom_bond_site_symmetry_2
_geom_bond_distance
_geom_bond_publ_flag
C1 O1 . 1.21(2) yes
loop_
_geom_angle_atom_site_label_1
_geom_angle
C1 109.5
"""


@pytest.mark.parametrize("structural_only", [True, False])
def test_bond_loops_render_like_the_fast_reader(structural_only: bool) -> None:
    (block,) = _read(BONDS_CIF)
    fast = read_cif(io.StringIO(BONDS_CIF), structural_only=structural_only)[0]
    assert [("bonded", _legacy_block(block, structural_only=structural_only))] == fast
    legacy = _legacy_block(block, structural_only=True)
    assert legacy["geom_bond_distance"] == ["1.21(2)"] and "geom_angle" not in legacy
    assert "journal_name_full" not in legacy


def test_comments_keep_text_position_and_inline_flag() -> None:
    (block,) = read_cif(LOSSLESS / "comments.cif", lossless=True)
    assert block.items == (
        CIFDataItem("_a", DecimalScalar("1.0"), ((1, C(" after value", True)),)),
        CIFDataItem("_b", DecimalScalar("2.0"), ((0, C(" between name and value")),)),
        CIFLoop(
            ("_atom_site_label", "_atom_site_fract_x"),
            (("Fe1", DecimalScalar("0.1")), ("O1", DecimalScalar("0.2"))),
            (
                (0, C(" after loop_", True)),
                (1, C(" after name", True)),
                (1, C(" own-line among names")),
                (4, C(" after first row", True)),
                (4, C(" between rows")),
                (5, C(" mid-row")),
                (6, C(" after last value", True)),
            ),
        ),
        C(" own-line after loop"),
        CIFDataItem("_c", "x"),
    )
    _assert_round_trip([block], False)


def test_header_leading_and_trailing_comments_of_blocks() -> None:
    blocks = read_cif(LOSSLESS / "multiblock.cif", lossless=True)
    header = (C(" header line 1"), C(" header line 2"))
    assert [block.name for block in blocks] == ["first", "notes", "last"]
    assert all(block.header == header for block in blocks)
    assert [block.leading for block in blocks] == [
        (C(" leading of first block"),),
        (C(" trailing of first: moves to second.leading"),),
        (C(" between blocks"),),
    ]
    assert blocks[0].items[0] == C(" inline on data line", True)
    assert blocks[-1].items[-1] == C(" trailing of the last block")
    assert blocks[1].find("_x") == CIFLoop(("_x",), ((DecimalScalar("1"),), (DecimalScalar("2"),)))
    text = _assert_round_trip(blocks, False)
    assert text.startswith("# header line 1\n# header line 2\n\n# leading of first block\ndata_first # inline")


def test_one_block_is_written_alone_with_its_header() -> None:
    blocks = read_cif(LOSSLESS / "multiblock.cif", lossless=True)
    text = _write(blocks[1])
    assert text.startswith("# header line 1\n# header line 2\n\n# trailing of first")
    assert _read(text) == [blocks[1]]


def test_blocks_must_share_header_and_dialect() -> None:
    first, second, _ = read_cif(LOSSLESS / "multiblock.cif", lossless=True)
    message = "blocks carry different file headers or dialects"
    with pytest.raises(ValueError, match=message):
        _write([first, dataclasses.replace(second, header=())])
    with pytest.raises(ValueError, match=message):
        _write([dataclasses.replace(first, dialect="mcif"), second])


def test_non_last_block_must_not_end_with_own_line_comments() -> None:
    first, second, _ = read_cif(LOSSLESS / "multiblock.cif", lossless=True)
    ending = dataclasses.replace(first, items=(*first.items, C(" stray")))
    with pytest.raises(ValueError, match="block 'first' ends with own-line comments.*dataclasses.replace"):
        _write([ending, second])
    _write([second, ending])  # the last block keeps its trailing comments
    moved = [first, dataclasses.replace(second, leading=(C(" stray"), *second.leading))]
    assert _read(_write(moved)) == moved


def test_write_cif_refuses_mixtures_and_header() -> None:
    (block,) = _read("data_x\n_a 1\n")
    with pytest.raises(TypeError, match="not a mixture"):
        write_cif(io.StringIO(), [block, ("y", {"a": "1"})])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="header"):
        write_cif(io.StringIO(), block, header="# h")


def test_write_cif_to_a_path(tmp_path: Path) -> None:
    blocks = read_cif(LOSSLESS / "multiblock.cif", lossless=True)
    target = tmp_path / "out.cif"
    write_cif(target, blocks)
    assert read_cif(target, lossless=True) == blocks
    assert target.read_bytes() == _write(blocks).encode()


def test_numbers_come_back_as_written() -> None:
    (block,) = read_cif(LOSSLESS / "numbers.cif", lossless=True)
    tokens = {
        "_n1": "0.50",
        "_n2": "0.5",
        "_n3": "1.2345(6)",
        "_n4": "12(1)",
        "_n5": "1.5e-3",
        "_n6": "007",
        "_n7": "+1.0",
        "_n10": "1e-999999999",
        "_n11": "1234567890123456789012345678901234567890.5",
        "_n12": "1(2)e999999999",
        "_n14": "1.5(2)E-3",
        "_n15": "1.23(06)",
    }
    for name, token in tokens.items():
        number = block.value(name)
        assert isinstance(number, DecimalScalar) and number.token == token
    assert block.number("_n3").uncertainty == Decimal("0.0006")
    assert block.number("_n1") != block.number("_n2")
    assert block.value("_n8") is DecimalGap.UNKNOWN and block.value("_n9") is DecimalGap.INAPPLICABLE
    assert [block.value(name) for name in ("_s1", "_s2", "_s3")] == ["1.5", "?", "."]
    # D7: a NUMBER-shaped token beyond Decimal's exponent range is a string, written bare.
    assert block.value("_n13") == "1e99999999999999999999"
    text = _assert_round_trip([block], False)
    assert "\n_n13 1e99999999999999999999\n" in text and "\n_s1 '1.5'\n" in text
    for token in tokens.values():
        assert f" {token}\n" in text


def test_lookup_is_caseless_and_exposes_numbers() -> None:
    (block,) = _read(
        "data_x\n_Cell_Length_A 5.0(1)\n_cell_length_b ?\n_cell_length_c 7\n_title 'T'\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\nA 0.1 0.2\nB . 0.25\n"
    )
    assert block.find("_cell_length_a") == CIFDataItem("_Cell_Length_A", DecimalScalar("5.0(1)"))
    assert block.number("_CELL_LENGTH_A").token == "5.0(1)"
    assert block.number("_cell_length_b") == DecimalScalar("?")
    cell = block.numbers("_cell_length_a", "_cell_length_b", "_cell_length_c")
    assert cell == DecimalVector(["5.0(1)", "?", "7"])
    assert block.numbers("_atom_site_fract_x") == DecimalVector(["0.1", "."])
    assert block.numbers("_atom_site_fract_x", "_atom_site_fract_y").dim == (2, 2)
    assert block.column("_atom_site_label") == ("A", "B")
    with pytest.raises(KeyError):
        block.find("_missing")
    with pytest.raises(ValueError, match="loop column"):
        block.value("_atom_site_label")
    with pytest.raises(ValueError, match="not a loop column"):
        block.column("_title")
    with pytest.raises(ValueError, match="not a number"):
        block.number("_title")
    with pytest.raises(ValueError, match="numbers"):
        block.numbers("_cell_length_a", "_atom_site_fract_x")
    with pytest.raises(ValueError, match="not a number"):
        block.numbers("_atom_site_label")
    (repeated,) = _read("data_x\n_a 1\n_A 2\n")
    with pytest.raises(ValueError, match="occurs 2 times"):
        repeated.find("_a")
    frame = CIFSaveFrame("f", (CIFDataItem("_a", DecimalScalar("1")),))
    assert frame.number("_A").token == "1"


def test_text_fields_keep_raw_text() -> None:
    (block,) = read_cif(LOSSLESS / "folded.cif", lossless=True)
    assert block.value("_t") == CIFTextField("\\\nabc\\\ndef\nghi")
    assert block.value("_u") == CIFTextField("  \nblank-ish first line")
    legacy = _legacy_block(block, structural_only=False)
    assert legacy == {"t": "abcdef\nghi", "u": "blank-ish first line"}
    assert legacy == read_cif(LOSSLESS / "folded.cif")[0][0][1]
    (inline,) = _read("data_x\n_a\n;first line\nsecond\n; # after the field\n_b 1\n")
    assert inline.items[0] == CIFDataItem("_a", CIFTextField("first line\nsecond"), ((1, C(" after the field", True)),))
    _assert_round_trip([block], False)
    _assert_round_trip([inline], False)


def test_cif2_lists_tables_and_triple_quotes() -> None:
    (block,) = read_cif(LOSSLESS / "cif2.cif", lossless=True)
    assert _cif_mode(block) == "cif2"
    one, two_fifty = DecimalScalar("1"), DecimalScalar("2.50")
    assert block.value("_list") == (one, two_fifty, ("a", "b"), DecimalGap.UNKNOWN)
    assert block.value("_table") == CIFTable(
        (
            ("k1", DecimalScalar("1.0")),
            ("k 2", ("x", "y")),
            ("k3", CIFTable((("n", DecimalGap.INAPPLICABLE),))),
        )
    )
    assert block.value("_tq") == "line one\nline \"two\" and 'three'\n"
    assert block.value("_tq2") == "single"
    assert block.value("_s") == "it's"
    assert block.column("_b") == ((one, DecimalScalar("2")), CIFTable((("z", DecimalGap.UNKNOWN),)))
    _assert_round_trip([block], False)


def test_cif2_comment_inside_a_compound_value_moves_after_it() -> None:
    (block,) = _read(CIF2 + "data_x\n_a [1 # inside\n2] # after\n_b 3\n")
    assert block.items == (
        CIFDataItem("_a", (DecimalScalar("1"), DecimalScalar("2")), ((1, C(" after", True)),)),
        C(" inside"),
        CIFDataItem("_b", DecimalScalar("3")),
    )
    _assert_round_trip([block], False)


def test_mcif_syntax_is_cif1_plus_lists() -> None:
    (brackets,) = read_cif(LOSSLESS / "mcif_brackets.mcif", allow_cif2=True, lossless=True)
    assert _cif_mode(brackets) == "mcif"  # despite the CIF2 header
    assert brackets.value("_word") == "x[1]"
    assert brackets.value("_q") == "it's"
    assert brackets.value("_parent_propagation_vector.kxkykz") == ((DecimalScalar("0.125"), DecimalScalar("0"), "1/3"),)
    assert brackets.value("_inlist") == ("a", "b'", "c d")
    assert brackets.column("_v") == ((DecimalScalar("1"), DecimalScalar("2")),)
    (nolists,) = read_cif(LOSSLESS / "mcif_nolists.mcif", allow_cif2=True, lossless=True)
    assert nolists.value("_word") == "a{b}"
    (table,) = read_cif(LOSSLESS / "mcif_table.mcif", allow_cif2=True, lossless=True)
    assert table.value("_t") == "{'k':1}"
    for path, mcif in ((LOSSLESS / "mcif_triple.mcif", True), (LOSSLESS / "cif1_triple.cif", False)):
        (triple,) = read_cif(path, allow_cif2=mcif, lossless=True)
        assert triple.value("_a") == "''abc''"
    for block in (brackets, nolists, table):
        _assert_round_trip([block], True)


def test_tables_and_lists_are_checked_per_mode() -> None:
    table = CIFDataItem("_t", CIFTable((("k", DecimalScalar("1")),)))
    listed = CIFDataItem("_l", ("a", ("b",)))
    with pytest.raises(ValueError, match="tables need"):
        CIFBlock("x", (table,), dialect="mcif", header=(C(r"\#CIF_2.0"),))
    with pytest.raises(ValueError, match="tables need"):
        CIFBlock("x", (table,))
    with pytest.raises(ValueError, match="lists need"):
        CIFBlock("x", (listed,))
    CIFBlock("x", (listed,), dialect="mcif")
    cif2 = CIFBlock("x", (table, listed), header=(C(r"\#CIF_2.0"),))
    _assert_round_trip([cif2], False)


@pytest.mark.parametrize(
    ("magic", "cif2"),
    [
        (r"\#CIF_2.0", True),
        (r"\#CIF_2.0 extra text", True),
        ("\\#CIF_2.0\tx", True),
        (r"\#CIF_2.0x", False),
        (r"\#CIF_1.1", False),
        (r" \#CIF_2.0", False),
    ],
)
def test_one_cif2_header_predicate(magic: str, cif2: bool) -> None:
    assert _is_cif2_header(magic) is cif2
    text = f"#{magic}\ndata_x\n_t {{'k':1}}\n"
    (block,) = _read(text)
    assert block.header == (C(magic),)
    assert _cif_mode(block) == ("cif2" if cif2 else "cif1")
    assert isinstance(block.value("_t"), CIFTable if cif2 else str)
    table = CIFDataItem("_t", CIFTable((("k", DecimalScalar("1")),)))
    if cif2:
        CIFBlock("x", (table,), header=(C(magic),))
    else:
        with pytest.raises(ValueError, match="tables need"):
            CIFBlock("x", (table,), header=(C(magic),))
    assert _write(block) == f"#{magic}\ndata_x\n_t {{'k':1}}\n"
    braced = CIFBlock("x", (CIFDataItem("_s", "a{b}"),), header=(C(magic),))
    assert _write(braced).endswith("_s 'a{b}'\n" if cif2 else "_s a{b}\n")
    (indented,) = _read(f"   #{magic}\ndata_x\n")  # an indented magic counts too
    assert _cif_mode(indented) == _cif_mode(block)


@pytest.mark.parametrize(
    ("text", "mcif", "message"),
    [
        ("data_x\n_a\n;abc\n", False, "unterminated CIF text field starting at line 3"),
        (CIF2 + "data_x\n_a '''abc\n", False, "unterminated CIF2 triple-quoted string starting at line 3"),
        ("data_x\nsave_f\n_a 1\n", False, "unterminated CIF save frame starting at line 2"),
        (CIF2 + "data_x\n_a 'abc\n", False, "unterminated CIF quoted string at line 3"),
        ("data_x\nloop_\n_a\n_b\n1 2 3\n", False, "has 3 values for 2 data names"),
        ("data_x\n_a\n\n_b 1\n", False, "'_a' has no value at line 2"),
        ("data_x\n_a\nloop_\n_b\n1\n", False, "'_a' has no value at line 2"),
        ("data_x\n_a 'x y' b\n", False, "'b' without a data name at line 2"),
        ("data_x\n_a 1\n.\n", False, "'.' without a data name at line 3"),
        ("junk\ndata_x\n_a 1\n", False, "content before the first data_ at line 1"),
        ("data_\n_a 1\n", False, "empty CIF data block name at line 1"),
        ("data_x\nglobal_\n", False, "reserved word 'global_' at line 2"),
        ("data_x\n_a stop_\n", False, "reserved word 'stop_' used as a value at line 2"),
        ("data_x\n_a [STOP_]\n", True, "reserved word 'STOP_' used as a value at line 2"),
        ("data_x\nsave_f\ndata_y\n", False, "data_ inside a save frame at line 3"),
        ("data_x\nsave_\n", False, "save_ without an open save frame at line 2"),
        ("data_x\nloop_\nloop_\n_a\n1\n", False, "loop_ without data names at line 2"),
        ("data_x\n_a [1 2\n", True, "unterminated CIF list or table starting at line 2"),
        (CIF2 + "data_x\n_a {1:2}\n", False, "table key expected at line 3"),
        (CIF2 + "data_x\n_a {'k':1 'k':2}\n", False, "repeated CIF2 table key 'k' at line 3"),
        (CIF2 + "data_x\n_a ]\n", False, "unexpected CIF token ']' at line 3"),
        ("data_x\n_a x 'p' q \"r\" s\n", False, "cannot be written in CIF1 syntax (line 2)"),
        ("data_x\n_a x 'p _q'\n", False, "would split a quoted string"),
    ],
)
def test_reader_errors(text: str, mcif: bool, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        _read(text, mcif)


@pytest.mark.parametrize(
    ("text", "pragmatic"),
    [
        ("data_x\n_a 'x y' b\n_c 1\n", True),
        ("data_x\n_a 1\n.\n", True),
        ("data_x\n_a\n\n_b 1\n", True),
        ("data_x\n_a\nloop_\n_b\n1\n", True),
        ("data_x\n_a stop_\n", True),
        ("data_x\n_a\n;abc\n", True),
        ("junk\ndata_x\n_a 1\n", True),
        ("data_x\n_a 'abc def\n", False),
        ("data_x\n_a x 'p' q \"r\" s\n", True),
    ],
)
def test_d8_inputs_the_fast_reader_accepts_are_rejected(text: str, pragmatic: bool) -> None:
    read_cif(io.StringIO(text), pragmatic=pragmatic)
    with pytest.raises(ValueError):
        _read(text, pragmatic=pragmatic)


def test_d8_inputs_the_fast_reader_misreads() -> None:
    hashed = "data_x\nloop_\n_atom_site_label\n_atom_site_fract_x\nC1# 0.1\n"
    (block,) = _read(hashed)
    assert block.column("_atom_site_label") == ("C1#",)
    assert read_cif(io.StringIO(hashed))[0][0][1]["atom_site_label"] == ["C1"]
    quoted = "data_x\n_l [a b' 'c d']\n"
    (block,) = _read(quoted, True)
    assert block.value("_l") == ("a", "b'", "c d")
    assert read_cif(io.StringIO(quoted), allow_cif2=True)[0][0][1]["l"] != ["a", "b'", "c d"]


ORDERED = (FIXTURES / "cif_bonds" / "ordered.cif").read_text()


def _loads_like_ordered(text: str) -> None:
    """``text`` reads like ``ordered.cif`` in the fast reader, the lossless model and ``load()``."""
    (block,) = _read(text)
    assert read_cif(io.StringIO(text))[0][0][1] == _legacy_block(block, structural_only=False)
    assert read_cif(io.StringIO(text))[0][0][1] == read_cif(io.StringIO(ORDERED))[0][0][1]
    structure = httk.core.load(io.StringIO(text), format="cif")
    ordered = httk.core.load(FIXTURES / "cif_bonds" / "ordered.cif")
    assert (
        structure == ordered
        and structure.bonds == ordered.bonds
        and structure.bonds is not None
        and structure.bonds.pairs
    )


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("loop_\n_atom_site_label\n", "loop_ # sites\n_atom_site_label\n"),
        ("_atom_site_label\n", "_atom_site_label # label\n"),
        ("_atom_site_label\n", "_atom_site_label\n# own line\n\n"),
        ("_geom_bond_distance\n", "_geom_bond_distance #d\n"),
        ("loop_\n_space_group_symop_operation_xyz\n", "loop_\n#c\n_space_group_symop_operation_xyz #c\n#c\n"),
    ),
)
def test_fp1_a_comment_in_a_loop_header_is_ignored(old: str, new: str) -> None:
    _loads_like_ordered(ORDERED.replace(old, new, 1))


def test_fp1_the_comments_fixture_reads_alike_in_both_readers() -> None:
    (block,) = read_cif(LOSSLESS / "comments.cif", lossless=True)
    assert read_cif(LOSSLESS / "comments.cif")[0][0][1] == _legacy_block(block, structural_only=False)


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("_atom_site_label\n_atom_site_type_symbol\n", "_atom_site_label _atom_site_type_symbol\n"),
        ("loop_\n_geom_bond_atom_site_label_1\n", "loop_ _geom_bond_atom_site_label_1\n"),
        ("_atom_site_calc_flag\nFe1", "_atom_site_calc_flag Fe1"),
        ("_geom_bond_site_symmetry_2\nFe1 O1", "_geom_bond_site_symmetry_2 Fe1 # c\nO1"),
        ("_space_group_symop_operation_xyz\nx,y,z", "_space_group_symop_operation_xyz 'x,y,z'"),
    ),
)
def test_fp2_a_loop_header_line_is_split_into_tokens(old: str, new: str) -> None:
    _loads_like_ordered(ORDERED.replace(old, new, 1))


def test_fp2_a_value_after_a_loop_name_starts_the_values() -> None:
    # structreading/111: a one-name loop, with its value on the name's line, before a data item
    path = FIXTURES / "structreading" / "111.cif"
    fast = read_cif(path)[0][0][1]
    assert fast["publ_author_name"] == ["Ketelaar, J A A"]
    assert fast["journal_name_full"].startswith("Zeitschrift fuer Kristallographie")
    text = "data_x\nloop_ _a 1 _b\n2\n"
    assert read_cif(io.StringIO(text))[0][0][1] == {"loop_0": ["a"], "a": ["1"], "b": "2"}


def test_pragmatic_join_d1() -> None:
    text = "data_x\n_a x  y 'z w'\n_b x y#c d\n_c 1.5 2 # note\n"
    (block,) = _read(text)
    assert block.items == (
        CIFDataItem("_a", "x  y 'z w'"),
        CIFDataItem("_b", "x y", ((1, C("c d", True)),)),
        CIFDataItem("_c", "1.5 2", ((1, C(" note", True)),)),
    )
    fast = read_cif(io.StringIO(text))[0][0][1]
    assert _legacy_block(block, structural_only=False) == fast
    written = _assert_round_trip([block], False)
    assert "\n_a 'x  y 'z w''\n_b 'x y' #c d\n" in written
    with pytest.raises(ValueError, match="without a data name"):
        _read(text, pragmatic=False)


def test_pragmatic_unterminated_quote_d9() -> None:
    text = "data_x\n_a \"M. J. M. Leask   \n_b 'M. Duc Le\",\n"
    (block,) = _read(text)
    assert block.value("_a") == '"M. J. M. Leask'
    assert block.value("_b") == "'M. Duc Le\","
    assert _legacy_block(block, structural_only=False) == read_cif(io.StringIO(text))[0][0][1]
    written = _assert_round_trip([block], False)
    assert "_a '\"M. J. M. Leask'\n" in written
    with pytest.raises(ValueError, match="unterminated CIF quoted string at line 2"):
        _read(text, pragmatic=False)
    (mcif,) = _read(text, True)
    assert mcif.value("_a") == '"M. J. M. Leask'


@pytest.mark.parametrize(
    "raw",
    [b"\xef\xbb\xbfdata_bom\r\n_a 1\r\n_b 'x'\r\n", b"data_bom\r_a 1\r_b 'x'\r", b"data_bom\n_a 1\n_b 'x'"],
)
def test_bom_and_line_ends_are_normalised(raw: bytes, tmp_path: Path) -> None:
    expected = [CIFBlock("bom", (CIFDataItem("_a", DecimalScalar("1")), CIFDataItem("_b", "x")))]
    assert read_cif(io.BytesIO(raw), lossless=True) == expected
    path = tmp_path / "bom.cif"
    path.write_bytes(raw)
    assert read_cif(path, lossless=True) == expected
    assert _read(raw.decode("utf-8-sig").splitlines()) == expected


def test_latin1_only_under_repair(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    raw = "data_x\n_author 'Müller'\n".encode("latin-1")
    path = tmp_path / "latin.cif"
    path.write_bytes(raw)
    for source in (path, io.BytesIO(raw)):
        with pytest.raises(UnicodeDecodeError):
            read_cif(source, lossless=True)
    with caplog.at_level(logging.WARNING):
        (block,) = read_cif(path, lossless=True, repair=True)
        (stream,) = read_cif(io.BytesIO(raw), lossless=True, repair=True)
    assert block.value("_author") == stream.value("_author") == "Müller"
    assert "decoded it as Latin-1 under repair=True" in caplog.text
    assert _write(block).encode("utf-8").count("Müller".encode()) == 1


def test_lossless_excludes_structural_only() -> None:
    with pytest.raises(TypeError, match="structural_only"):
        read_cif(io.StringIO("data_x\n"), lossless=True, structural_only=True)  # type: ignore[call-overload]


@pytest.mark.parametrize("mcif", [False, True])
def test_written_strings_read_back_through_the_fast_reader(mcif: bool) -> None:
    strings = [
        "a#b",
        "#",
        "a b",
        "  lead",
        "x]",
        "]",
        "[x",
        "it's",
        "say \"hi\"",
        "_x",
        "$x",
        ";x",
        "data_x",
        "?",
        "1.5",
    ]
    block = CIFBlock(
        "x",
        (
            *(CIFDataItem(f"_s{i}", value) for i, value in enumerate(strings)),
            CIFLoop(("_l", "_m"), tuple((value, value) for value in strings)),
        ),
        dialect="mcif" if mcif else "cif",
    )
    text = _assert_round_trip([block], mcif)
    fast = read_cif(io.StringIO(text), allow_cif2=mcif)[0][0][1]
    assert fast == _legacy_block(block, structural_only=False)


def test_unwritable_strings_are_rejected_at_construction() -> None:
    CIFBlock("x", (CIFDataItem("_a", "p' q\""),))
    with pytest.raises(ValueError, match="cannot be written in CIF1 syntax"):
        CIFBlock("x", (CIFDataItem("_a", "p' q\" r"),))
    with pytest.raises(ValueError, match="cannot be written in magnetic CIF syntax"):
        CIFBlock("x", (CIFDataItem("_a", "two\nlines"),), dialect="mcif")
    cif2 = CIFBlock("x", (CIFDataItem("_a", "two\nlines 'and' \"quotes\""),), header=(C(r"\#CIF_2.0"),))
    _assert_round_trip([cif2], False)
    with pytest.raises(ValueError, match="cannot be written in CIF2 syntax"):
        CIFBlock("x", (CIFDataItem("_a", "a\rb"),), header=(C(r"\#CIF_2.0"),))


def test_item_invariants() -> None:
    with pytest.raises(ValueError, match="data name"):
        CIFDataItem("a", "x")
    with pytest.raises(ValueError, match="data name"):
        CIFDataItem("_", "x")
    with pytest.raises(ValueError, match="without whitespace"):
        CIFLoop(("_a b",), ())
    with pytest.raises(ValueError, match="without whitespace"):
        CIFBlock("", ())
    with pytest.raises(ValueError, match="at least one data name"):
        CIFLoop((), ())
    with pytest.raises(ValueError, match="needs 2 values"):
        CIFLoop(("_a", "_b"), (("1",),))
    with pytest.raises(ValueError, match="trailing whitespace"):
        C("text ")
    with pytest.raises(ValueError, match="line break"):
        C("a\nb")
    with pytest.raises(ValueError, match="line starting with ';'"):
        CIFTextField("a\n;b")
    with pytest.raises(ValueError, match="DecimalGap value"):
        CIFDataItem("_a", DecimalScalar("?"))
    with pytest.raises(TypeError, match="a CIF value"):
        CIFDataItem("_a", 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        CIFTable((("k", "1"), ("k", "2")))
    with pytest.raises(ValueError, match="own-line"):
        CIFDataItem("_a", "x", ((1, C("own")),))
    with pytest.raises(ValueError, match="must come first"):
        CIFDataItem("_a", "x", ((0, C("own")), (0, C("in", True))))
    with pytest.raises(ValueError, match="must not decrease"):
        CIFDataItem("_a", "x", ((1, C("in", True)), (0, C("own"))))
    with pytest.raises(ValueError, match="outside 0..2"):
        CIFLoop(("_a",), (("x",),), ((3, C("in", True)),))
    with pytest.raises(ValueError, match="inline CIF comment item"):
        CIFBlock("x", (CIFDataItem("_a", "x"), C("in", True)))
    with pytest.raises(ValueError, match="own-line CIFComments"):
        CIFBlock("x", (), leading=(C("in", True),))
    for after in ((CIFDataItem("_b", "x"),), (C("own"),)):
        with pytest.raises(ValueError, match="without rows cannot be followed"):
            CIFBlock("x", (CIFLoop(("_a",), ()), *after))
    with pytest.raises(ValueError, match="dialect"):
        CIFBlock("x", (), dialect="cif2")  # type: ignore[arg-type]


def test_save_frames_and_empty_loops_round_trip() -> None:
    text = (
        "data_x\nsave_outer # first in frame\n_a 1\nsave_inner\nloop_\n_b\nsave_\n# frame end\nsave_ # after frame\n"
        "loop_\n_c\nloop_\n_d\n1\n"
    )
    (block,) = _read(text)
    inner = CIFSaveFrame("inner", (CIFLoop(("_b",), ()),))
    outer = CIFSaveFrame(
        "outer", (C(" first in frame", True), CIFDataItem("_a", DecimalScalar("1")), inner, C(" frame end"))
    )
    assert block.items == (
        outer,
        C(" after frame", True),
        CIFLoop(("_c",), ()),
        CIFLoop(("_d",), ((DecimalScalar("1"),),)),
    )
    _assert_round_trip([block], False)
    flat = "data_x\n_a 1\nsave_f\n_b 2\nsave_g\n_a 3\nsave_\nsave_\n"
    (block,) = _read(flat)
    assert _legacy_block(block, structural_only=False) == read_cif(io.StringIO(flat))[0][0][1] == {"a": "3", "b": "2"}


def test_loop_layout_wraps_rows_and_keeps_text_fields() -> None:
    values = tuple((f"label{i}", DecimalScalar("0.123456789")) for i in range(3))
    loop = CIFLoop(("_a", "_b"), (*values, (CIFTextField("multi\nline"), DecimalScalar("1"))))
    block = CIFBlock("x", (loop,))
    text = _write(block)
    assert "\nlabel0 0.123456789\nlabel1 0.123456789\n" in text
    assert "\n;multi\nline\n; 1\n" in text
    out = io.StringIO()
    write_cif(out, block, max_line_length=10)
    assert "\nlabel0\n0.123456789\n" in out.getvalue()
    assert _read(out.getvalue()) == [block]


def test_repr_round_trips_through_the_constructor() -> None:
    (block,) = read_cif(LOSSLESS / "cif2.cif", lossless=True)
    namespace = {
        "CIFBlock": CIFBlock,
        "CIFComment": CIFComment,
        "CIFDataItem": CIFDataItem,
        "CIFLoop": CIFLoop,
        "CIFTable": CIFTable,
        "CIFTextField": CIFTextField,
        "DecimalGap": DecimalGap,
        "DecimalScalar": DecimalScalar,
    }
    assert eval(repr(block), namespace) == block
    assert hash(block) == hash(eval(repr(block), namespace))


def test_nesting_is_capped_at_64() -> None:
    deep_list = "[" * 64 + "]" * 64
    (block,) = _read(f"data_x\n_a {deep_list}\n", True)
    _assert_round_trip([block], True)
    for text, mcif in (
        (f"data_x\n_a {'[' * 65}{']' * 65}\n", True),
        (f"data_x\n_a {'[' * 1000}\n", True),
        (CIF2 + "data_x\n_a " + "{'k':" * 1000 + "\n", False),
        ("data_x\n" + "save_f\n" * 2000, False),
    ):
        with pytest.raises(ValueError, match="CIF lists, tables or save frames nested deeper than 64 at line"):
            _read(text, mcif)
    frames = "".join(f"save_f{i}\n" for i in range(63)) + "_a [1]\n" + "save_\n" * 63
    _read(f"data_x\n{frames}", True)
    with pytest.raises(ValueError, match="nested deeper than 64 at line 66"):
        _read("data_x\n" + "".join(f"save_f{i}\n" for i in range(64)) + "_a [1]\n" + "save_\n" * 64, True)
    value: Any = DecimalScalar("1")
    for _ in range(64):
        value = (value,)
    CIFDataItem("_a", value)
    with pytest.raises(ValueError, match="nested deeper than 64"):
        CIFDataItem("_a", (value,))
    table: Any = DecimalScalar("1")
    for _ in range(64):
        table = CIFTable((("k", table),))
    with pytest.raises(ValueError, match="nested deeper than 64"):
        CIFTable((("k", table),))
    frame = CIFSaveFrame("f", ())
    for _ in range(63):
        frame = CIFSaveFrame("f", (frame,))
    with pytest.raises(ValueError, match="nested deeper than 64"):
        CIFSaveFrame("f", (frame,))
    with pytest.raises(ValueError, match="nested deeper than 64"):
        CIFSaveFrame("f", (CIFDataItem("_a", value),))


def test_pragmatic_join_is_linear_in_whitespace() -> None:
    text = "data_x\n_a x" + " " * 200_000 + "y _b 1\n"
    start = time.perf_counter()
    (block,) = _read(text)
    assert time.perf_counter() - start < 2
    assert block.value("_a") == "x" + " " * 200_000 + "y" and block.value("_b") == DecimalScalar("1")
    start = time.perf_counter()
    fast = read_cif(io.StringIO(text))[0][0][1]
    assert time.perf_counter() - start < 2  # FP3: the fast reader's split took minutes here
    assert fast == _legacy_block(block, structural_only=False)


def test_a_joined_value_leaves_the_name_or_keyword_that_ends_it() -> None:
    # The fast reader used to drop the '_', 'data_' or 'loop_' that ended a joined value.
    text = "data_x\n_a x y _b 1\n_c x y loop_ _d\n1\n_e x # y _f 2\n_g p q data_z\n_h 1\n"
    fast = read_cif(io.StringIO(text))[0]
    assert fast == [
        ("x", {"a": "x y", "b": "1", "c": "x y", "loop_0": ["d"], "d": ["1"], "e": "x", "g": "p q"}),
        ("z", {"h": "1"}),
    ]
    assert [(block.name, _legacy_block(block, structural_only=False)) for block in _read(text)] == fast


def test_a_joined_value_ending_in_an_empty_block_code_raises() -> None:
    for text in ("data_x\n_a x y data_\n", "data_\n_a 1\n"):
        with pytest.raises(ValueError, match="empty CIF data block name"):
            read_cif(io.StringIO(text))
        with pytest.raises(ValueError, match="empty CIF data block name"):
            _read(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("data_x\nloop_\n_a\n'x y' _b\n2\n", {"loop_0": ["a"], "a": ["x y"], "b": "2"}),
        ("data_x\nloop_\n_a\n;\nt\n; _b 2\n", {"loop_0": ["a"], "a": ["t"], "b": "2"}),
        ("data_x\nloop_ _a 'x y' _b\n2\n", {"loop_0": ["a"], "a": ["x y"], "b": "2"}),
        (  # a skipped loop: the structural item after its quoted value is kept
            "data_x\nloop_\n_publ_author_name\n'Smith, J' _cell_length_a 5.0\n",
            {"loop_0": ["publ_author_name"], "publ_author_name": ["Smith, J"], "cell_length_a": "5.0"},
        ),
        (
            "data_x\nloop_\n_publ_author_name\n;\nSmith\n; _cell_length_a 5.0\n",
            {"loop_0": ["publ_author_name"], "publ_author_name": ["Smith"], "cell_length_a": "5.0"},
        ),
    ),
)
def test_fp4_a_data_name_after_a_quoted_value_or_text_field_ends_the_loop(text: str, expected: dict[str, Any]) -> None:
    (block,) = _read(text)
    assert read_cif(io.StringIO(text))[0][0][1] == expected == _legacy_block(block, structural_only=False)
    structural = read_cif(io.StringIO(text), structural_only=True)[0][0][1]
    assert structural == _legacy_block(block, structural_only=True)
    assert structural.get("cell_length_a") == expected.get("cell_length_a")


def test_cif1_quotes_before_brackets_are_ordinary() -> None:
    text = "data_x\n_a 'a'] b\" c'\n_b 'p\" q'] r\n"
    (block,) = _read(text)
    assert block.value("_a") == "a'] b\" c"
    assert block.value("_b") == "'p\" q'] r"  # D9 with both quote kinds followed by space
    assert _legacy_block(block, structural_only=False) == read_cif(io.StringIO(text))[0][0][1]
    _assert_round_trip([block], False)
    with pytest.raises(ValueError, match="magnetic CIF syntax"):
        CIFBlock("x", (CIFDataItem("_a", "a'] b\" c"),), dialect="mcif")


@pytest.mark.parametrize(
    "text",
    [
        "data_x\nloop_\n_a\n# own\nloop_\n_b\n1\n",
        "data_x\nloop_\n_a\n# own\n",
        "data_x\nsave_f\nloop_\n_a\n# own\nsave_\n",
    ],
)
def test_empty_loop_followed_by_a_comment_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match=r"CIF loop_ at line \d has no values"):
        _read(text)


def test_empty_loop_before_the_next_blocks_leading_comments() -> None:
    blocks = _read("data_x\nloop_\n_a\n#lead\ndata_y\n_b 1\n")
    assert blocks[0].items == (CIFLoop(("_a",), ()),)
    assert blocks[1].leading == (C("lead"),)
    _assert_round_trip(blocks, False)


def test_cif2_names_end_only_at_whitespace() -> None:
    (block,) = _read(CIF2 + "data_b[1]\n_a[1] 2\nsave_f{x}\n_c [3]\nsave_\n")
    assert block.name == "b[1]"
    assert block.items == (
        CIFDataItem("_a[1]", DecimalScalar("2")),
        CIFSaveFrame("f{x}", (CIFDataItem("_c", (DecimalScalar("3"),)),)),
    )
    _assert_round_trip([block], False)
