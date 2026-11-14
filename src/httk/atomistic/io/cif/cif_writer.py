#
#    The high-throughput toolkit (httk)
#    Copyright (C) 2012-2025 The httk AUTHORS
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import io
import os
import re
import sys
import warnings
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any, cast

from httk.core import DecimalGap, DecimalScalar

from .cif_items import (
    CIFBlock,
    CIFComment,
    CIFDataItem,
    CIFItem,
    CIFLoop,
    CIFTextField,
    CIFValue,
    _cif_mode,
    _cif_quote,
    _Mode,
)

_cif_ordinary_char = r'!%&()*+,-./0123456789:<=>?@ABCDEFGHIHJKLMNOPQRSTUVWXYZ\^`abcdefghijklmnopqrstuvwxyz{|}~'
_cif_non_blank_char = _cif_ordinary_char + '"' + "#$" + "'" + "_" + ";[]"
_cif_text_lead_char = _cif_ordinary_char + '"' + "#$" + "'" + "_ \t[]"
_cif_any_print_char = _cif_ordinary_char + '"' + "#$" + "'" + "_ \t;[]"
_cif_non_blank_char_table = str.maketrans(_cif_non_blank_char, _cif_non_blank_char)
_cif_unicode_translation_table: dict[int, int | None] = {}
for i in range(sys.maxunicode + 1):
    _cif_unicode_translation_table[i] = None
for key, value in _cif_non_blank_char_table.items():
    _cif_unicode_translation_table[key] = value
_cif_integer_regex = re.compile(r'^[+-]?[0-9]+$')
_cif_float_regex = re.compile(r'^[+-]?[0-9]+[eE][+-]?[0-9]+|([+-]?[0-9]*\.[0-9]+|[+-]?[0-9]\.)([eE][+-]?[0-9]+)?$')
_cif_simplestring_regex = re.compile(r'^[A-Za-z0-9()][A-Za-z0-9()+-]*$')
# Lossy exact values remain recoverable through the local companion tag.
_HTTK_CIF_DECIMAL_DIGITS = 16


class _Bare(str):
    """A value written unquoted as it is, such as the CIF inapplicable marker ``.``."""


_INAPPLICABLE = _Bare(".")


def _cif_validate_name(name_unfiltered, context=None):
    if context is not None:
        context = context + ": " + name_unfiltered
    name = _cif_validate_non_blank_char(name_unfiltered, context)
    if len(name) > 75:
        raise ValueError(f"CIF data name exceeds 75 characters: {name_unfiltered}")
    return name


def _cif_is_float(data_value):
    return _cif_float_regex.match(data_value) is not None


def _cif_is_simplestring(data_value):
    return _cif_simplestring_regex.match(data_value) is not None


def _cif_is_int(data_value):
    return _cif_integer_regex.match(data_value) is not None


def _cif_validate_non_blank_char(s, context=None):
    out = s.translate(_cif_unicode_translation_table)
    if out != s:
        if context is not None:
            warnings.warn(f"write_cif removed non-permitted characters in {context}", RuntimeWarning, stacklevel=2)
        else:
            warnings.warn("write_cif removed non-permitted characters", RuntimeWarning, stacklevel=2)
    return out


def _cif_write_semicolontextfield(f, lines, noteol, max_line_length):
    if noteol:
        f.write("\n")
        noteol = False
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip("\r\n")
        if lines[i] and lines[i][0] == ';':
            warnings.warn(
                "write_cif inserted a space before a semicolon-leading text line", RuntimeWarning, stacklevel=2
            )
            lines[i] = ' ' + lines[i]
        if len(lines[i]) > max_line_length:
            f.write(";\\" + "\n")
            break
    else:
        f.write(";")
    for line in lines:
        if len(line) > max_line_length:
            sublines = [line[i : i + max_line_length - 2] for i in range(0, len(line), max_line_length - 2)]
            # Handle a wonderful corner case: the line splitting for length creates lines that start with one, or more, semi-colons..., sigh...
            for i in range(1, len(sublines)):
                if sublines[i][0] == ";":
                    if len(sublines[i]) > 1 and sublines[i][1] != ";":
                        # If its just a single semi-colon, move it to the previous line, which we saved space for by splitting at max_line_length-2
                        sublines[i - 1] += ";"
                        sublines[i] = sublines[i][1:]
                    else:
                        # Multiple semi-colons in a row, or a semi-colon + newline, this is a possibly unresolvable case (think long string of only semi-colons)
                        # fudge a solution by inserting a space
                        warnings.warn(
                            "write_cif inserted a space before a semicolon in a long text line",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        sublines[i] = " " + sublines[i]
            for subline in sublines:
                f.write(subline + "\\" + "\n")
        else:
            f.write(line + "\n")

    f.write(";\n")
    return False


def _cif_write_data_value(f, orig_data_value, noteol, max_line_length, inloop):
    if orig_data_value is None:
        f.write("?")
        return True
    elif isinstance(orig_data_value, _Bare):
        f.write(orig_data_value)
        return True
    else:
        data_value = str(orig_data_value)
    has_whitespace = len(data_value.split()) > 1
    lines = data_value.splitlines()
    has_lines = len(lines) > 1
    has_single_quote = data_value.find("'") != -1
    has_double_quote = data_value.find('"') != -1
    too_long = len(data_value) + 2 > max_line_length
    if has_lines or (has_single_quote and has_double_quote) or too_long:
        noteol = _cif_write_semicolontextfield(f, lines, noteol, max_line_length)
        return noteol
    elif has_double_quote or (has_whitespace and not has_single_quote) or data_value == "":
        f.write("'" + data_value + "'")
        return True
    elif has_single_quote or (has_whitespace and not has_double_quote):
        f.write('"' + data_value + '"')
        return True
    elif _cif_is_float(data_value) or _cif_is_int(data_value) or inloop and _cif_is_simplestring(data_value):
        f.write(data_value)
        return True
    else:
        f.write("'" + data_value + "'")
        return True


def is_sequence(val: object) -> bool:
    """Report whether a value should be written as a CIF loop or sequence.

    :param val: Value to classify, treating strings as scalar values.
    :return: ``True`` for iterable non-string values.
    """
    return isinstance(val, Iterable) and not isinstance(val, str)


def _cif_write_data_block(f, data_block, max_line_length):
    for key in data_block:
        val = data_block[key]
        if key.startswith("loop_"):
            f.write("loop_\n")
            outdata_columns = []
            for unfiltered_column in val:
                column = _cif_validate_name(unfiltered_column, "column name: " + unfiltered_column)
                f.write("_" + column + "\n")
                outdata_columns += [data_block[unfiltered_column]]
            if len(outdata_columns) > 0:
                noteol = False
                for i in range(len(outdata_columns[0])):
                    column_count = 0
                    for j in range(len(outdata_columns)):
                        column_count += len(str(outdata_columns[j][i])) + 2
                        if column_count > max_line_length and noteol:
                            f.write("\n")
                            column_count = 0
                            noteol = False
                        noteol = _cif_write_data_value(f, outdata_columns[j][i], noteol, max_line_length, inloop=True)
                        if noteol:
                            f.write(" ")
                            column_count += 1
                        else:
                            column_count = 0
                    if noteol:
                        noteol = False
                        f.write("\n")
        elif is_sequence(val):
            continue
        else:
            data_name = _cif_validate_name(key)
            # Do we have space _ + key + space + quote + the whole data value + quote?, if not, preemptively break line
            f.write("_" + data_name + " ")
            if len(data_name) + len(str(val)) + 4 > max_line_length:
                f.write("\n")
                noteol = False
            else:
                noteol = True
            noteol = _cif_write_data_value(f, val, noteol, max_line_length, inloop=False)
            if noteol:
                f.write("\n")
                noteol = False


def _put_comment(out: list[str], comment: CIFComment) -> None:
    if comment.inline:
        out[-1] += " #" + comment.text
    else:
        out.append("#" + comment.text)


def _put_value(out: list[str], text: str, new_line: bool) -> None:
    """Append a rendered value, on a new line or after the current one."""
    if text.startswith("\n"):  # a text field: its ';' lines stand alone
        out += text[1:].split("\n")
        return
    parts = text.split("\n")
    if new_line:
        out.append(parts[0])
    else:
        out[-1] += " " + parts[0]
    out += parts[1:]


def _lossless_value(value: CIFValue, mode: _Mode) -> str:
    """Render a value; a text field starts with a line break so its ';' lands in column 0."""
    if isinstance(value, str):
        return _cif_quote(value, mode)
    if isinstance(value, DecimalScalar):
        return value.token
    if isinstance(value, DecimalGap):
        return value.value
    if isinstance(value, CIFTextField):
        return "\n;" + value.text + "\n;"
    if isinstance(value, tuple):
        return "[" + " ".join(_lossless_value(entry, mode) for entry in value) + "]"
    return (
        "{"
        + " ".join(_cif_quote(key, mode, key=True) + ":" + _lossless_value(entry, mode) for key, entry in value.entries)
        + "}"
    )


def _write_lossless_loop(out: list[str], loop: CIFLoop, mode: _Mode, max_line_length: int) -> None:
    placed: dict[int, list[CIFComment]] = {}
    for position, comment in loop.comments:
        placed.setdefault(position, []).append(comment)

    def put(position: int) -> bool:
        for comment in placed.get(position, ()):
            _put_comment(out, comment)
        return position in placed

    out.append("loop_")
    put(0)
    for index, name in enumerate(loop.names):
        out.append(name)
        put(index + 1)
    width = len(loop.names)
    fresh = True  # the next value starts a line
    for index, value in enumerate(entry for row in loop.rows for entry in row):
        text = _lossless_value(value, mode)
        first = text.partition("\n")[0]
        _put_value(out, text, fresh or index % width == 0 or len(out[-1]) + 1 + len(first) > max_line_length)
        fresh = put(width + index + 1)


def _write_lossless_items(out: list[str], items: tuple[CIFItem, ...], mode: _Mode, max_line_length: int) -> None:
    for item in items:
        if isinstance(item, CIFComment):
            _put_comment(out, item)
        elif isinstance(item, CIFDataItem):
            out.append(item.name)
            before = [comment for position, comment in item.comments if position == 0]
            for comment in before:
                _put_comment(out, comment)
            _put_value(out, _lossless_value(item.value, mode), bool(before))
            for position, comment in item.comments:
                if position == 1:
                    _put_comment(out, comment)
        elif isinstance(item, CIFLoop):
            _write_lossless_loop(out, item, mode, max_line_length)
        else:
            out.append("save_" + item.name)
            _write_lossless_items(out, item.items, mode, max_line_length)
            out.append("save_")


def _lossless_cif_text(blocks: list[CIFBlock], max_line_length: int) -> str:
    """Return the CIF text of lossless blocks (a pure function of the blocks)."""
    first = blocks[0]
    if any(block.header != first.header or block.dialect != first.dialect for block in blocks):
        raise ValueError(
            "blocks carry different file headers or dialects; use dataclasses.replace(block, header=..., dialect=...)"
        )
    for block in blocks[:-1]:
        if block.items and isinstance(block.items[-1], CIFComment) and not block.items[-1].inline:
            raise ValueError(
                f"block {block.name!r} ends with own-line comments, which would re-read as the next block's "
                "leading; move them into the next block's `leading` with dataclasses.replace"
            )
    mode = _cif_mode(first)
    out = ["#" + comment.text for comment in first.header]
    if first.leading:
        out.append("")  # ends the header
    for block in blocks:
        out += ["#" + comment.text for comment in block.leading]
        out.append("data_" + block.name)
        _write_lossless_items(out, block.items, mode, max_line_length)
    return "\n".join(out) + "\n"


def write_cif(
    destination: str | os.PathLike[str] | io.TextIOBase,
    data: CIFBlock | Iterable[CIFBlock] | Iterable[tuple[str | None, Mapping[str, object]]],
    header: str | None = None,
    max_line_length: int = 80,
) -> None:
    """Write CIF ``data`` to a path or open text stream.

    ``data`` is an iterable of ``(block_name, block)`` pairs. A block maps data names to
    scalar values and uses ``loop_N`` keys to list its loop columns. ``header``, when
    supplied, is written before the data blocks.

    ``data`` may instead be one :class:`~httk.atomistic.io.cif.CIFBlock` or an iterable of them,
    as ``read_cif(..., lossless=True)`` returns. They are written losslessly, in the syntax of
    their dialect and header, with the header (shared by all blocks) once at the top; reading the
    output back gives equal blocks. ``header`` must then be ``None``; ``max_line_length`` wraps
    loop rows.

    :param destination: Filename or open text stream receiving the CIF text.
    :param data: Data-block pairs containing scalar data names and loop columns, or CIF blocks.
    :param header: Optional text written before the data blocks.
    :param max_line_length: Maximum preferred line length for emitted values.
    :raises TypeError: If CIF blocks are mixed with pairs, or given with ``header``.
    :raises ValueError: If the blocks differ in header or dialect, or a block other than the last
        ends with an own-line comment.
    """
    entries = [data] if isinstance(data, CIFBlock) else list(data)
    if any(isinstance(entry, CIFBlock) for entry in entries):
        if not all(isinstance(entry, CIFBlock) for entry in entries):
            raise TypeError("write_cif takes CIFBlocks or (name, block) pairs, not a mixture")
        if header is not None:
            raise TypeError("CIF blocks carry their own header; write_cif(header=...) does not apply")
        text = _lossless_cif_text(cast(list[CIFBlock], entries), max_line_length)
        if isinstance(destination, (str, os.PathLike)):
            with open(destination, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        else:
            destination.write(text)
        return
    pairs = cast(list[tuple[str | None, Mapping[str, object]]], entries)

    with ExitStack() as stack:
        f: io.TextIOBase
        if isinstance(destination, (str, os.PathLike)):
            f = stack.enter_context(open(destination, "w", encoding="utf-8"))
        else:
            f = destination
        if header is not None:
            lines = header.splitlines()
            for line in lines:
                if len(line) > max_line_length:
                    header = "#\n" + header
                    break
            for line in lines:
                if len(line) > max_line_length:
                    sublines = [line[i : i + 79] for i in range(0, len(line), 79)]
                    for subline in sublines:
                        f.write(subline + "\\" + "\n")
                else:
                    f.write(line + "\n")

        data_block_count = -1
        for data_block in pairs:
            data_block_count += 1
            data_block_name_unfiltered = data_block[0]
            if data_block_name_unfiltered is None:
                data_block_name = "data_" + str(data_block_count)
            else:
                data_block_name = _cif_validate_name(data_block_name_unfiltered, "data block name")
                if data_block_name == "":
                    data_block_name = "data_" + str(data_block_count)

            f.write("data_" + data_block_name + "\n")
            _cif_write_data_block(f, data_block[1], max_line_length)


def _finite_decimal(value: Fraction) -> str | None:
    denominator = value.denominator
    while denominator % 2 == 0:
        denominator //= 2
    while denominator % 5 == 0:
        denominator //= 5
    if denominator != 1:
        return None
    places = max(
        _decimal_places(value.denominator, 2),
        _decimal_places(value.denominator, 5),
    )
    integer = value.numerator * 2 ** max(0, places - _decimal_places(value.denominator, 2))
    integer *= 5 ** max(0, places - _decimal_places(value.denominator, 5))
    sign = "-" if integer < 0 else ""
    digits = str(abs(integer)).rjust(places + 1, "0")
    if places == 0:
        return sign + digits
    return f"{sign}{digits[:-places]}.{digits[-places:]}".rstrip("0").rstrip(".")


def _decimal_places(denominator: int, factor: int) -> int:
    places = 0
    while denominator % factor == 0:
        denominator //= factor
        places += 1
    return places


def _dual_cif_value(value: object) -> tuple[object, str | None]:
    """Return a standard decimal and companion, using 16 significant digits when lossy."""
    if value is None:
        return None, None
    token = str(value).strip().strip("'\"")
    exact = Fraction(token)
    finite = _finite_decimal(exact)
    if finite is not None:
        return (finite if "/" in token or "e" in token.lower() else token), None
    with localcontext() as context:
        context.prec = _HTTK_CIF_DECIMAL_DIGITS
        decimal = Decimal(exact.numerator) / Decimal(exact.denominator)
        standard = format(decimal, f".{_HTTK_CIF_DECIMAL_DIGITS}g")
    return standard, token


def _dual_cif_column(
    values: Iterable[object], *, pad_coordinates: bool = False, decimals: int = _HTTK_CIF_DECIMAL_DIGITS
) -> tuple[list[object], list[str | None], bool]:
    standard: list[object] = []
    exact: list[str | None] = []
    for value in values:
        visible, companion = _dual_cif_value(value)
        if pad_coordinates and visible is not None and companion is None:
            visible = _pad_coordinate_decimal(visible, decimals)
        standard.append(visible)
        exact.append(companion)
    return standard, exact, any(value is not None for value in exact)


def _pad_coordinate_decimal(value: object, decimals: int = _HTTK_CIF_DECIMAL_DIGITS) -> object:
    """Give a standard coordinate decimal a fixed ``decimals``-place precision claim.

    ``decimals`` defaults to 16 (an exact structure's coordinate reads as machine-exact); a relaxed
    structure passes its precision-derived width so the digit count states the precision it has.
    """
    text = str(value)
    whole, dot, fraction = text.partition(".")
    if not dot:
        return text + "." + "0" * decimals
    return whole + "." + fraction.ljust(decimals, "0")


def _neutral_cif_block(block: Mapping[str, object], *, exact_companions: bool = False) -> dict[str, object]:
    """Turn one ``read_cif_asus`` block into the low-level writer's block shape.

    ``_httk_*_exact`` companion columns (carrying exact rational tokens such as ``1/3``) are a
    non-standard httk extension, emitted only when ``exact_companions`` is true; by default the
    block carries the standard rounded decimals alone.
    """
    raw: dict[str, object] = {}
    cell_tags = (
        "cell_length_a",
        "cell_length_b",
        "cell_length_c",
        "cell_angle_alpha",
        "cell_angle_beta",
        "cell_angle_gamma",
    )
    exact_cell = cast(Iterable[object] | None, block.get("cell_parameters_exact"))
    if exact_cell is not None:
        for tag, value in zip(cell_tags, exact_cell):
            raw[tag], exact = _dual_cif_value(value)
            if exact is not None and exact_companions:
                raw[f"httk_{tag}_exact"] = exact
    for source, target in (
        ("space_group_nbr", "space_group_IT_number"),
        ("space_group_name_hm", "space_group_name_H-M_alt"),
        ("space_group_name_hall", "space_group_name_Hall"),
    ):
        if source in block:
            raw[target] = block[source]

    symops = list(cast(Iterable[str], block["symops_xyz"]))
    if block.get("bond_rows") is not None:  # bond symmetry codes name the operations by id
        raw["loop_symops"] = ["space_group_symop_id", "space_group_symop_operation_xyz"]
        raw["space_group_symop_id"] = [str(number) for number in range(1, len(symops) + 1)]
    else:
        raw["loop_symops"] = ["space_group_symop_operation_xyz"]
    raw["space_group_symop_operation_xyz"] = symops

    positions = list(cast(Iterable[Sequence[object]], block["positions_exact"]))
    symbols = list(cast(Iterable[str], block.get("symbols", ())))
    labels = list(cast(Iterable[str], block.get("labels", symbols)))
    loop_atoms = [
        "atom_site_label",
        "atom_site_type_symbol",
        "atom_site_fract_x",
        "atom_site_fract_y",
        "atom_site_fract_z",
    ]
    raw["loop_atoms"] = loop_atoms
    raw["atom_site_label"] = labels
    raw["atom_site_type_symbol"] = symbols
    # A relaxed block states how many coordinate decimals its precision resolves; exact blocks
    # (and reader-round-tripped blocks that omit the key) keep the full 16-place claim.
    coordinate_decimals = cast("int | None", block.get("coordinate_decimals")) or _HTTK_CIF_DECIMAL_DIGITS
    for index, tag in enumerate(("atom_site_fract_x", "atom_site_fract_y", "atom_site_fract_z")):
        values, exact_values, has_exact = _dual_cif_column(
            (row[index] for row in positions), pad_coordinates=True, decimals=coordinate_decimals
        )
        raw[tag] = values
        if has_exact and exact_companions:
            raw[f"httk_{tag}_exact"] = exact_values
            loop_atoms.append(f"httk_{tag}_exact")
    occupancies_exact = cast(Iterable[object] | None, block.get("occupancies_exact"))
    occupancies = cast(Iterable[object] | None, block.get("occupancies"))
    occupancy_values = occupancies_exact if occupancies_exact is not None else occupancies
    if occupancy_values is not None:
        values, exact_values, has_exact = _dual_cif_column(occupancy_values)
        loop_atoms.append("atom_site_occupancy")
        raw["atom_site_occupancy"] = values
        if has_exact and exact_companions:
            loop_atoms.append("httk_atom_site_occupancy_exact")
            raw["httk_atom_site_occupancy_exact"] = exact_values
    for source, target in (
        ("attached_hydrogens", "atom_site_attached_hydrogens"),
        ("calc_flags", "atom_site_calc_flag"),
    ):
        column = block.get(source)
        if column is not None:
            column_values = list(cast(Iterable[object], column))
            if len(column_values) != len(positions):
                raise ValueError(f"CIF {source} and positions must have matching lengths")
            loop_atoms.append(target)
            raw[target] = column_values
    atom_type_symbols = list(cast(Iterable[str], block.get("atom_type_symbols", ())))
    atom_type_masses = list(cast(Iterable[object], block.get("atom_type_masses", ())))
    if atom_type_symbols or atom_type_masses:
        if len(atom_type_symbols) != len(atom_type_masses):
            raise ValueError("CIF atom-type symbols and masses must have matching lengths")
        raw["loop_atom_types"] = ["atom_type_symbol", "atom_type_mass"]
        raw["atom_type_symbol"] = atom_type_symbols
        raw["atom_type_mass"] = atom_type_masses
    _neutral_cif_network(raw, block)
    return raw


def _neutral_cif_network(raw: dict[str, object], block: Mapping[str, object]) -> None:
    """Add a block's bond network to the low-level writer's block, if it has one.

    ``bond_rows`` holds ``(label_1, label_2, symmetry_code_2, distance)`` per bond; none means an
    empty network, written as ``_httk_bonds_count 0``. ``bonds_source`` and ``shapes_source`` hold
    the four source fields (``None`` written ``?``), ``shape_rows`` holds ``(label, symbol)`` pairs.
    """
    rows = cast("Sequence[Sequence[str]] | None", block.get("bond_rows"))
    if rows is None:
        return
    if rows:
        columns = ("atom_site_label_1", "atom_site_label_2", "site_symmetry_1", "site_symmetry_2", "distance")
        raw["loop_bonds"] = [f"geom_bond_{name}" for name in columns]
        raw["geom_bond_atom_site_label_1"] = [row[0] for row in rows]
        raw["geom_bond_atom_site_label_2"] = [row[1] for row in rows]
        raw["geom_bond_site_symmetry_1"] = [_INAPPLICABLE] * len(rows)
        raw["geom_bond_site_symmetry_2"] = [_Bare(row[2]) for row in rows]
        raw["geom_bond_distance"] = [row[3] for row in rows]
    else:
        raw["httk_bonds_count"] = 0
    _neutral_cif_source(raw, "bonds_source", block)
    shapes = cast("Sequence[Sequence[str]]", block.get("shape_rows", ()))
    if shapes:
        raw["loop_shapes"] = ["httk_shape_atom_site_label", "httk_shape_symbol"]
        raw["httk_shape_atom_site_label"] = [row[0] for row in shapes]
        raw["httk_shape_symbol"] = [row[1] for row in shapes]
    _neutral_cif_source(raw, "shapes_source", block)


def _neutral_cif_source(raw: dict[str, object], name: str, block: Mapping[str, object]) -> None:
    """Add the four ``_httk_<name>_*`` items of a source the block holds, if any."""
    source = cast("Sequence[str | None] | None", block.get(name))
    if source is not None:
        for field, value in zip(("description", "software", "version", "uri"), source, strict=True):
            raw[f"httk_{name}_{field}"] = value


def _write_cif_payload(
    destination: str | os.PathLike[str] | io.TextIOBase,
    data: Mapping[str, object],
    *,
    approximate: bool = True,
    exact_companions: bool = False,
    **kwargs: object,
) -> None:
    r"""Write the neutral CIF payload returned by ``read_cif_asus`` or the structure serializer.

    Saving into CIF renders whatever the format can hold: a block the serializer flagged as
    ``approximate`` (an irrational or orientation-losing cell that has no exact CIF representation)
    is written as rounded decimals by default, with ``_httk_*_exact`` companions carrying the exact
    rational tokens where they exist. Passing ``approximate=False`` opts into the strict guarantee
    and refuses such a block instead. ``httk.core.save`` forwards this and any remaining keyword
    options (``header``, ``max_line_length``) here.

    :param destination: Filename or open text stream receiving the CIF text.
    :param data: The neutral CIF payload, either a single block or a ``blocks`` sequence, or
        lossless ``cif_blocks``, which are written as they are.
    :param approximate: Whether lossy rounding of a cell with no exact CIF form is allowed
        (default) rather than refused.
    :param exact_companions: Whether to emit the non-standard ``_httk_*_exact`` companion columns
        (exact rational tokens such as ``1/3``). Off by default so the file carries only standard
        CIF columns; pass ``exact_companions=True`` for an exact httk round-trip.
    :param \**kwargs: Remaining low-level :func:`write_cif` options.
    :raises ValueError: If a block requires approximation and ``approximate`` is ``False``.
    """
    cif_blocks = data.get("cif_blocks")
    if cif_blocks is not None:  # lossless blocks carry their own header and need no rounding
        write_cif(destination, cast(Iterable[CIFBlock], cif_blocks), **cast(dict[str, Any], kwargs))
        return
    blocks = data.get("blocks")
    if blocks is None:
        blocks = [data]
    block_list = list(cast(Iterable[Mapping[str, object]], blocks))
    if not approximate and any(block.get("approximate") for block in block_list):
        raise ValueError(
            "CIF cannot exactly represent this structure's cell parameters (an irrational or "
            "orientation-losing basis); drop approximate=False to write rounded decimals, or keep "
            "the original structure to preserve the exact basis"
        )
    options: dict[str, Any] = {"header": cast(str | None, data.get("header"))}
    options.update(kwargs)
    write_cif(
        destination,
        [("structure", _neutral_cif_block(block, exact_companions=exact_companions)) for block in block_list],
        **options,
    )
