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

import dataclasses
import io
import logging
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Any, Literal, Self, cast, overload

from httk.core import DecimalGap, DecimalScalar, TextstreamFileView

from .cif_items import (
    _MAX_NESTING,
    CIFBlock,
    CIFComment,
    CIFDataItem,
    CIFDialect,
    CIFItem,
    CIFLoop,
    CIFSaveFrame,
    CIFTable,
    CIFTextField,
    CIFValue,
    _cif_number,
    _cif_quote,
    _cif_space,
    _is_cif2_header,
    _Mode,
)
from .cif_tags import CIF_TAGS

logger = logging.getLogger(__name__)


# Normalized tags consumed by cif_parser.py and mcif_parser.py; a loop is droppable only
# when none of its columns is in this set. The Fourier coefficient template is checked below.
_PROTECTED_LOOP_TAGS = frozenset(
    {
        'cell_modulation_dimension',
        'cell_length_a',
        'cell_length_b',
        'cell_length_c',
        'cell_angle_alpha',
        'cell_angle_beta',
        'cell_angle_gamma',
        'atom_site_type_symbol',
        'atom_site_label',
        'atom_site_fract_x',
        'atom_site_fract_y',
        'atom_site_fract_z',
        'atom_site_occupancy',
        'atom_type_symbol',
        'atom_type.symbol',
        'atom_type_mass',
        'atom_type.mass',
        'atom_type_atomic_mass',
        'atom_type.atomic_mass',
        'atom_site_wyckoff_label',
        'atom_site_wyckoff_symbol',
        'atom_site_symmetry_multiplicity',
        'atom_site_site_symmetry_multiplicity',
        'atom_site_site_symmetry_order',
        'httk_atom_site_fract_x_exact',
        'httk_atom_site_fract_y_exact',
        'httk_atom_site_fract_z_exact',
        'httk_atom_site_occupancy_exact',
        'space_group_symop.operation_xyz',
        'space_group_symop_operation_xyz',
        'symmetry_equiv_pos_as_xyz',
        'space_group_name_h-m_alt',
        'symmetry_space_group_name_h-m',
        'space_group_name_hall',
        'symmetry_space_group_name_hall',
        'space_group_it_number',
        'symmetry_space_group_it_number',
        'symmetry_int_tables_number',
        'database_code_icsd',
        'citation_doi',
        'parent_propagation_vector.kxkykz',
        'atom_site_moment.label',
        'atom_site_moment.crystalaxis_x',
        'atom_site_moment.crystalaxis_y',
        'atom_site_moment.crystalaxis_z',
        'space_group_symop_magn_operation.xyz',
        'space_group_symop_magn_ssg_operation.algebraic',
        'space_group_symop_magn_centering.xyz',
        'space_group_symop_magn_ssg_centering.algebraic',
        'space_group_magn.number_bns',
        'space_group_magn.name_bns',
        'parent_space_group.name_h-m_alt',
        'parent_space_group.it_number',
        *CIF_TAGS['structural_q'],
        CIF_TAGS['structural_displacement_label'],
        CIF_TAGS['structural_occupancy_label'],
        *CIF_TAGS['magnetic_cartesian_moment'],
        CIF_TAGS['magnetic_fourier_label'],
        CIF_TAGS['magnetic_ssg_name'],
    }
)
_MAGNETIC_FOURIER_COEFFICIENT_RE = re.compile(
    '^' + re.escape(CIF_TAGS['magnetic_fourier_coeff']).replace(r'\{\}', r'\d+') + '$'
)
_PRAGMATIC_VALUE_SPLIT_RE = re.compile(r'\s+_|\s+data_|\s+loop_')
_STRUCTURAL_LOOP_PREFIXES = ('atom_site', 'space_group', 'symmetry')
# Bond-loop columns read by the structure adapter, without their _geom_bond_ prefix. Not
# protected: a malformed bond loop is an auxiliary loop, which structure loading drops with a
# warning even without repair, since a bond loop never stops a structure from loading.
_BOND_COLUMNS = ('atom_site_label_1', 'atom_site_label_2', 'site_symmetry_1', 'site_symmetry_2', 'distance')
_BOND_LOOP_TAGS = frozenset(f'geom_bond_{name}' for name in _BOND_COLUMNS)


def _is_consumed_tag(name: str) -> bool:
    return (
        name in _PROTECTED_LOOP_TAGS
        or name.startswith(_STRUCTURAL_LOOP_PREFIXES)
        or _MAGNETIC_FOURIER_COEFFICIENT_RE.fullmatch(name) is not None
    )


def _is_retained_tag(name: str) -> bool:
    return _is_consumed_tag(name) or name in _BOND_LOOP_TAGS


def _is_repairable_loop(header: list[str]) -> bool:
    return not any(_is_consumed_tag(name) for name in header)


class _RewindableIterator:
    def __init__(self, iterator: Iterable[str]) -> None:
        self._iter: Iterator[str] = iter(iterator)
        self._rewind = False
        self._cache: str | None = None

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> str:
        if self._rewind:
            self._rewind = False
        else:
            self._cache = next(self._iter)
        assert self._cache is not None
        return self._cache

    def rewind(self, rewindstr: str | None = None) -> None:
        if self._rewind:
            raise RuntimeError("Tried to backup more than one step.")
        elif self._cache is None:
            raise RuntimeError("Can't backup past the beginning.")
        self._rewind = True
        if rewindstr is not None:
            self._cache = rewindstr


def _read_cif_rewind_if_needed(f: _RewindableIterator, row: str, done_fields: int) -> bool:
    splitstr = row.lstrip().split(None, done_fields)
    if len(splitstr) > 1:
        rest = splitstr[-1]
        if rest.strip() != "":
            f.rewind(rest)
            return True
        return False
    else:
        return False


def _read_cif_loop(
    f: _RewindableIterator,
    pragmatic: bool = True,
    allow_cif2: bool = False,
    *,
    block_name: str,
    repair: bool = False,
    structural_only: bool = False,
) -> dict[str, list[Any]] | None:
    noteol = False
    loop_data: dict[str, list[Any]] = {}
    header = []
    for row in f:
        striprow = row.strip()
        lowrow = striprow.lower()
        if lowrow.startswith("_"):
            name = lowrow[1:]
            loop_data[name] = []
            header.append(name)
            noteol = _read_cif_rewind_if_needed(f, row, 1)
        else:
            f.rewind()
            break
    columns = [loop_data[name] for name in header]

    if structural_only and header and not any(_is_retained_tag(name) for name in header) and not allow_cif2:
        count = _skip_cif_loop(f, pragmatic)
        counts = {name: count // len(header) + (index < count % len(header)) for index, name in enumerate(header)}
        _validate_loop_counts(header, counts, block_name, repair, structural_only)
        return None

    # _read_cif_data_value recognizes and rewinds the next loop/data/tag token, so
    # peeking here would strip, lowercase, and traverse every ordinary value twice.
    while columns:
        for column in columns:
            val, noteol = _read_cif_data_value(f, noteol, pragmatic, allow_cif2, inloop=True)
            if val is None:
                break
            column.append(val)
        else:
            continue
        break
    counts = {name: len(values) for name, values in loop_data.items()}
    if not _validate_loop_counts(header, counts, block_name, repair, structural_only):
        return None
    return loop_data


def _validate_loop_counts(
    header: list[str], counts: dict[str, int], block_name: str, repair: bool, structural_only: bool = False
) -> bool:
    """Apply the common strict/repair policy to CIF loop column counts.

    Structure loading (``structural_only``) drops a malformed bond loop even in strict mode.
    """
    if len(set(counts.values())) > 1:
        rendered_counts = ", ".join(f"{name}={count}" for name, count in counts.items())
        message = f"CIF loop with {len(header)} columns has mismatched value counts: {rendered_counts}"
        if _is_repairable_loop(header):
            if repair or (structural_only and not _BOND_LOOP_TAGS.isdisjoint(header)):
                logger.warning(
                    "CIF block %r: dropped malformed auxiliary loop starting with _%s",
                    block_name,
                    header[0],
                    extra={'context': 'cif'},
                )
                return False
            message += " (an auxiliary loop like this can be dropped by loading with repair=True, which applies documented repairs with warnings)"
        raise ValueError(message)
    return True


def _skip_cif_loop(f: _RewindableIterator, pragmatic: bool) -> int:
    """Count an unneeded CIF1 loop without materializing its ordinary data rows."""
    count = 0
    for row in f:
        striprow = row.strip()
        if not striprow:
            continue
        if striprow.startswith("#"):
            # Preserve the ordinary parser's strict token-boundary behavior around
            # comments until that CIF conformance question is settled explicitly.
            f.rewind()
            return _skip_cif_loop_tokens(f, pragmatic, count)
        if row.startswith("_") or (striprow[0] in "dDlL" and striprow.lower().startswith(("data_", "loop_"))):
            f.rewind()
            return count
        if "'" in row or '"' in row:
            f.rewind()
            return _skip_cif_loop_tokens(f, pragmatic, count)
        if row.startswith(";"):
            count += 1
            for continuation in f:
                if not continuation.startswith(";"):
                    continue
                tail = continuation[1:].strip()
                if tail:
                    f.rewind(tail)
                    return _skip_cif_loop_tokens(f, pragmatic, count)
                break
            continue
        tokens = striprow.split()
        for index, token in enumerate(tokens):
            if token.startswith("_") or (token[0] in "dDlL" and token.lower().startswith(("data_", "loop_"))):
                f.rewind(" ".join(tokens[index:]))
                return count
            value, marker, _ = token.partition("#")
            if value:
                count += 1
            if marker:
                break
    return count


def _skip_cif_loop_tokens(f: _RewindableIterator, pragmatic: bool, count: int) -> int:
    """Finish skipping a CIF1 loop through the ordinary tokenizer."""
    noteol = False
    while True:
        try:
            row = next(f)
            while row.isspace():
                row = next(f)
        except StopIteration:
            return count
        lowrow = row.strip().lower()
        if not row or row.startswith("_") or lowrow.startswith(("data_", "loop_")):
            f.rewind()
            return count
        f.rewind()
        value, noteol = _read_cif_data_value(f, noteol, pragmatic, inloop=True)
        if value is not None:
            count += 1


def _read_cif_data_value(
    f: _RewindableIterator,
    noteol: bool,
    pragmatic: bool = True,
    allow_cif2: bool = False,
    inloop: bool = False,
    inlist: bool = False,
) -> tuple[Any, bool]:
    data_value: Any = None
    for row in f:
        if inloop and not row:
            f.rewind()
            return None, False
        striprow = row.strip()
        if striprow.startswith("#") or striprow == "":
            noteol = False
            continue
        elif inloop and (
            row.startswith("_") or (striprow[0] in "dDlL" and striprow.lower().startswith(("data_", "loop_")))
        ):
            f.rewind()
            return None, False
        elif (not noteol) and row.startswith(';'):
            folded = False
            newline = False
            data_parts = []
            if row[1] == "\\" and row[2:].rstrip("\r\n") == "":
                folded = True
            elif row[1:].isspace():
                if not pragmatic:
                    data_parts.append(row.lstrip().rstrip('\r\n'))
                    newline = True
            else:
                data_parts.append(row.lstrip()[1:].rstrip('\r\n'))
                newline = True
            last_irow = ""
            content_lines = []
            for irow in f:
                last_irow = irow
                if irow.startswith(';'):
                    break
                content_lines.append(irow.rstrip('\r\n'))
            # Join once: repeated string concatenation is quadratic for large text fields.
            if folded:
                for trimmed in content_lines:
                    if newline:
                        data_parts.append('\n')
                    if trimmed.endswith("\\"):
                        data_parts.append(trimmed.rstrip("\\"))
                        newline = False
                    else:
                        data_parts.append(trimmed)
                        newline = True
                data_value = ''.join(data_parts)
            else:
                if data_parts:
                    data_parts.extend(content_lines)
                else:
                    data_parts = content_lines
                data_value = '\n'.join(data_parts)
            stripirow = last_irow.strip()
            if len(stripirow) > 1:
                f.rewind(stripirow[1:])
                noteol = True
            else:
                noteol = False
            break
        elif striprow.startswith(("'", '"')):
            # The cif quoting rules are ... weird. Quotes are "escaped" if they are not followed by whitespace.
            quote = striprow[0]
            starti = 1
            for chari in range(1, len(striprow) - 1):
                if striprow[chari] == quote and str(striprow[chari + 1]).isspace():
                    endi = chari
                    endq = chari + 1
                    break
            else:
                if striprow[-1] != quote:
                    starti = 0
                    endi = len(striprow)
                    endq = len(striprow)
                else:
                    endi = len(striprow) - 1
                    endq = len(striprow)
            data_value = striprow[starti:endi]
            if endq != len(striprow):
                f.rewind(striprow[endq:])
                noteol = True
            else:
                noteol = False
            break
        elif allow_cif2 and inlist and striprow.startswith("]"):
            # TODO: Is ] allowed without whitespace after? I need to check the spec
            splitstr = striprow.split("]", 1)
            if len(splitstr) > 1 and len(splitstr[1]) > 0:
                f.rewind(splitstr[1])
                noteol = True
            data_value = None
            break
        elif allow_cif2 and striprow.startswith("["):
            if len(striprow) > 1:
                f.rewind(striprow[1:])
                noteol = True
            data_value = []
            while True:
                innerval, noteol = _read_cif_data_value(f, noteol, pragmatic, allow_cif2, inloop=False, inlist=True)
                if innerval is None:
                    break
                data_value += [innerval]
            break
        elif allow_cif2 and inlist and ("]" in striprow):
            splitstr2 = striprow.split("]", 1)
            splitstr = splitstr2[0].split(None, 1)
            data_value = splitstr[0].strip()
            rightside = ""
            if len(splitstr) > 1:
                f.rewind(splitstr[1] + "]" + splitstr2[1])
            else:
                f.rewind("]" + splitstr2[1])
            noteol = True
            break
        else:
            if pragmatic and not inloop:
                # In pragmatic mode, if we are not in a loop and there is more than one data value
                # separated by whitespace, read all of it. This should always be ok to do, since
                # multiple data values in this situation would be an
                # error in the file otherwise, but if there is whitespace + underscore/data_/loop_ we parse that
                # as a new symbol, since otherwise we COULD misread valid files (with very weird formatting...).
                splitstr = _PRAGMATIC_VALUE_SPLIT_RE.split(striprow, maxsplit=1)
            else:
                splitstr = striprow.split(None, 1)
            # "Data on a line following a hash character `#' is considered to be a comment,
            # except if it is contained within a text string."
            data_value = splitstr[0].partition("#")[0].strip()
            rightside = ""
            if len(splitstr) > 1:
                rightside = splitstr[1].strip()
            if rightside != "":
                f.rewind(rightside)
                noteol = True
            else:
                noteol = False
            break
    return data_value, noteol


def _read_cif_data_block(
    f: _RewindableIterator,
    pragmatic: bool = True,
    allow_cif2: bool = False,
    *,
    block_name: str,
    repair: bool = False,
    structural_only: bool = False,
) -> dict[str, Any]:
    data_items: dict[str, Any] = {}
    loops = 0
    for row in f:
        striprow = row.strip()
        if striprow.startswith("#"):
            continue
        elif striprow and striprow[0] in "dD" and striprow[:5].lower() == "data_":
            f.rewind()
            return data_items
        elif striprow and striprow[0] in "lL" and striprow[:5].lower() == "loop_":
            _read_cif_rewind_if_needed(f, row, 1)
            loopdata = _read_cif_loop(
                f,
                pragmatic,
                allow_cif2,
                block_name=block_name,
                repair=repair,
                structural_only=structural_only,
            )
            if loopdata is None:
                continue
            data_items['loop_' + str(loops)] = list(loopdata.keys())
            loops += 1
            data_items.update(loopdata)
        elif striprow.startswith(";"):
            # Multi-line string that we've failed to tie to a name, lets just skip it, maybe we should warn
            for irow in f:
                if irow.rstrip() == ";":
                    break
        elif striprow.startswith("_"):
            lowrow = striprow.lower()
            lowsplit = lowrow.split()
            data_name = lowsplit[0][1:]
            if len(lowsplit) > 1:
                noteol = True
                rightside = striprow.split(None, 1)[1].strip()
                f.rewind(rightside)
            else:
                noteol = False
            data_value, noteol = _read_cif_data_value(f, noteol, pragmatic, allow_cif2, inloop=False)
            if not structural_only or _is_retained_tag(data_name):
                data_items[data_name] = data_value
    return data_items


def _read_cif(
    f: _RewindableIterator,
    pragmatic: bool,
    allow_cif2: bool,
    repair: bool,
    structural_only: bool,
) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    header = ""
    datalist = []
    for row in f:
        if row.strip().startswith("#"):
            header += row
        else:
            f.rewind()
            break

    for row in f:
        striprow = row.strip()
        if striprow and striprow[0] in "dD" and striprow[:5].lower() == "data_":
            lowrow = striprow.lower()
            data_block_name = lowrow.partition('_')[2].split()[0].strip()
            _read_cif_rewind_if_needed(f, row, 1)
            datalist.append(
                (
                    data_block_name,
                    _read_cif_data_block(
                        f,
                        pragmatic,
                        allow_cif2,
                        block_name=data_block_name,
                        repair=repair,
                        structural_only=structural_only,
                    ),
                )
            )
    return datalist, header


# The fast path's split is quadratic on long whitespace runs (FP3); this one matches the same
# cuts in linear time.
_LOSSLESS_VALUE_SPLIT_RE = re.compile(r"(?<!\s)\s+(?:_|data_|loop_)")


# Lossless reading: read_cif(..., lossless=True) returns CIFBlocks. The fast path above is
# untouched; _legacy_block renders a block as the dictionary that path returns.


@dataclasses.dataclass(frozen=True, slots=True)
class _Token:
    kind: str  # bare, quoted, text, comment, key (CIF2 table key), [, ], {, }
    text: str
    line: int  # the 1-based line the token ends on
    first: int  # the 1-based line it starts on
    start: int  # its column on the first line
    end: int  # the column after it on the last line
    inline: bool = False  # a comment sharing its line with a preceding token


def _lossless_text(source: str | os.PathLike[str] | IO[bytes] | Iterable[str], repair: bool) -> str:
    """Return the decoded source with LF line ends and no byte-order mark."""
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        try:
            with TextstreamFileView(path) as stream:
                text = stream.read()
        except UnicodeDecodeError:
            if not repair:
                raise
            logger.warning(
                "CIF file %r is not valid UTF-8; decoded it as Latin-1 under repair=True",
                str(path),
                extra={"context": "cif"},
            )
            with TextstreamFileView(path, encoding="latin-1") as stream:
                text = stream.read()
    elif isinstance(source, (io.BufferedIOBase, io.RawIOBase)):
        data = source.read()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            if not repair:
                raise
            logger.warning(
                "CIF stream is not valid UTF-8; decoded it as Latin-1 under repair=True", extra={"context": "cif"}
            )
            text = data.decode("latin-1")
    else:
        text = "".join(line if line.endswith("\n") else line + "\n" for line in cast(Iterable[str], source))
    return text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")


def _lex(lines: list[str], mode: _Mode, pragmatic: bool) -> list[_Token]:
    """Split CIF lines into tokens under the syntax of ``mode``."""
    cif2 = mode == "cif2"
    tokens: list[_Token] = []
    depth = 0  # open mCIF lists: there ']' ends a word and may close a quote
    index = 0
    while index < len(lines):
        line = lines[index]
        pos = 0
        seen = False
        if line.startswith(";"):
            close = index + 1
            while close < len(lines) and not lines[close].startswith(";"):
                close += 1
            if close == len(lines):
                raise ValueError(f"unterminated CIF text field starting at line {index + 1}")
            text = "\n".join([line[1:], *lines[index + 1 : close]])
            tokens.append(_Token("text", text, close + 1, index + 1, 0, 1))
            index, line, pos, seen = close, lines[close], 1, True
        while pos < len(line):
            char = line[pos]
            lineno = index + 1
            if _cif_space(char):
                pos += 1
                continue
            if char == "#":
                tokens.append(_Token("comment", line[pos + 1 :].rstrip(), lineno, lineno, pos, len(line), seen))
                break
            seen = True
            if cif2 and line.startswith(("'''", '"""'), pos):
                delimiter = line[pos : pos + 3]
                last, offset = index, pos + 3
                while (stop := lines[last].find(delimiter, offset)) < 0:
                    last, offset = last + 1, 0
                    if last == len(lines):
                        raise ValueError(f"unterminated CIF2 triple-quoted string starting at line {lineno}")
                if last == index:
                    text = line[pos + 3 : stop]
                else:
                    text = "\n".join([line[pos + 3 :], *lines[index + 1 : last], lines[last][:stop]])
                kind = "key" if lines[last].startswith(":", stop + 3) else "quoted"
                tokens.append(_Token(kind, text, last + 1, lineno, pos, stop + 3))
                index, line, pos = last, lines[last], stop + 3 + (kind == "key")
                continue
            if char in "'\"":
                if cif2:
                    stop = line.find(char, pos + 1)
                else:
                    # A CIF1 quote closes only before whitespace, the line end or, in a list, ']'.
                    stop = pos + 1
                    while (stop := line.find(char, stop)) >= 0 and not (
                        stop + 1 == len(line) or _cif_space(line[stop + 1]) or (depth and line[stop + 1] == "]")
                    ):
                        stop += 1
                if stop < 0:
                    if cif2 or not pragmatic:
                        raise ValueError(f"unterminated CIF quoted string at line {lineno}")
                    # D9: as the fast reader, the rest of the line, opening quote included.
                    tokens.append(_Token("quoted", line[pos:].rstrip(), lineno, lineno, pos, len(line)))
                    break
                kind = "key" if cif2 and line.startswith(":", stop + 1) else "quoted"
                tokens.append(_Token(kind, line[pos + 1 : stop], lineno, lineno, pos, stop + 1))
                pos = stop + 1 + (kind == "key")
                continue
            if (char == "[" and mode != "cif1") or (char == "]" and (cif2 or depth)) or (cif2 and char in "{}"):
                depth = depth + 1 if char == "[" else max(depth - (char == "]"), 0)
                tokens.append(_Token(char, char, lineno, lineno, pos, pos + 1))
                pos += 1
                continue
            # CIF2 data names and block/frame codes end only at whitespace; values also at brackets.
            brackets = cif2 and not (char == "_" or line[pos : pos + 5].lower() in ("data_", "save_"))
            stop = pos
            while (
                stop < len(line)
                and not _cif_space(line[stop])
                and not (brackets and line[stop] in "[]{}")
                and not (mode == "mcif" and depth and line[stop] == "]")
            ):
                stop += 1
            tokens.append(_Token("bare", line[pos:stop], lineno, lineno, pos, stop))
            pos = stop
        index += 1
    return tokens


def _hoist_compound_comments(tokens: list[_Token]) -> list[_Token]:
    """Move comments inside lists and tables to just after the value, as own-line comments (D5)."""
    result: list[_Token] = []
    moved: list[_Token] = []
    depth = 0
    for token in tokens:
        if token.kind == "comment" and depth:
            moved.append(dataclasses.replace(token, inline=False))
            continue
        if moved and not depth and not (token.kind == "comment" and token.inline):
            result += moved
            moved = []
        result.append(token)
        if token.kind in "[{":
            depth += 1
        elif token.kind in "]}" and depth:
            depth -= 1
    return result + moved


def _is_tag(token: _Token) -> bool:
    return token.kind == "bare" and token.text.startswith("_")


def _keyword(token: _Token) -> str | None:
    """Return the lowercase CIF reserved word a bare token is (``data_``/``save_`` by prefix)."""
    if token.kind != "bare":
        return None
    word = token.text.lower()
    if word.startswith(("data_", "save_")):
        return word[:5]
    return word if word in ("loop_", "global_", "stop_") else None


class _LosslessParser:
    """Build CIF items from a token list."""

    def __init__(self, tokens: list[_Token], lines: list[str], mode: _Mode, pragmatic: bool) -> None:
        self.tokens = tokens
        self.lines = lines
        self.mode: _Mode = mode
        self.pragmatic = pragmatic
        self.k = 0
        self.depth = 0  # open lists, tables and save frames

    def nest(self, token: _Token, step: int) -> None:
        self.depth += step
        if self.depth > _MAX_NESTING:
            raise ValueError(f"CIF lists, tables or save frames nested deeper than {_MAX_NESTING} at line {token.line}")

    def peek(self) -> _Token | None:
        return self.tokens[self.k] if self.k < len(self.tokens) else None

    def string(self, text: str, line: int) -> str:
        """Return a string value after checking it can be written back."""
        try:
            _cif_quote(text, self.mode)
        except ValueError as error:
            raise ValueError(f"{error} (line {line})") from None
        return text

    def value(self) -> CIFValue:
        token = self.tokens[self.k]
        self.k += 1
        if token.kind == "text":
            return CIFTextField(token.text)
        if token.kind == "quoted":
            return self.string(token.text, token.line)
        if token.kind == "bare":
            if _keyword(token) is not None:
                raise ValueError(f"CIF reserved word {token.text[:60]!r} used as a value at line {token.line}")
            if token.text in ("?", "."):
                return DecimalGap(token.text)
            number = _cif_number(token.text)
            return self.string(token.text, token.line) if number is None else number
        if token.kind in ("[", "{"):
            close = "]" if token.kind == "[" else "}"
            self.nest(token, 1)
            values: list[CIFValue] = []
            entries: list[tuple[str, CIFValue]] = []
            while (inner := self.peek()) is None or inner.kind != close:
                if inner is None:
                    raise ValueError(f"unterminated CIF list or table starting at line {token.line}")
                if token.kind == "[":
                    values.append(self.value())
                    continue
                if inner.kind != "key":
                    raise ValueError(f"CIF2 table key expected at line {inner.line}")
                self.k += 1
                if any(key == inner.text for key, _ in entries):
                    raise ValueError(f"repeated CIF2 table key {inner.text[:60]!r} at line {inner.line}")
                entries.append((inner.text, self.value()))
            self.k += 1
            self.nest(token, -1)
            return tuple(values) if token.kind == "[" else CIFTable(tuple(entries))
        raise ValueError(f"unexpected CIF token {token.text[:60]!r} at line {token.line}")

    def items(self, frame_line: int | None) -> list[CIFItem]:
        """Read items up to the next block (or, in a frame, its ``save_`` terminator)."""
        items: list[CIFItem] = []
        while (token := self.peek()) is not None:
            word = _keyword(token)
            if word == "data_":
                if frame_line is not None:
                    raise ValueError(f"data_ inside a save frame at line {token.line}")
                return items
            self.k += 1
            if token.kind == "comment":
                items.append(CIFComment(token.text, token.inline))
            elif word == "save_":
                if token.text[5:]:
                    self.nest(token, 1)
                    items.append(CIFSaveFrame(token.text[5:], tuple(self.items(token.line))))
                    self.nest(token, -1)
                elif frame_line is None:
                    raise ValueError(f"save_ without an open save frame at line {token.line}")
                else:
                    return items
            elif word == "loop_":
                items.append(self.loop(token))
            elif word is not None:
                raise ValueError(f"CIF reserved word {token.text!r} at line {token.line}")
            elif _is_tag(token):
                items.append(self.data_item(token))
            else:
                raise ValueError(f"CIF value {token.text[:60]!r} without a data name at line {token.line}")
        if frame_line is not None:
            raise ValueError(f"unterminated CIF save frame starting at line {frame_line}")
        return items

    def data_item(self, name: _Token) -> CIFDataItem:
        comments: list[tuple[int, CIFComment]] = []
        while (token := self.peek()) is not None and token.kind == "comment":
            comments.append((0, CIFComment(token.text, token.inline)))
            self.k += 1
        if token is None or _is_tag(token) or _keyword(token) in ("data_", "save_", "loop_"):
            raise ValueError(f"CIF data name {name.text!r} has no value at line {name.line}")
        value = self.value()
        if self.pragmatic and token.kind == "bare":
            joined = self.pragmatic_join(token)
            if joined is not None:
                value, comment = joined
                if comment is not None:
                    comments.append((1, comment))
        after = self.peek()
        if after is not None and after.kind == "comment" and after.inline:
            comments.append((1, CIFComment(after.text, True)))
            self.k += 1
        return CIFDataItem(name.text, value, tuple(comments))

    def pragmatic_join(self, token: _Token) -> tuple[str, CIFComment | None] | None:
        """D1: join a bare value with the further words on its line, as the fast reader does.

        :param token: The bare value token, already consumed.
        :return: The joined value and the comment cut from its line, or ``None`` if nothing follows.
        """
        k = self.k
        run_end = -1
        while k < len(self.tokens):
            other = self.tokens[k]
            if not (
                other.first == other.line == token.line
                and other.kind in ("bare", "quoted")
                and not _is_tag(other)
                and _keyword(other) is None
            ):
                break
            run_end = other.end
            k += 1
        if run_end < 0:
            return None
        raw = self.lines[token.line - 1][token.start :]
        split = _LOSSLESS_VALUE_SPLIT_RE.search(raw)
        cut = len(raw) if split is None else split.start()
        span = run_end - token.start
        hash_at = raw.find("#", 0, min(cut, span))
        comment = None
        if hash_at >= 0:
            # Everything after that '#' is the item's inline comment: consume the whole line.
            value = raw[:hash_at].rstrip()
            comment = CIFComment(raw[hash_at + 1 :].rstrip(), True)
            while (rest := self.peek()) is not None and rest.first == token.line:
                if rest.line != token.line:
                    raise ValueError(f"pragmatic CIF value at line {token.line} runs into a multi-line value")
                self.k += 1
        elif cut < span:
            raise ValueError(f"pragmatic CIF value at line {token.line} would split a quoted string")
        else:
            value = raw[:span]
            self.k = k
        return self.string(value, token.line), comment

    def loop(self, keyword: _Token) -> CIFLoop:
        names: list[str] = []
        comments: list[tuple[int, CIFComment]] = []
        while (token := self.peek()) is not None and (_is_tag(token) or token.kind == "comment"):
            if token.kind == "comment":
                comments.append((len(names), CIFComment(token.text, token.inline)))
            else:
                names.append(token.text)
            self.k += 1
        if not names:
            raise ValueError(f"CIF loop_ without data names at line {keyword.line}")
        values: list[CIFValue] = []
        pending: list[_Token] = []
        while (token := self.peek()) is not None and not _is_tag(token) and _keyword(token) is None:
            if token.kind == "comment":
                pending.append(token)
                self.k += 1
                continue
            comments += [(len(names) + len(values), CIFComment(c.text, c.inline)) for c in pending]
            pending = []
            values.append(self.value())
        end = len(names) + len(values)
        if pending and pending[0].inline:
            comments.append((end, CIFComment(pending.pop(0).text, True)))
        # Own-line comments after the loop's last token belong to the enclosing item sequence.
        self.k -= len(pending)
        while comments and comments[-1][0] == end and not comments[-1][1].inline:
            comments.pop()
            self.k -= 1
        if not values:
            # On re-reading, an empty loop ends only at a reserved word or the end of input, or at
            # own-line comments that lead the next data_ block.
            k = self.k
            while k < len(self.tokens) and self.tokens[k].kind == "comment":
                k += 1
            after = self.tokens[k] if k < len(self.tokens) else None
            word = None if after is None else _keyword(after)
            if (word != "data_") if k > self.k else (after is not None and word is None):
                raise ValueError(f"CIF loop_ at line {keyword.line} has no values")
        if len(values) % len(names):
            raise ValueError(
                f"CIF loop starting at line {keyword.line} has {len(values)} values for {len(names)} data names"
            )
        rows = tuple(tuple(values[i : i + len(names)]) for i in range(0, len(values), len(names)))
        return CIFLoop(tuple(names), rows, tuple(comments))


def _read_cif_lossless(
    source: str | os.PathLike[str] | IO[bytes] | Iterable[str], pragmatic: bool, allow_cif2: bool, repair: bool
) -> list[CIFBlock]:
    lines = _lossless_text(source, repair).split("\n")
    # The header is the run of comment lines from line 1, as on the fast path.
    header_lines = 0
    while header_lines < len(lines) and lines[header_lines].strip().startswith("#"):
        header_lines += 1
    dialect: CIFDialect = "mcif" if allow_cif2 else "cif"
    mode: _Mode = "mcif" if allow_cif2 else "cif1"
    if not allow_cif2 and header_lines and _is_cif2_header(lines[0].strip()[1:].rstrip()):
        mode = "cif2"
    parser = _LosslessParser(_hoist_compound_comments(_lex(lines, mode, pragmatic)), lines, mode, pragmatic)
    header = tuple(CIFComment(token.text) for token in parser.tokens[:header_lines])
    parser.k = header_lines
    leading: list[CIFComment] = []
    while (token := parser.peek()) is not None and token.kind == "comment":
        leading.append(CIFComment(token.text))
        parser.k += 1
    blocks: list[CIFBlock] = []
    while (token := parser.peek()) is not None:
        if _keyword(token) != "data_":
            raise ValueError(f"CIF content before the first data_ at line {token.line}")
        if not token.text[5:]:
            raise ValueError(f"empty CIF data block name at line {token.line}")
        parser.k += 1
        items = parser.items(None)
        trailing: list[CIFComment] = []
        if parser.peek() is not None:
            # Own-line comments after a block's last token lead the next block.
            while items and isinstance(items[-1], CIFComment) and not items[-1].inline:
                trailing.insert(0, cast(CIFComment, items.pop()))
        blocks.append(CIFBlock(token.text[5:], tuple(items), header, tuple(leading), dialect))
        leading = trailing
    return blocks


def _legacy_value(value: CIFValue) -> Any:
    if isinstance(value, DecimalScalar):
        return value.token
    if isinstance(value, DecimalGap):
        return value.value
    if isinstance(value, CIFTextField):
        first, *rest = value.text.split("\n")
        if first == "\\":
            parts: list[str] = []
            newline = False
            for trimmed in rest:
                if newline:
                    parts.append("\n")
                newline = not trimmed.endswith("\\")
                parts.append(trimmed.rstrip("\\"))
            return "".join(parts)
        return "\n".join(rest) if not first.strip() else value.text
    if isinstance(value, tuple):
        return [_legacy_value(entry) for entry in value]
    if isinstance(value, CIFTable):
        return {key: _legacy_value(entry) for key, entry in value.entries}
    return value


def _legacy_block(block: CIFBlock, *, structural_only: bool) -> dict[str, Any]:
    """Return the dictionary the fast reader returns for this block.

    Mirrors ``read_cif(..., allow_cif2=block.dialect == "mcif", structural_only=structural_only)``
    with ``pragmatic=True``: lowercase keys without the leading ``_`` (later entries win, save frames
    flattened), ``loop_N`` keys listing each loop's columns, and values as the fast reader's text.
    With ``structural_only``, only retained data items are kept and, outside mCIF, loops without
    a retained column are skipped.
    """
    result: dict[str, Any] = {}
    loops = 0
    skip_loops = structural_only and block.dialect != "mcif"

    def walk(items: tuple[CIFItem, ...]) -> None:
        nonlocal loops
        for item in items:
            if isinstance(item, CIFDataItem):
                key = item.name[1:].lower()
                if not structural_only or _is_retained_tag(key):
                    result[key] = _legacy_value(item.value)
            elif isinstance(item, CIFLoop):
                header = [name[1:].lower() for name in item.names]
                if skip_loops and not any(_is_retained_tag(name) for name in header):
                    continue
                columns: dict[str, list[Any]] = {name: [] for name in header}
                for row in item.rows:
                    for name, value in zip(header, row, strict=True):
                        columns[name].append(_legacy_value(value))
                result[f"loop_{loops}"] = list(columns)
                loops += 1
                result.update(columns)
            elif isinstance(item, CIFSaveFrame):
                walk(item.items)

    walk(block.items)
    return result


@overload
def read_cif(
    source: str | os.PathLike[str] | IO[bytes] | Iterable[str],
    pragmatic: bool = True,
    allow_cif2: bool = False,
    *,
    repair: bool = False,
    structural_only: bool = False,
    lossless: Literal[False] = False,
) -> tuple[list[tuple[str, dict[str, Any]]], str]: ...


@overload
def read_cif(
    source: str | os.PathLike[str] | IO[bytes] | Iterable[str],
    pragmatic: bool = True,
    allow_cif2: bool = False,
    *,
    repair: bool = False,
    structural_only: Literal[False] = False,
    lossless: Literal[True],
) -> list[CIFBlock]: ...


def read_cif(
    source: str | os.PathLike[str] | IO[bytes] | Iterable[str],
    pragmatic: bool = True,
    allow_cif2: bool = False,
    *,
    repair: bool = False,
    structural_only: bool = False,
    lossless: bool = False,
) -> tuple[list[tuple[str, dict[str, Any]]], str] | list[CIFBlock]:
    """Read CIF text as ``(data_blocks, header)``, or losslessly as :class:`~httk.atomistic.io.cif.CIFBlock` objects.

    Paths are opened through :class:`httk.core.TextstreamFileView`, including compressed
    CIF files. A binary stream is decoded like a file (UTF-8, with the same Latin-1 fallback
    under ``repair``). Open streams and iterables are consumed but left open.

    With ``lossless=True`` the result is one :class:`~httk.atomistic.io.cif.CIFBlock` per ``data_`` block, keeping
    data names, comments, numbers and every item in written order. Its syntax is CIF1, full
    CIF2 when line 1 is the CIF2 magic code, or the magnetic CIF syntax (CIF1 plus lists,
    ``dialect="mcif"``) under ``allow_cif2``. What it normalises: whitespace, quote style, line
    ends, a byte-order mark, comments inside CIF2 lists and tables (moved after the value), and,
    under ``pragmatic``, a bare value followed by more words on its line (joined as the fast
    reader joins it) and a quote left open at the line end (the rest of the line, opening quote
    included). Broken syntax raises ``ValueError`` with a line number, including inputs the fast
    reader tolerates: a value without a data name, a data name without a value, a malformed loop
    (also under ``repair``), an unterminated text field, and content before the first ``data_``.

    :param source: A filename, open text or binary stream, or iterable of CIF lines.
    :param pragmatic: Accept selected common deviations from strict CIF tokenization.
    :param allow_cif2: Parse CIF2 list values in addition to CIF1 data.
    :param repair: Apply documented, warning-emitting CIF repairs, including malformed
        auxiliary-loop removal and a Latin-1 fallback for legacy files. Lossless reading applies
        only the Latin-1 fallback.
    :param structural_only: Retain only tags consumed by httk's structural adapters (bond loops included) and
        skip auxiliary CIF1 loops.
    :param lossless: Return :class:`~httk.atomistic.io.cif.CIFBlock` objects instead of dictionaries.
    :return: The data blocks and the leading comment header, or the lossless blocks.
    :raises ValueError: If a loop contains mismatched column value counts, or, when lossless,
        the CIF syntax is broken.
    :raises TypeError: If ``lossless`` is combined with ``structural_only``.
    """
    if lossless:
        if structural_only:
            raise TypeError("read_cif(lossless=True) keeps every item; structural_only does not apply")
        return _read_cif_lossless(source, pragmatic, allow_cif2, repair)
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        try:
            with TextstreamFileView(path) as stream:
                return _read_cif(_RewindableIterator(stream), pragmatic, allow_cif2, repair, structural_only)
        except UnicodeDecodeError:
            if not repair:
                raise
            logger.warning(
                "CIF file %r is not valid UTF-8; decoded it as Latin-1 under repair=True",
                str(path),
                extra={"context": "cif"},
            )
            with TextstreamFileView(path, encoding="latin-1") as stream:
                return _read_cif(_RewindableIterator(stream), pragmatic, allow_cif2, repair, structural_only)
    if isinstance(source, (io.BufferedIOBase, io.RawIOBase)):
        data = source.read()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            if not repair:
                raise
            logger.warning(
                "CIF stream is not valid UTF-8; decoded it as Latin-1 under repair=True", extra={"context": "cif"}
            )
            text = data.decode("latin-1")
        lines = io.StringIO(text, newline=None)
        return _read_cif(_RewindableIterator(lines), pragmatic, allow_cif2, repair, structural_only)
    return _read_cif(_RewindableIterator(cast(Iterable[str], source)), pragmatic, allow_cif2, repair, structural_only)
