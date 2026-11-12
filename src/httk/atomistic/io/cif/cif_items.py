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

"""The lossless CIF item model: data blocks as ordered, structured syntax.

A :class:`CIFBlock` holds exactly what a CIF data block says, in written order: data items,
loops, comments and save frames, with numbers kept as written
(:class:`~httk.core.DecimalScalar`). It knows nothing about structures. What is part of the
model follows the content rule of :class:`~httk.atomistic.CIFStructure`.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

import httk.core
from httk.core import DecimalGap, DecimalScalar, DecimalVector

__all__ = [
    "CIFBlock",
    "CIFComment",
    "CIFDataItem",
    "CIFDialect",
    "CIFItem",
    "CIFLoop",
    "CIFSaveFrame",
    "CIFTable",
    "CIFTextField",
    "CIFValue",
]

#: One CIF value: a string, a text field, a number, a gap (``?``/``.``), a list (a tuple) or a
#: CIF2 table.
type CIFValue = str | CIFTextField | httk.core.DecimalScalar | httk.core.DecimalGap | tuple[CIFValue, ...] | CIFTable
#: The syntax of a block: ``"cif"`` (CIF1, or full CIF2 under the CIF2 header) or ``"mcif"``
#: (the magnetic CIF reader's syntax: CIF1 plus lists, whatever the header says).
type CIFDialect = Literal["cif", "mcif"]
#: One entry of a block or save frame.
type CIFItem = CIFDataItem | CIFLoop | CIFComment | CIFSaveFrame
#: A comment placed inside a data item or loop: ``(position, comment)``.
type _PlacedComment = tuple[int, CIFComment]
type _Mode = Literal["cif1", "cif2", "mcif"]

_CIF2_MAGIC = r"\#CIF_2.0"
_RESERVED_PREFIXES = ("data_", "save_", "loop_", "global_", "stop_")
_NUMBER_START = frozenset("+-.0123456789")
# Lists, tables and save frames together nest at most this deep (no RecursionError anywhere).
_MAX_NESTING = 64
_TOO_DEEP = f"CIF lists, tables or save frames nested deeper than {_MAX_NESTING}"


def _cif_space(char: str) -> bool:
    """Return whether ``char`` separates tokens within a line (any whitespace but ``\\n``)."""
    return char.isspace() and char != "\n"


def _is_cif2_header(text: str) -> bool:
    """Return whether a first-line comment text (after ``#``) is the CIF2 magic code."""
    return text == _CIF2_MAGIC or (text.startswith(_CIF2_MAGIC) and _cif_space(text[len(_CIF2_MAGIC)]))


def _cif_number(token: str) -> DecimalScalar | None:
    """Return the number a bare token denotes, or ``None`` for any other word (D7 included)."""
    if not token or token[0] not in _NUMBER_START:
        return None
    try:
        return DecimalScalar(token)
    except ValueError:  # not a NUMBER, or its exponent is outside Decimal's range (D7)
        return None


def _cif_bare(text: str, mode: _Mode) -> bool:
    """Return whether a string may be written unquoted and re-read as the same string."""
    return (
        bool(text)
        and not any(_cif_space(c) or c in "\n#]" for c in text)
        and text[0] not in "_$'\";["
        and not (mode == "cif2" and any(c in "[]{}" for c in text))
        and text not in ("?", ".")
        and _cif_number(text) is None
        and not text.lower().startswith(_RESERVED_PREFIXES)
    )


def _cif_quote(text: str, mode: _Mode, *, key: bool = False) -> str:
    """Return the CIF spelling of a string (or a CIF2 table key) in ``mode``.

    :param text: The string.
    :param mode: The syntax it is written in.
    :param key: Whether it is a CIF2 table key, which is always quoted.
    :return: The string bare or quoted, as it is written.
    :raises ValueError: If no quoting re-reads as the same string.
    """
    if "\r" not in text:
        if not key and _cif_bare(text, mode):
            return text
        newline = "\n" in text
        for quote in "'\"":
            if mode == "cif2":
                if quote not in text and not newline:
                    return quote + text + quote
            elif not newline and not any(
                text[i] == quote and (_cif_space(text[i + 1]) or (mode == "mcif" and text[i + 1] == "]"))
                for i in range(len(text) - 1)
            ):
                return quote + text + quote
        if mode == "cif2":
            for quote in ("'''", '"""'):
                if quote not in text and not text.endswith(quote[0]):
                    return quote + text + quote
    syntax = {"cif1": "CIF1", "cif2": "CIF2", "mcif": "magnetic CIF"}[mode]
    raise ValueError(f"the string {text[:60]!r} cannot be written in {syntax} syntax")


def _check_utf8(text: str) -> str:
    """Return ``text``, or raise for a string UTF-8 cannot encode (a lone surrogate)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"CIF text must be encodable as UTF-8: {text[:60]!r}") from None
    return text


def _check_name(name: object, what: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"a CIF {what} must be a str, not {type(name).__name__}")
    _check_utf8(name)
    if not name or any(c.isspace() for c in name):
        raise ValueError(f"a CIF {what} must be non-empty without whitespace: {name!r}")
    return name


def _check_data_name(name: object) -> str:
    name = _check_name(name, "data name")
    if len(name) < 2 or name[0] != "_":
        raise ValueError(f"a CIF data name starts with '_' and has at least one more character: {name!r}")
    return name


def _normalize_value(value: object, depth: int = 0) -> CIFValue:
    """Return ``value`` with nested tuples normalised, or raise for a value kind CIF does not have."""
    if isinstance(value, DecimalScalar):
        if isinstance(value.value, DecimalGap):
            raise ValueError("a CIF gap is a DecimalGap value, not a DecimalScalar")
        return value
    if isinstance(value, str):
        return _check_utf8(value)
    if isinstance(value, (CIFTextField, DecimalGap)):
        return value
    if isinstance(value, (tuple, CIFTable)):
        if depth >= _MAX_NESTING:
            raise ValueError(_TOO_DEEP)
        if isinstance(value, tuple):
            return tuple(_normalize_value(entry, depth + 1) for entry in value)
        for _, entry in value.entries:
            _normalize_value(entry, depth + 1)
        return value
    raise TypeError(
        f"a CIF value is a str, CIFTextField, DecimalScalar, DecimalGap, tuple or CIFTable, not {type(value).__name__}"
    )


def _normalize_comments(comments: Iterable[_PlacedComment], end: int) -> tuple[_PlacedComment, ...]:
    """Validate placed comments against the §3 invariants for a container with ``end`` positions."""
    placed = tuple((position, comment) for position, comment in comments)
    previous: _PlacedComment | None = None
    for position, comment in placed:
        if type(position) is not int or not isinstance(comment, CIFComment):
            raise TypeError("a placed CIF comment is a (position, CIFComment) pair")
        if not 0 <= position <= end:
            raise ValueError(f"CIF comment position {position} is outside 0..{end}")
        if not comment.inline and position == end:
            raise ValueError("an own-line CIF comment after the last token belongs to the enclosing item sequence")
        if previous is not None:
            if position < previous[0]:
                raise ValueError("placed CIF comment positions must not decrease")
            if position == previous[0] and comment.inline:
                raise ValueError("an inline CIF comment must come first at its position, once")
        previous = (position, comment)
    return placed


def _check_nesting(items: tuple[CIFItem, ...], depth: int) -> None:
    for item in items:
        if isinstance(item, CIFDataItem):
            _normalize_value(item.value, depth)
        elif isinstance(item, CIFLoop):
            for row in item.rows:
                for value in row:
                    _normalize_value(value, depth)
        elif isinstance(item, CIFSaveFrame):
            if depth >= _MAX_NESTING:
                raise ValueError(_TOO_DEEP)
            _check_nesting(item.items, depth + 1)


def _normalize_items(items: Iterable[CIFItem], depth: int) -> tuple[CIFItem, ...]:
    """Validate the type, placement and nesting rules of a block's or frame's item sequence."""
    result = tuple(items)
    for index, item in enumerate(result):
        if not isinstance(item, (CIFDataItem, CIFLoop, CIFComment, CIFSaveFrame)):
            raise TypeError(
                f"a CIF item is a CIFDataItem, CIFLoop, CIFComment or CIFSaveFrame, not {type(item).__name__}"
            )
        if isinstance(item, CIFComment) and item.inline and index and not isinstance(result[index - 1], CIFSaveFrame):
            raise ValueError("an inline CIF comment item may only come first or directly after a save frame")
        if (
            isinstance(item, CIFLoop)
            and not item.rows
            and index + 1 < len(result)
            and isinstance(result[index + 1], (CIFDataItem, CIFComment))
        ):
            # Re-reading would absorb the following item into the loop header.
            raise ValueError("a CIF loop without rows cannot be followed by a data item or comment")
    _check_nesting(result, depth)
    return result


def _check_mode_value(value: CIFValue, mode: _Mode) -> None:
    if isinstance(value, str):
        _cif_quote(value, mode)
    elif isinstance(value, tuple):
        if mode == "cif1":
            raise ValueError("CIF lists need dialect 'mcif' or the CIF2 header")
        for entry in value:
            _check_mode_value(entry, mode)
    elif isinstance(value, CIFTable):
        if mode != "cif2":
            raise ValueError("CIF tables need dialect 'cif' with the CIF2 header")
        for key, entry in value.entries:
            _cif_quote(key, mode, key=True)
            _check_mode_value(entry, mode)


def _check_mode_items(items: tuple[CIFItem, ...], mode: _Mode) -> None:
    for item in items:
        if isinstance(item, CIFDataItem):
            _check_mode_value(item.value, mode)
        elif isinstance(item, CIFLoop):
            for row in item.rows:
                for value in row:
                    _check_mode_value(value, mode)
        elif isinstance(item, CIFSaveFrame):
            _check_mode_items(item.items, mode)


@dataclass(frozen=True)
class CIFComment:
    """A CIF comment.

    :param text: Everything after ``#``, without trailing whitespace or line breaks.
    :param inline: Whether the comment shares its line with the preceding token.
    """

    text: str
    inline: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or type(self.inline) is not bool:
            raise TypeError("CIFComment takes a str text and a bool inline flag")
        if "\n" in self.text or "\r" in self.text or self.text != self.text.rstrip():
            raise ValueError(f"CIF comment text has a line break or trailing whitespace: {self.text[:60]!r}")
        _check_utf8(self.text)


@dataclass(frozen=True)
class CIFTextField:
    """A CIF semicolon text field.

    :param text: The raw characters between the delimiters: the rest of the opening ``;`` line,
        then each following line joined by ``\\n``, up to the line break before the closing ``;``.
        Line folding and prefixes are not decoded.
    """

    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("CIFTextField takes a str")
        if "\n;" in self.text or "\r" in self.text:
            raise ValueError("a CIF text field cannot contain a line starting with ';' or a carriage return")
        _check_utf8(self.text)


@dataclass(frozen=True)
class CIFTable:
    """A CIF2 table: unique string keys with values, in written order.

    :param entries: The ``(key, value)`` pairs.
    """

    entries: tuple[tuple[str, CIFValue], ...]

    def __post_init__(self) -> None:
        entries = tuple((key, _normalize_value(value, 1)) for key, value in self.entries)
        if not all(isinstance(key, str) for key, _ in entries):
            raise TypeError("CIF table keys are str")
        for key, _ in entries:
            _check_utf8(key)
        if len({key for key, _ in entries}) != len(entries):
            raise ValueError("CIF table keys must be unique")
        object.__setattr__(self, "entries", entries)


@dataclass(frozen=True)
class CIFDataItem:
    """A data name with its value.

    :param name: The data name as written, starting with ``_``.
    :param value: The value.
    :param comments: Placed comments: ``(0, c)`` between the name and the value, ``(1, c)``
        inline after the value.
    """

    name: str
    value: CIFValue
    comments: tuple[tuple[int, CIFComment], ...] = ()

    def __post_init__(self) -> None:
        _check_data_name(self.name)
        object.__setattr__(self, "value", _normalize_value(self.value))
        object.__setattr__(self, "comments", _normalize_comments(self.comments, 1))


@dataclass(frozen=True)
class CIFLoop:
    """A CIF loop: data names and rows of values.

    :param names: The data names as written.
    :param rows: The rows, each with one value per name; there may be none.
    :param comments: Placed comments: position ``0`` follows ``loop_``, ``1..n`` the n-th name and
        ``n+j`` the j-th value, row-major.
    """

    names: tuple[str, ...]
    rows: tuple[tuple[CIFValue, ...], ...]
    comments: tuple[tuple[int, CIFComment], ...] = ()

    def __post_init__(self) -> None:
        names = tuple(_check_data_name(name) for name in self.names)
        if not names:
            raise ValueError("a CIF loop needs at least one data name")
        rows = tuple(tuple(_normalize_value(value) for value in row) for row in self.rows)
        if any(len(row) != len(names) for row in rows):
            raise ValueError(f"every CIF loop row needs {len(names)} values")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "comments", _normalize_comments(self.comments, len(names) * (1 + len(rows))))


def _locate(items: tuple[CIFItem, ...], name: str) -> tuple[int, CIFDataItem | CIFLoop, int]:
    """Return ``(item index, item, column index or -1)`` for a unique top-level data name."""
    target = name.casefold()
    found = [
        (index, item, column)
        for index, item in enumerate(items)
        if isinstance(item, (CIFDataItem, CIFLoop))
        for column, written in enumerate(item.names if isinstance(item, CIFLoop) else (item.name,))
        if written.casefold() == target
    ]
    if not found:
        raise KeyError(name)
    if len(found) > 1:
        raise ValueError(f"CIF data name {name!r} occurs {len(found)} times")
    index, item, column = found[0]
    return index, item, column if isinstance(item, CIFLoop) else -1


def _value(items: tuple[CIFItem, ...], name: str) -> CIFValue:
    item = _locate(items, name)[1]
    if not isinstance(item, CIFDataItem):
        raise ValueError(f"CIF data name {name!r} is a loop column; use column()")
    return item.value


def _column(items: tuple[CIFItem, ...], name: str) -> tuple[CIFValue, ...]:
    _, item, column = _locate(items, name)
    if not isinstance(item, CIFLoop):
        raise ValueError(f"CIF data name {name!r} is not a loop column; use value()")
    return tuple(row[column] for row in item.rows)


def _numeric(value: CIFValue, name: str) -> DecimalScalar:
    if isinstance(value, DecimalScalar):
        return value
    if isinstance(value, DecimalGap):
        return DecimalScalar(value)
    raise ValueError(f"CIF data name {name!r} holds a {type(value).__name__}, not a number")


def _numbers(items: tuple[CIFItem, ...], names: tuple[str, ...]) -> DecimalVector:
    located = [_locate(items, name) for name in names]
    if located and all(column < 0 for _, _, column in located):
        return DecimalVector(
            [_numeric(cast(CIFDataItem, item).value, name) for name, (_, item, _) in zip(names, located)]
        )
    if located and all(column >= 0 for _, _, column in located) and len({index for index, _, _ in located}) == 1:
        rows = cast(CIFLoop, located[0][1]).rows
        columns = [column for _, _, column in located]
        entries = [[_numeric(row[column], name) for name, column in zip(names, columns)] for row in rows]
        return DecimalVector([entry[0] for entry in entries] if len(names) == 1 else entries)
    raise ValueError("numbers() takes data items only, or columns of one loop")


@dataclass(frozen=True)
class CIFSaveFrame:
    """A CIF save frame; frames may nest.

    :param name: The frame name as written (after ``save_``).
    :param items: The frame's items in written order.
    """

    name: str
    items: tuple[CIFItem, ...]

    def __post_init__(self) -> None:
        _check_name(self.name, "save frame name")
        object.__setattr__(self, "items", _normalize_items(self.items, 1))

    def find(self, name: str) -> CIFDataItem | CIFLoop:
        """Return the top-level data item or loop holding ``name`` (caseless).

        :param name: The data name.
        :return: The data item, or the loop with that column.
        :raises KeyError: If the name is absent.
        :raises ValueError: If the name occurs more than once.
        """
        return _locate(self.items, name)[1]

    def value(self, name: str) -> CIFValue:
        """Return the value of a top-level data item.

        :param name: The data name (caseless).
        :return: The value.
        :raises KeyError: If the name is absent.
        :raises ValueError: If the name is repeated or is a loop column.
        """
        return _value(self.items, name)

    def column(self, name: str) -> tuple[CIFValue, ...]:
        """Return one column of a top-level loop.

        :param name: The column's data name (caseless).
        :return: The column values, one per row.
        :raises KeyError: If the name is absent.
        :raises ValueError: If the name is repeated or is not a loop column.
        """
        return _column(self.items, name)

    def number(self, name: str) -> DecimalScalar:
        """Return the number, or gap, of a top-level data item.

        :param name: The data name (caseless).
        :return: The number as written; a gap as a gap-valued scalar.
        :raises ValueError: If the value is not a number or gap.
        """
        return _numeric(self.value(name), name)

    def numbers(self, *names: str) -> DecimalVector:
        r"""Return numbers as a vector.

        ``k`` data items give shape ``(k,)``, one loop column ``(rows,)`` and ``k`` columns of
        one loop ``(rows, k)``.

        :param \*names: The data names (caseless).
        :return: The numbers as written, gaps included.
        :raises ValueError: For any other combination, or a value that is not a number or gap.
        """
        return _numbers(self.items, names)


@dataclass(frozen=True)
class CIFBlock:
    """One CIF data block, losslessly: its items in written order and the comments around it.

    :param name: The block name as written (after ``data_``).
    :param items: The block's items in written order; the inline comment on the ``data_`` line,
        if any, comes first.
    :param header: The file's header comment lines (the run of comment lines from line 1),
        carried by every block of the file so a block can be written alone.
    :param leading: The own-line comments between the header (or the previous block) and the
        ``data_`` line.
    :param dialect: ``"cif"`` (CIF1, or full CIF2 when the header's first line is the CIF2 magic
        code) or ``"mcif"`` (CIF1 plus lists, as the magnetic CIF reader reads it).
    :raises ValueError: If a value, name or comment placement cannot be written in the block's
        syntax, such as a table outside CIF2 or a list in CIF1.
    """

    name: str
    items: tuple[CIFItem, ...]
    header: tuple[CIFComment, ...] = ()
    leading: tuple[CIFComment, ...] = ()
    dialect: CIFDialect = "cif"

    def __post_init__(self) -> None:
        _check_name(self.name, "block name")
        if self.dialect not in ("cif", "mcif"):
            raise ValueError(f"CIF dialect must be 'cif' or 'mcif', not {self.dialect!r}")
        for attribute in ("header", "leading"):
            comments = tuple(getattr(self, attribute))
            if not all(isinstance(comment, CIFComment) and not comment.inline for comment in comments):
                raise ValueError(f"CIF block {attribute} holds own-line CIFComments only")
            object.__setattr__(self, attribute, comments)
        object.__setattr__(self, "items", _normalize_items(self.items, 0))
        _check_mode_items(self.items, _cif_mode(self))

    def find(self, name: str) -> CIFDataItem | CIFLoop:
        """Return the top-level data item or loop holding ``name`` (caseless).

        :param name: The data name.
        :return: The data item, or the loop with that column.
        :raises KeyError: If the name is absent.
        :raises ValueError: If the name occurs more than once.
        """
        return _locate(self.items, name)[1]

    def value(self, name: str) -> CIFValue:
        """Return the value of a top-level data item.

        :param name: The data name (caseless).
        :return: The value.
        :raises KeyError: If the name is absent.
        :raises ValueError: If the name is repeated or is a loop column.
        """
        return _value(self.items, name)

    def column(self, name: str) -> tuple[CIFValue, ...]:
        """Return one column of a top-level loop.

        :param name: The column's data name (caseless).
        :return: The column values, one per row.
        :raises KeyError: If the name is absent.
        :raises ValueError: If the name is repeated or is not a loop column.
        """
        return _column(self.items, name)

    def number(self, name: str) -> DecimalScalar:
        """Return the number, or gap, of a top-level data item.

        :param name: The data name (caseless).
        :return: The number as written; a gap as a gap-valued scalar.
        :raises ValueError: If the value is not a number or gap.
        """
        return _numeric(self.value(name), name)

    def numbers(self, *names: str) -> DecimalVector:
        r"""Return numbers as a vector.

        ``k`` data items give shape ``(k,)``, one loop column ``(rows,)`` and ``k`` columns of
        one loop ``(rows, k)``.

        :param \*names: The data names (caseless).
        :return: The numbers as written, gaps included.
        :raises ValueError: For any other combination, or a value that is not a number or gap.
        """
        return _numbers(self.items, names)


def _cif_mode(block: CIFBlock) -> _Mode:
    """Return the syntax a block is read and written in."""
    if block.dialect == "mcif":
        return "mcif"
    return "cif2" if block.header and _is_cif2_header(block.header[0].text) else "cif1"
