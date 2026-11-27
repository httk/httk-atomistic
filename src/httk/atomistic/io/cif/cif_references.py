"""Extract the bibliographic references a CIF data block states as reference records.

The publication the block describes (the ``_journal_*``/``_publ_*`` items) and every row of the
``citation`` category become :class:`httk.core.ReferenceRecord` objects. Text is kept as written,
except that CIF 1.1 special-character codes are decoded to Unicode and layout whitespace is
collapsed.
"""

import logging
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from httk.core import DecimalGap, DecimalScalar, PersonRecord, ReferenceRecord

from .cif_items import CIFBlock, CIFDataItem, CIFLoop, CIFTextField, CIFValue

__all__ = ["cif_references"]

logger = logging.getLogger(__name__)

# One reference's source: the (data name as written, value) pairs stated for it, gaps left out.
type _Pairs = list[tuple[str, CIFValue]]

# The CIF 1.1 special-character codes ("Common semantic features" of the CIF 1.1 specification).
_GREEK = dict(zip("abgdezhqiklmnxoprstufcyw", "αβγδεζηθικλμνξοπρστυφχψω", strict=True))
_LETTERS = {
    **_GREEK,
    **{code.upper(): letter.upper() for code, letter in _GREEK.items()},
    "%a": "å",
    "%A": "Å",
    "/o": "ø",
    "/O": "Ø",
    "/l": "ł",
    "/L": "Ł",
    "/d": "đ",
    "/D": "Đ",
    "?i": "ı",
    "&s": "ß",
}
_ACCENTS = {
    "'": "́",  # acute
    "`": "̀",  # grave
    "^": "̂",  # circumflex
    '"': "̈",  # umlaut
    "~": "̃",  # tilde
    ",": "̧",  # cedilla
    ";": "̨",  # ogonek
    ">": "̋",  # double acute
    "=": "̄",  # overbar
    ".": "̇",  # overdot
    "<": "̌",  # caron
    "(": "̆",  # breve
}
# ``\\`` (which starts codes such as ``\\db``) is matched first so that its second backslash never
# starts a letter code; it and every undefined sequence are kept as written. ``\%`` not before a
# letter is the degree sign.
_MARKUP = re.compile(
    r"\\\\|\\%(?![^\W\d_])|\\[%/?&][A-Za-z]|\\[A-Za-z]|\\[" + re.escape("".join(_ACCENTS)) + r"][^\s\\]"
)

# Reference field -> the data names it is read from, the first one present winning; ``pages``
# names the first and last page.
_PUBLICATION = {
    "title": ("_publ_section_title",),
    "journal": ("_journal_name_full", "_journal_name_abbrev"),
    "volume": ("_journal_volume",),
    "number": ("_journal_issue",),
    "year": ("_journal_year",),
    "pages": ("_journal_page_first", "_journal_page_last"),
    "doi": ("_journal_paper_doi",),
}
_CITATION = {
    "title": ("_citation_title",),
    "journal": ("_citation_journal_full", "_citation_journal_abbrev"),
    "volume": ("_citation_journal_volume",),
    "number": ("_citation_journal_issue",),
    "year": ("_citation_year",),
    "pages": ("_citation_page_first", "_citation_page_last"),
    "doi": ("_citation_doi",),
    "booktitle": ("_citation_book_title",),
    "publisher": ("_citation_book_publisher", "_citation_publisher"),
    "address": ("_citation_book_publisher_city",),
    "note": ("_citation_special_details",),
}


def _decode(match: re.Match[str]) -> str:
    code = match.group()[1:]
    if code == "%":
        return "°"
    if code in _LETTERS:
        return _LETTERS[code]
    following = match.string[match.end() : match.end() + 1]
    if (
        len(code) == 2
        and code[0] in _ACCENTS
        and code[1].isalpha()
        and unicodedata.normalize("NFD", code[1]) == code[1]  # an unaccented base letter
        and not (following and unicodedata.category(following).startswith("M"))  # not a combining sequence
    ):
        return unicodedata.normalize("NFC", code[1] + _ACCENTS[code[0]])
    return match.group()


def _written(value: CIFValue) -> str:
    """Return a value as written in the file (a list or table as its repr)."""
    if isinstance(value, str):
        return value
    if isinstance(value, CIFTextField):
        return value.text
    if isinstance(value, DecimalScalar):
        return value.token
    return repr(value)


def _text(value: CIFValue, *, markup: bool = True) -> str | None:
    """Return a value as reference text, or None when it is blank; raise ValueError for a list or table."""
    if not isinstance(value, (str, CIFTextField, DecimalScalar)):
        raise ValueError(f"a {type(value).__name__} cannot be reference text")
    text = " ".join(_written(value).split())
    return (_MARKUP.sub(_decode, text) if markup else text) or None


def _rows(
    block: CIFBlock, prefixes: tuple[str, ...], exclude: tuple[str, ...] = (), *, attach: bool = False
) -> list[_Pairs]:
    """Return the items with a name prefix: the top-level data items as one row, then the rows of loops.

    Names are caseless; gaps and empty rows are left out. With ``attach``, the top-level items instead
    join the one loop row they unambiguously belong to (the only loop row, else the only one whose
    ``_citation_id`` is ``primary``) when that row's loop has none of their names as a column.
    """

    def belongs(name: str) -> bool:
        return name.casefold().startswith(prefixes) and not name.casefold().startswith(exclude)

    def stated(pairs: _Pairs) -> _Pairs:
        return [(name, value) for name, value in pairs if not isinstance(value, DecimalGap)]

    loose: _Pairs = []
    loops: list[tuple[set[str], list[_Pairs]]] = []
    for item in block.items:
        if isinstance(item, CIFDataItem) and belongs(item.name):
            loose.append((item.name, item.value))
        elif isinstance(item, CIFLoop):
            columns = [index for index, name in enumerate(item.names) if belongs(name)]
            if columns:
                rows = [stated([(item.names[index], row[index]) for index in columns]) for row in item.rows]
                loops.append(({item.names[index].casefold() for index in columns}, [row for row in rows if row]))
    loose = stated(loose)
    if attach and loose:
        candidates = [(names, row) for names, rows in loops for row in rows]
        if len(candidates) > 1:
            candidates = [
                (names, row) for names, row in candidates if (_key(row, "_citation_id") or "").casefold() == "primary"
            ]
        if len(candidates) == 1 and not any(name.casefold() in candidates[0][0] for name, _ in loose):
            candidates[0][1].extend(loose)
            loose = []
    return [row for row in [loose, *(row for _, rows in loops for row in rows)] if row]


def _values(pairs: _Pairs, name: str) -> list[CIFValue]:
    return [value for written, value in pairs if written.casefold() == name]


def _field(pairs: _Pairs, *names: str, markup: bool = True) -> str | None:
    """Return the text of the first of ``names`` present in ``pairs``; raise ValueError for a repeated name."""
    for name in names:
        values = _values(pairs, name)
        if len(values) > 1:
            raise ValueError(f"{name} is given {len(values)} times")
        if values and (text := _text(values[0], markup=markup)) is not None:
            return text
    return None


def _key(row: _Pairs, name: str) -> str | None:
    """Return the citation id a row states under ``name``, or None."""
    values = _values(row, name)
    return " ".join(_written(values[0]).split()) if len(values) == 1 else None


def _persons(names: list[CIFValue]) -> tuple[PersonRecord, ...] | None:
    persons: list[PersonRecord] = []
    for value in names:
        name = _text(value)
        if name is not None:
            parts = [part.strip() for part in name.split(",")]
            split = len(parts) == 2 and all(parts)
            persons.append(PersonRecord(name, parts[1] if split else None, parts[0] if split else None))
    return tuple(persons) or None


def _reference(
    pairs: _Pairs, mapping: Mapping[str, tuple[str, ...]], authors: list[CIFValue], editors: list[CIFValue]
) -> ReferenceRecord:
    """Build one reference from its pairs; raise ValueError when nothing fits a field."""
    fields: dict[str, Any] = {
        field: _field(pairs, *names, markup=field != "doi") for field, names in mapping.items() if field != "pages"
    }
    first, last = (_field(pairs, name) for name in mapping["pages"])
    fields["pages"] = f"{first}-{last}" if first and last else first or last
    fields["authors"] = _persons(authors)
    fields["editors"] = _persons(editors)
    if fields["journal"] is not None:
        fields["bib_type"] = "article"
    stated = {field: value for field, value in fields.items() if value is not None}
    if not stated:
        raise ValueError("no item fits a reference field")
    return ReferenceRecord(**stated)


def cif_references(block: CIFBlock) -> tuple[ReferenceRecord, ...]:
    """Return the bibliographic references a CIF data block states.

    The publication the block describes (``_publ_section_title``, ``_publ_author_name`` and the
    ``_journal_*`` items) comes first, then one reference per row of the ``citation`` category
    (``_citation_*``, looped or single; single items join the only looped row, else the only
    ``primary`` one, when its loop does not also state them), with the names of the ``_citation_author_*`` and
    ``_citation_editor_*`` rows that state its ``_citation_id`` (id-less rows go with the only
    id-less citation; other unmatched people form a reference of their own). Exact duplicates are returned
    once. Values are kept as written, except that CIF 1.1 special-character codes (accents,
    special letters, Greek letters, the degree sign; not in DOIs) are decoded to Unicode and runs of whitespace
    become one space. A person name with exactly one comma also gives ``lastname, firstname``.
    Items without a fitting field are dropped. A reference that would be empty or cannot be
    represented is not returned; instead an INFO note (report context ``"cif"``) lists the data
    stated for it.

    :param block: The data block, for example a :class:`~httk.atomistic.CIFStructure`.
    :return: The references, in order.
    """
    publication = [pair for row in _rows(block, ("_journal_", "_publ_author_", "_publ_section_title")) for pair in row]
    sources: list[tuple[_Pairs, Mapping[str, tuple[str, ...]], list[CIFValue], list[CIFValue], _Pairs]] = [
        (publication, _PUBLICATION, _values(publication, "_publ_author_name"), [], publication)
    ]
    citations = _rows(block, ("_citation_",), ("_citation_author_", "_citation_editor_"), attach=True)
    keys: list[object] = [_key(row, "_citation_id") for row in citations]
    if keys.count(None) > 1:  # id-less people then belong to none of them
        keys = [object() if key is None else key for key in keys]
    people: dict[str, dict[object, list[_Pairs]]] = {"author": {}, "editor": {}}
    for kind, groups in people.items():
        prefix = f"_citation_{kind}_"
        for row in _rows(block, (prefix,)):
            person_key = _key(row, f"{prefix}citation_id")
            groups.setdefault(person_key, []).append(row)
            if person_key not in keys:  # people of a citation without a row of its own
                keys.append(person_key)
                citations.append([])
    for pairs, citation_key in zip(citations, keys, strict=True):
        author_rows, editor_rows = people["author"].get(citation_key, []), people["editor"].get(citation_key, [])
        sources.append(
            (
                pairs,
                _CITATION,
                [value for row in author_rows for value in _values(row, "_citation_author_name")],
                [value for row in editor_rows for value in _values(row, "_citation_editor_name")],
                pairs + [pair for row in author_rows + editor_rows for pair in row],
            )
        )
    references: list[ReferenceRecord] = []
    for pairs, mapping, authors, editors, stated in sources:
        if not stated:
            continue
        try:
            references.append(_reference(pairs, mapping, authors, editors))
        except (TypeError, ValueError) as error:
            logger.info(
                "CIF block %r: a reference could not be represented (%s); its data: %s",
                block.name,
                error,
                "; ".join(f"{name} = {_written(value)!r}" for name, value in stated),
                extra={"context": "cif"},
            )
    return tuple(dict.fromkeys(references))
