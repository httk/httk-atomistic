"""Read the geometry of CP2K input and restart files; write it as a ``&SUBSYS`` block.

CP2K restart files use the input syntax, so one reader serves both. The reader returns a
neutral geometry payload (see ``httk.atomistic.integrations._geometry``) tagged
``"cp2k-input"``; constructs whose geometry it cannot reproduce exactly are refused.
"""

import io
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from httk.atomistic.elements import SYMBOLS
from httk.atomistic.integrations._geometry import (
    _element_from_label,
    _geometry_from_structure,
    _number,
    _structure_from_geometry,
    _write_text,
)
from httk.atomistic.integrations.vasp.io._text import source_lines
from httk.atomistic.io import _check_as_written, _check_load_precision

_TAG = "cp2k-input"
_COMMENT = re.compile(r"[!#].*")
_TRUE = frozenset({"", "T", "TRUE", ".TRUE.", "Y", "YES", "ON"})
_FALSE = frozenset({"F", "FALSE", ".FALSE.", "N", "NO", "OFF"})
_PERIODIC = frozenset({"XYZ", "XY", "XZ", "YZ", "X", "Y", "Z", "NONE"})
_KIND_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]*")


@dataclass
class _Section:
    """One ``&NAME parameter`` … ``&END`` section: keyword lines and subsections in order."""

    name: str
    parameter: str
    line: int
    keywords: list[tuple[int, list[str]]] = field(default_factory=list)
    sections: list["_Section"] = field(default_factory=list)

    def all(self, name: str) -> list["_Section"]:
        """The subsections called ``name``."""
        return [section for section in self.sections if section.name == name]

    def only(self, name: str) -> "_Section | None":
        """The single subsection called ``name``, or ``None``; refuse repeats."""
        found = self.all(name)
        if len(found) > 1:
            raise ValueError(f"CP2K input line {found[1].line}: more than one &{name} section in &{self.name}")
        return found[0] if found else None

    def keyword(self, name: str) -> tuple[int, list[str]] | None:
        """The ``(line, value tokens)`` of keyword ``name``, or ``None``; refuse repeats."""
        found = [(line, tokens[1:]) for line, tokens in self.keywords if tokens[0].upper() == name]
        if len(found) > 1:
            raise ValueError(f"CP2K input line {found[1][0]}: keyword {name} repeated in &{self.name}")
        return found[0] if found else None


def read_cp2k_input(source: Any, *, precision: float | None = None, as_written: bool = False) -> dict[str, Any]:
    """Read the structure of a CP2K input or restart file into a neutral geometry payload.

    The geometry is the single ``&FORCE_EVAL/&SUBSYS`` section (or a file holding just one
    ``&SUBSYS`` block, as the ``cp2k-input`` writer produces): ``&CELL`` from ``A``/``B``/``C``
    vectors or ``ABC`` with ``ALPHA_BETA_GAMMA`` (payload ``cell_parameters``, built as for
    CIF in CP2K's frame: ``a`` along x, ``b`` in the xy plane), ``PERIODIC``, ``&COORD`` (Cartesian or
    ``SCALED``), and each atom's element from its ``&KIND`` ``ELEMENT`` keyword or else its
    kind name. Lengths must be in Å, CP2K's default. Keyword names are case-insensitive;
    ``!``/``#`` comments and ``\\`` line continuations are honoured.

    Refused: preprocessor directives (``@INCLUDE``, ``@SET``, ``@IF``, …) and ``$`` variables,
    several ``&FORCE_EVAL`` or ``&SUBSYS`` sections, external coordinates (``&TOPOLOGY``
    ``COORD_FILE_NAME``/``COORD_FILE_FORMAT``), ``MULTIPLE_UNIT_CELL`` other than ``1 1 1``,
    ``&CELL_REF``, ``CELL_FILE_NAME``, ``SYMMETRY`` other than ``NONE``, units other than
    ``[angstrom]`` (and ``[deg]`` for angles), ghost kinds, and atoms whose element cannot be
    determined.

    :param source: A path (compressed files are decompressed) or an open text or binary stream.
    :param precision: Cartesian coordinate precision in Å overriding the digit-derived one,
        or ``None``. Must be a finite number greater than zero when given.
    :param as_written: Already exact; accepted for a uniform load interface.
    :return: The geometry payload, with the original text in ``raw``.
    :raises ValueError: If the source is not a CP2K input, is malformed, uses a refused
        construct, or ``precision`` is invalid.
    :raises TypeError: If ``source`` is neither a path nor a stream.
    """
    _check_load_precision(precision, "read_cp2k_input")
    _check_as_written(as_written, "read_cp2k_input")
    try:
        with source_lines(source, preserve_path=True, capture_stream=True) as (_, raw):
            pass
    except UnicodeDecodeError:
        raise ValueError("not a CP2K input: the source is not UTF-8 text") from None
    if raw is None:
        raise TypeError(f"CP2K input source must be a path or a stream, got {type(source).__name__}")
    subsys = _subsys(_parse(raw))
    cell, periodicity = _cell(subsys)
    labels, positions, scaled = _coord(subsys)
    # CP2K upper-cases kind names: "si" and "SI" are one kind, named by its first spelling.
    first: dict[str, str] = {}
    names = [first.setdefault(label.upper(), label) for label in labels]
    return {
        "format": _TAG,
        **cell,
        "cell_unit": "angstrom",
        "positions": positions,
        "positions_kind": "fractional" if scaled else "cartesian",
        "positions_unit": "angstrom",
        "species": _species(subsys, names),
        "species_at_sites": names,
        "periodicity": periodicity,
        "precision_override": precision,
        "raw": raw,
    }


def _parse(text: str) -> _Section:
    """Parse CP2K input text into its section tree (an unnamed root)."""
    logical: list[tuple[int, str]] = []
    pending, first = "", 0
    for number, line in enumerate(text.splitlines(), 1):
        if not pending:
            first = number
        content = pending + _COMMENT.sub("", line).strip()
        if content.endswith("\\"):
            pending = content[:-1] + " "
            continue
        pending = ""
        if content.strip():
            logical.append((first, content.strip()))
    if pending.strip():
        logical.append((first, pending.strip()))
    if not any(content.startswith("&") for _, content in logical):
        raise ValueError("not a CP2K input: no &SECTION lines")
    for number, content in logical:
        if content.startswith("@"):
            raise ValueError(
                f"CP2K input line {number}: preprocessor directive {content.split()[0]} is not supported; "
                "read a preprocessed input such as the .restart file CP2K writes"
            )
        if "$" in content:
            raise ValueError(f"CP2K input line {number}: preprocessor variables ($...) are not supported")

    root = _Section("", "", 0)
    stack = [root]
    for number, content in logical:
        if not content.startswith("&"):
            if len(stack) == 1:
                raise ValueError(f"CP2K input line {number}: content outside any &SECTION: {content!r}")
            stack[-1].keywords.append((number, content.split()))
            continue
        head, _, rest = content[1:].replace("\t", " ").partition(" ")
        if head.upper() != "END":
            section = _Section(head.upper(), rest.strip(), number)
            stack[-1].sections.append(section)
            stack.append(section)
            continue
        if len(stack) == 1:
            raise ValueError(f"CP2K input line {number}: &END without an open section")
        closing = rest.split()
        if closing and closing[0].upper() != stack[-1].name:
            raise ValueError(f"CP2K input line {number}: &END {closing[0]} closes &{stack[-1].name}")
        stack.pop()
    if len(stack) > 1:
        raise ValueError(f"CP2K input line {stack[-1].line}: &{stack[-1].name} is never closed (&END)")
    return root


def _subsys(root: _Section) -> _Section:
    """The single ``&FORCE_EVAL/&SUBSYS`` (or top-level ``&SUBSYS``) section."""
    force_evals = root.all("FORCE_EVAL")
    if len(force_evals) > 1:
        raise ValueError(
            f"CP2K input line {force_evals[1].line}: several &FORCE_EVAL sections are not supported (one geometry only)"
        )
    holder = force_evals[0] if force_evals else root
    subsys = holder.only("SUBSYS")
    if subsys is None:
        raise ValueError("not a CP2K input: no &FORCE_EVAL/&SUBSYS section")
    topology = subsys.only("TOPOLOGY")
    if topology is not None:
        found = topology.keyword("COORD_FILE_NAME")
        if found is not None:
            raise ValueError(f"CP2K input line {found[0]}: external coordinates (COORD_FILE_NAME) are not supported")
        found = topology.keyword("COORD_FILE_FORMAT")
        if found is not None and [token.upper() for token in found[1]] != ["OFF"]:
            raise ValueError(f"CP2K input line {found[0]}: external coordinates (COORD_FILE_FORMAT) are not supported")
        _single_cell(topology)
    return subsys


def _single_cell(section: _Section) -> None:
    """Refuse a ``MULTIPLE_UNIT_CELL`` other than ``1 1 1``."""
    found = section.keyword("MULTIPLE_UNIT_CELL")
    if found is not None and found[1] != ["1", "1", "1"]:
        raise ValueError(f"CP2K input line {found[0]}: MULTIPLE_UNIT_CELL {' '.join(found[1])} is not supported")


def _values(found: tuple[int, list[str]], name: str, unit: str) -> list[str]:
    """Three decimal tokens of a keyword, after an optional ``[unit]`` that must be ``unit``."""
    line, tokens = found
    if tokens and tokens[0].startswith("["):
        given = tokens[0].lower()
        if given != f"[{unit}]":
            raise ValueError(f"CP2K input line {line}: {name} unit {tokens[0]} is not supported, only [{unit}]")
        tokens = tokens[1:]
    return _checked(line, name, tokens)


def _checked(line: int, name: str, tokens: list[str]) -> list[str]:
    """Three decimal tokens, validated."""
    if len(tokens) != 3:
        raise ValueError(f"CP2K input line {line}: {name} needs three numbers, got {tokens}")
    for token in tokens:
        try:
            _number(token)
        except ValueError as error:
            raise ValueError(f"CP2K input line {line}: {name}: {error}") from None
    return tokens


def _cell(subsys: _Section) -> tuple[dict[str, Any], list[bool]]:
    """The payload's ``cell`` or ``cell_parameters`` entry (Å, degrees) and periodicity of ``&SUBSYS/&CELL``."""
    cell = subsys.only("CELL")
    if cell is None:
        raise ValueError(f"CP2K input line {subsys.line}: &SUBSYS has no &CELL section")
    if cell.sections:
        raise ValueError(f"CP2K input line {cell.sections[0].line}: &{cell.sections[0].name} is not supported")
    _single_cell(cell)
    for name in ("CELL_FILE_NAME", "SYMMETRY"):
        found = cell.keyword(name)
        if found is not None and [token.upper() for token in found[1]] != ["NONE"]:
            # ponytail: SYMMETRY other than NONE is refused, not checked against the given cell.
            raise ValueError(f"CP2K input line {found[0]}: {name} {' '.join(found[1])} is not supported")
    found = cell.keyword("PERIODIC")
    periodic = "XYZ" if found is None else " ".join(found[1]).upper()
    if found is not None and periodic not in _PERIODIC:
        raise ValueError(f"CP2K input line {found[0]}: PERIODIC {periodic!r} is not one of {sorted(_PERIODIC)}")
    periodicity = [axis in periodic.replace("NONE", "") for axis in "XYZ"]

    vectors = [cell.keyword(name) for name in "ABC"]
    abc = cell.keyword("ABC")
    if abc is None:
        if None in vectors:
            raise ValueError(f"CP2K input line {cell.line}: &CELL needs A, B and C vectors or ABC")
        return {"cell": [_values(found, name, "angstrom") for found, name in zip(vectors, "ABC") if found]}, periodicity
    if any(vectors):
        raise ValueError(f"CP2K input line {abc[0]}: &CELL gives both ABC and A/B/C vectors")
    lengths = _values(abc, "ABC", "angstrom")
    found = cell.keyword("ALPHA_BETA_GAMMA")
    angles = ["90", "90", "90"] if found is None else _values(found, "ALPHA_BETA_GAMMA", "deg")
    # CellParams' frame (a along +x, b in the xy plane, c with positive z) is CP2K's, so
    # Cartesian &COORD rows are meant in it too.
    return {"cell_parameters": lengths + angles}, periodicity


def _coord(subsys: _Section) -> tuple[list[str], list[list[str]], bool]:
    """Site kind names, coordinate tokens and the ``SCALED`` flag of ``&SUBSYS/&COORD``."""
    coord = subsys.only("COORD")
    if coord is None:
        raise ValueError(f"CP2K input line {subsys.line}: &SUBSYS has no &COORD section")
    names: list[str] = []
    positions: list[list[str]] = []
    scaled = False
    for line, tokens in coord.keywords:
        key = tokens[0].upper()
        if key == "SCALED":
            value = " ".join(tokens[1:]).upper()
            if value not in _TRUE | _FALSE:
                raise ValueError(f"CP2K input line {line}: SCALED {value!r} is not a logical value")
            scaled = value in _TRUE
        elif key == "UNIT":
            if [token.lower() for token in tokens[1:]] != ["angstrom"]:
                raise ValueError(
                    f"CP2K input line {line}: &COORD UNIT {' '.join(tokens[1:])} is not supported, only angstrom"
                )
        elif len(tokens) in (4, 5):
            names.append(tokens[0])
            positions.append(_checked(line, f"&COORD atom {tokens[0]}", tokens[1:4]))
        else:
            raise ValueError(f"CP2K input line {line}: expected 'kind x y z [molecule]' in &COORD, got {tokens}")
    if not names:
        raise ValueError(f"CP2K input line {coord.line}: &COORD lists no atoms")
    return names, positions, scaled


def _species(subsys: _Section, names: list[str]) -> list[dict[str, Any]]:
    """One species per distinct site kind name, its element from ``&KIND ELEMENT`` or the name."""
    elements: dict[str, str | None] = {}
    for kind in subsys.all("KIND"):
        label = kind.parameter.split()[0].upper() if kind.parameter.split() else ""
        if not label or label in elements:
            raise ValueError(f"CP2K input line {kind.line}: &KIND needs a unique kind name, got {kind.parameter!r}")
        ghost = kind.keyword("GHOST")
        if ghost is not None and " ".join(ghost[1]).upper() in _TRUE:
            raise ValueError(f"CP2K input line {ghost[0]}: ghost kind {kind.parameter!r} is not supported")
        found = kind.keyword("ELEMENT")
        if found is not None and len(found[1]) != 1:
            raise ValueError(f"CP2K input line {found[0]}: ELEMENT needs one element symbol, got {found[1]}")
        elements[label] = None if found is None else found[1][0]
    species: list[dict[str, Any]] = []
    for name in dict.fromkeys(names):
        given = elements.get(name.upper())
        element = _element_from_label(name) if given is None else given.capitalize()
        if element not in SYMBOLS:
            reason = f"ELEMENT {given!r} is not an element" if given else "it has no ELEMENT and its name names none"
            raise ValueError(f"CP2K input: cannot determine the element of kind {name!r}: {reason}")
        species.append({"name": name, "element": element, "mass": None})
    return species


def _cp2k_input_payload_from_structure(obj: Any) -> dict[str, Any]:
    """Serialize a structure to a ``cp2k-input`` geometry payload.

    :param obj: Anything :class:`~httk.atomistic.UnitcellStructureView` accepts.
    :return: The geometry payload.
    """
    return _geometry_from_structure(obj, _TAG)


def _render_subsys(payload: Mapping[str, Any]) -> str:
    """Render a geometry payload as an ``&SUBSYS`` block (Å cell, scaled coordinates)."""
    # Re-normalizing through the structure makes any payload (Cartesian, Bohr, exact entries) fractional Å.
    data = _geometry_from_structure(_structure_from_geometry(payload), _TAG)
    if len({item["name"].upper() for item in data["species"]}) != len(data["species"]):
        raise ValueError(
            "cp2k-input cannot write species names differing only in letter case (CP2K kind names are not)"
        )
    for item in data["species"]:
        if not _KIND_NAME.fullmatch(item["name"]) or item["name"].upper() in ("SCALED", "UNIT"):
            raise ValueError(f"cp2k-input cannot write species name {item['name']!r} as a CP2K kind name")
    periodic = "".join(axis for axis, flag in zip("XYZ", data["periodicity"]) if flag) or "NONE"
    lines = ["&SUBSYS", "  &CELL"]
    lines += [f"    {name} {' '.join(row)}" for name, row in zip("ABC", data["cell"])]
    lines += [f"    PERIODIC {periodic}", "  &END CELL", "  &COORD", "    SCALED .TRUE."]
    lines += [f"    {name} {' '.join(row)}" for name, row in zip(data["species_at_sites"], data["positions"])]
    lines.append("  &END COORD")
    for item in data["species"]:
        if item["name"] != item["element"]:
            lines += [f"  &KIND {item['name']}", f"    ELEMENT {item['element']}", "  &END KIND"]
    lines.append("&END SUBSYS")
    return "\n".join(lines) + "\n"


def _write_cp2k_input_payload(
    destination: str | os.PathLike[str] | io.TextIOBase, payload: Mapping[str, Any], **kwargs: Any
) -> None:
    r"""Write a ``cp2k-input`` payload: its ``raw`` text verbatim, else an ``&SUBSYS`` block.

    :param destination: A path or an open text stream.
    :param payload: The geometry payload.
    :param \**kwargs: Not supported (the writer has no options).
    """
    _write_text(destination, payload, kwargs, _render_subsys)
