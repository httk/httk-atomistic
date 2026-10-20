"""Read and write LAMMPS data files (the ``read_data`` format) as neutral geometry payloads.

Data files carry no units and no element identities, so the reader takes lengths as Å and
resolves species from explicit sources only; see :func:`read_lammps_data`.
"""

import decimal
import io
import math
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from httk.atomistic.integrations._geometry import (
    _element_from_label,
    _geometry_from_structure,
    _normalized,
    _number,
    _render,
    _write_text,
)
from httk.atomistic.integrations.vasp.io._text import source_lines
from httk.atomistic.io import _check_as_written, _check_load_precision

__all__ = ["read_lammps_data"]

FORMAT = "lammps-data"
_ACCEPTED_UNITS = ("metal", "real")
_SECTIONS = frozenset(
    {"Atoms", "Velocities", "Masses", "Ellipsoids", "Lines", "Triangles", "Bodies", "Bonds", "Angles", "Dihedrals"}
    | {"Impropers", "Atom Type Labels", "Bond Type Labels", "Angle Type Labels", "Dihedral Type Labels"}
    | {"Improper Type Labels", "Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs"}
    | {"Improper Coeffs", "BondBond Coeffs", "BondAngle Coeffs", "MiddleBondTorsion Coeffs"}
    | {"EndBondTorsion Coeffs", "AngleTorsion Coeffs", "AngleAngleTorsion Coeffs", "BondBond13 Coeffs"}
    | {"AngleAngle Coeffs"}
)
_HEADER_TYPES = re.compile(r"types:\s*(.+)$")
_BOUND_KEYS = {("xlo", "xhi"): 0, ("ylo", "yhi"): 1, ("zlo", "zhi"): 2}
_DECIMAL = decimal.Context(prec=1000)


def read_lammps_data(
    source: Any,
    *,
    species: Mapping[int, str] | Sequence[str] | None = None,
    units: str = "metal",
    precision: float | None = None,
    as_written: bool = False,
) -> dict[str, Any]:
    """Parse a LAMMPS data file (``read_data`` format, ``atomic`` atom style) into a neutral geometry payload.

    The box gives the cell exactly: orthogonal, restricted triclinic (``xy xz yz``) and
    general triclinic (``avec``/``bvec``/``cvec`` plus ``abc origin``) headers are read.
    Atom positions are the wrapped Cartesian coordinates as written, relative to the box
    origin; image flags are ignored. Atoms are ordered by id. ``Velocities``, bonded
    sections and unknown header counts are skipped. The ``Atoms`` section must declare
    ``# atomic`` or, without a style comment, have exactly 5 columns
    (``id type x y z``); 8 columns (with image flags) need the ``# atomic`` comment.

    Data files carry no units: lengths are taken as Å, which holds for ``metal`` and
    ``real`` units. Species come, per atom type, from ``species``, then an
    ``Atom Type Labels`` section, then ``Masses`` line comments (``1 26.98 # Al``), then a
    ``types: Si O`` list in the first line (as the *httk* writers write it); the element is
    read from the name, never inferred from a mass. ``Masses`` values are kept as masses.
    No extension selects this reader: use ``load(path, format="lammps-data")``;
    ``save`` selects the writer by the ``.lmp`` extension.

    :param source: Filename (optionally compressed), open text stream or iterable of lines.
    :param species: Species names by type id (mapping) or in type order (sequence).
    :param units: LAMMPS unit style the file was written for: ``metal`` or ``real``.
    :param precision: Cartesian coordinate precision in Å, or ``None`` for the digit-derived one.
    :param as_written: Already exact; accepted for a uniform load interface.
    :return: The neutral ``lammps-data`` payload, including the original text in ``raw``.
    :raises ValueError: If the units are unsupported, the file is malformed or uses an
        unsupported atom style, or the species of some atom type cannot be determined.
    """
    if units not in _ACCEPTED_UNITS:
        raise ValueError(
            f"units must be one of {_ACCEPTED_UNITS} (lengths in angstrom); {units!r} is not supported, "
            "convert the data file to metal or real units"
        )
    _check_load_precision(precision, "read_lammps_data")
    _check_as_written(as_written, "read_lammps_data")
    with source_lines(source, preserve_path=True, capture_stream=True) as (lines, raw):
        text = list(lines)
    data = _parse(text, species)
    data["precision_override"] = precision
    data["raw"] = raw if raw is not None else "".join(line if line.endswith("\n") else line + "\n" for line in text)
    return data


def _strip(line: str) -> tuple[str, str]:
    """Split a line into its content and its comment."""
    content, _, comment = line.partition("#")
    return content.strip(), comment.strip()


def _decimal(token: str) -> decimal.Decimal:
    _number(token)  # validates the literal
    return decimal.Decimal(_normalized(token))


def _diff(high: str, low: str) -> str:
    return format(_DECIMAL.subtract(_decimal(high), _decimal(low)), "f")


def _header(lines: list[str]) -> tuple[int, dict[str, Any]]:
    """Parse header lines after the comment; return the index of the first section line and the header values."""
    values: dict[str, Any] = {}
    for index in range(1, len(lines)):
        content, _ = _strip(lines[index])
        if not content:
            continue
        if content in _SECTIONS:
            return index, values
        words = content.split()
        tail = tuple(words[-2:])
        if len(words) == 4 and tail in (("xlo", "xhi"), ("ylo", "yhi"), ("zlo", "zhi")):
            values[tail[0][0]] = words[:2]
        elif len(words) == 6 and words[3:] == ["xy", "xz", "yz"]:
            values["tilt"] = words[:3]
        elif len(words) == 4 and words[3] in ("avec", "bvec", "cvec"):
            values[words[3]] = words[:3]
        elif len(words) == 5 and words[3:] == ["abc", "origin"]:
            values["origin"] = words[:3]
        elif len(words) == 2 and words[1] == "atoms":
            values["atoms"] = _count(words[0])
        elif len(words) == 3 and words[1:] == ["atom", "types"]:
            values["types"] = _count(words[0])
        # other counts and unknown header lines are ignored
    return len(lines), values


def _count(token: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise ValueError(f"invalid LAMMPS data count {token!r}") from None


def _cell(values: dict[str, Any]) -> tuple[list[list[Any]], list[str]]:
    """The cell rows and origin tokens of a header."""
    zero: Any = 0
    if "avec" in values or "bvec" in values or "cvec" in values:
        if not ("avec" in values and "bvec" in values and "cvec" in values and "origin" in values):
            raise ValueError("a general triclinic LAMMPS data header needs avec, bvec, cvec and abc origin")
        if {"x", "y", "z"} & values.keys() or "tilt" in values:
            raise ValueError("LAMMPS data header mixes general (avec/bvec/cvec) and restricted box lines")
        return [list(values[key]) for key in ("avec", "bvec", "cvec")], list(values["origin"])
    if not {"x", "y", "z"} <= values.keys():
        raise ValueError("LAMMPS data header needs xlo xhi, ylo yhi and zlo zhi lines")
    lows = [values[axis][0] for axis in "xyz"]
    lengths = [_diff(values[axis][1], values[axis][0]) for axis in "xyz"]
    xy, xz, yz = values.get("tilt", [zero, zero, zero])
    return [[lengths[0], zero, zero], [xy, lengths[1], zero], [xz, yz, lengths[2]]], lows


def _parse(lines: list[str], species: Mapping[int, str] | Sequence[str] | None) -> dict[str, Any]:
    if not lines or not lines[0].strip():
        raise ValueError("not a LAMMPS data file: the first line must be a comment line")
    start, values = _header(lines)
    cell, origin = _cell(values)
    count, ntypes = values.get("atoms"), values.get("types")
    if count is None or ntypes is None or count < 1 or ntypes < 1:
        raise ValueError("LAMMPS data header needs positive 'N atoms' and 'N atom types' lines")

    sections: dict[str, tuple[str, list[tuple[int, list[str], str]]]] = {}
    current: list[tuple[int, list[str], str]] | None = None
    for number in range(start, len(lines)):
        content, comment = _strip(lines[number])
        if not content:
            continue
        if content in _SECTIONS:
            if content in sections:
                raise ValueError(f"line {number + 1}: section {content!r} appears twice")
            current = []
            sections[content] = (comment, current)
        elif current is None:
            raise ValueError(f"line {number + 1}: unexpected text before the first section: {content!r}")
        else:
            current.append((number + 1, content.split(), comment))
    if "Atoms" not in sections:
        raise ValueError("LAMMPS data file has no Atoms section")

    labels: dict[str, int] = {}
    for number, words, _ in sections.get("Atom Type Labels", ("", []))[1]:
        if len(words) != 2 or not words[0].isdecimal():
            raise ValueError(f"line {number}: Atom Type Labels lines are 'type label'")
        labels[words[1]] = int(words[0])

    def type_id(token: str, number: int) -> int:
        kind = int(token) if token.isdecimal() else labels.get(token)
        if kind is None or not 1 <= kind <= ntypes:
            raise ValueError(f"line {number}: unknown atom type {token!r} (the file declares {ntypes} atom types)")
        return kind

    style, rows = sections["Atoms"]
    if style and style.split()[0].lower() != "atomic":
        raise ValueError(f"unsupported LAMMPS atom style {style.split()[0]!r}: only atomic data files are read")
    atoms: dict[int, tuple[int, list[str]]] = {}
    for number, words, _ in rows:
        if len(words) != (8 if style else 5) and not (style and len(words) == 5):
            raise ValueError(
                f"line {number}: an atomic Atoms line has 5 columns (id type x y z), or 8 with image flags "
                f"under an explicit '# atomic' comment, got {len(words)}"
            )
        if not words[0].isdecimal() or int(words[0]) in atoms:
            raise ValueError(f"line {number}: atom id {words[0]!r} is not a unique positive integer")
        atoms[int(words[0])] = (type_id(words[1], number), words[2:5])
    if len(atoms) != count:
        raise ValueError(f"LAMMPS data header declares {count} atoms but the Atoms section has {len(atoms)}")
    order = sorted(atoms)

    masses: dict[int, str] = {}
    from_masses: dict[int, str] = {}
    for number, words, comment in sections.get("Masses", ("", []))[1]:
        if len(words) != 2:
            raise ValueError(f"line {number}: Masses lines are 'type mass'")
        kind = type_id(words[0], number)
        _number(words[1])
        masses[kind] = _normalized(words[1])
        if comment and _element_from_label(comment.split()[0]):
            from_masses[kind] = comment.split()[0]
    from_header: dict[int, str] = {}
    hint = _HEADER_TYPES.search(lines[0])
    if hint and len(hint.group(1).split()) == ntypes:
        from_header = {
            index: name for index, name in enumerate(hint.group(1).split(), start=1) if _element_from_label(name)
        }
    given: dict[int, str]
    if species is None:
        given = {}
    elif isinstance(species, Mapping):
        given = {int(key): str(name) for key, name in species.items()}
    else:
        if isinstance(species, str) or len(species) != ntypes:
            raise ValueError(f"species must be a mapping or a sequence of {ntypes} names in type order")
        given = {index: str(name) for index, name in enumerate(species, start=1)}
    by_label = {kind: name for name, kind in labels.items()}

    used = sorted({kind for kind, _ in atoms.values()})
    names: dict[int, str] = {}
    for kind in used:
        explicit = given.get(kind) or by_label.get(kind)
        name = explicit or from_masses.get(kind) or from_header.get(kind)
        if name is None:
            raise ValueError(
                f"cannot determine the species of atom type {kind}: pass species=... "
                "(a data file names no elements unless it has Atom Type Labels, Masses comments or a 'types:' comment)"
            )
        if explicit and _element_from_label(name) is None:
            raise ValueError(f"species name {name!r} of atom type {kind} does not start with an element symbol")
        names[kind] = name

    return {
        "format": FORMAT,
        "comment": lines[0].rstrip("\r\n"),
        "cell": cell,
        "cell_unit": "angstrom",
        "positions": [[_diff(token, low) for token, low in zip(atoms[i][1], origin, strict=True)] for i in order],
        "positions_kind": "cartesian",
        "positions_unit": "angstrom",
        "species": [
            {"name": names[kind], "element": _element_from_label(names[kind]), "mass": masses.get(kind)}
            for kind in used
        ],
        "species_at_sites": [names[atoms[i][0]] for i in order],
    }


def _lammps_data_payload_from_structure(obj: Any) -> dict[str, Any]:
    """Serialize a structure to a ``lammps-data`` geometry payload (the registered format serializer)."""
    return _geometry_from_structure(obj, FORMAT)


def _unit(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector]


def _dot(u: Sequence[float], v: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(u, v, strict=True))


def _cross(u: Sequence[float], v: Sequence[float]) -> list[float]:
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def _render_data(
    payload: Mapping[str, Any], *, masses: Mapping[str, float] | None = None, atom_style: str = "atomic"
) -> str:
    # ponytail: only `atomic`; add charge/molecular columns when a workflow needs them.
    if atom_style != "atomic":
        raise ValueError(f"atom_style must be 'atomic', not {atom_style!r}")
    cell = [[_number(token) for token in row] for row in payload["cell"]]
    fractional = [[_number(token) for token in row] for row in payload["positions"]]
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = cell
    if ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx) <= 0:
        raise ValueError("the cell is left-handed; LAMMPS needs a right-handed cell (reorder or negate a vector)")
    framed = ay == az == bz == 0 and ax > 0 and by > 0 and cz > 0
    cartesian = [[sum(f[i] * cell[i][j] for i in range(3)) for j in range(3)] for f in fractional]
    if framed:
        box = [_render(ax), _render(by), _render(cz)]
        tilts = [_render(bx), _render(cx), _render(cy)]
        sites = [[_render(value) for value in row] for row in cartesian]
        has_tilt = any(value != 0 for value in (bx, cx, cy))
    else:
        # The rotation into the LAMMPS frame (x along a, y in the ab plane) is done in floats.
        a, b, c = ([float(x) for x in row] for row in cell)
        x = _unit(a)
        y = _unit([bi - _dot(b, x) * xi for bi, xi in zip(b, x, strict=True)])
        z = _cross(x, y)
        numbers = [_dot(a, x), _dot(b, y), _dot(c, z), _dot(b, x), _dot(c, x), _dot(c, y)]
        box, tilts = ([format(v, ".16g") for v in numbers[:3]], [format(v, ".16g") for v in numbers[3:]])
        sites = [[format(_dot([float(v) for v in r], axis), ".16g") for axis in (x, y, z)] for r in cartesian]
        has_tilt = True

    at_sites = list(payload["species_at_sites"])
    names = tuple(dict.fromkeys(at_sites))
    if any(not name or re.search(r"[\s#]", name) for name in names):
        raise ValueError(f"species names must be non-empty without whitespace or '#' in a LAMMPS data file: {names}")
    types = {name: index for index, name in enumerate(names, start=1)}
    own = {item["name"]: item["mass"] for item in payload["species"] if item["mass"] is not None}
    table: dict[str, str] | None
    if masses is not None:
        missing = [name for name in names if name not in masses]
        if missing:
            raise ValueError(f"no mass given for species {', '.join(missing)}")
        table = {name: repr(float(masses[name])) for name in names}
    else:
        table = own if all(name in own for name in names) else None

    lines = [
        f"LAMMPS data file written by httk-atomistic; types: {' '.join(names)}",
        "",
        f"{len(at_sites)} atoms",
        f"{len(names)} atom types",
        "",
        f"0.0 {box[0]} xlo xhi",
        f"0.0 {box[1]} ylo yhi",
        f"0.0 {box[2]} zlo zhi",
    ]
    if has_tilt:
        lines.append(f"{tilts[0]} {tilts[1]} {tilts[2]} xy xz yz")
    if table is not None:
        lines += ["", "Masses", ""] + [f"{types[name]} {table[name]} # {name}" for name in names]
    lines += ["", "Atom Type Labels", ""] + [f"{types[name]} {name}" for name in names]
    lines += ["", f"Atoms # {atom_style}", ""]
    lines += [f"{i} {types[name]} {r[0]} {r[1]} {r[2]}" for i, (name, r) in enumerate(zip(at_sites, sites), start=1)]
    return "\n".join(lines) + "\n"


def _write_lammps_data_payload(
    destination: str | os.PathLike[str] | io.TextIOBase, payload: Mapping[str, Any], **kwargs: Any
) -> None:
    r"""Write a ``lammps-data`` payload as a LAMMPS data file (the registered writer).

    A payload carrying ``raw`` text is written verbatim and takes no options. Otherwise the
    cell is rotated into the LAMMPS restricted-triclinic frame (``a`` along x, ``b`` in the
    xy plane). A cell already in that frame is written exactly; any other cell is rotated in
    floating point and written to 16 significant digits, so ``save`` of it is lossy at that
    level. Atom types are numbered by first appearance, named in ``Masses`` comments and an
    ``Atom Type Labels`` section so the file reads back without ``species=``.

    :param destination: A filesystem path or an open text stream.
    :param payload: The ``lammps-data`` geometry payload.
    :param \**kwargs: ``masses`` (species name to mass; default the structure's own, with
        ``Masses`` omitted if any species has none) and ``atom_style`` (only ``atomic``).
    :raises ValueError: If the cell is left-handed, a mass is missing, or an option is unsupported.
    """
    _write_text(destination, payload, kwargs, _render_data)
