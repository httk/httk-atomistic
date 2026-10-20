"""ABINIT input files: the shared variable parser, the structure reader and the geometry writer.

ABINIT converts lengths with its own Bohr radius, 0.52917720859 Å (CODATA 2006). ABINIT
10.0.3 output confirms it: ``xcart 2.565`` Bohr is echoed as ``xangst 1.3573395400``, where
the CODATA 2018 value ASE uses would give ``1.3573395451``. Conversion through it is exact.
"""

import fractions
import io
import os
import re
from collections.abc import Mapping
from typing import Any

from httk.core import SurdScalar, exactmath

from httk.atomistic.elements import atomic_number, symbol_of
from httk.atomistic.integrations._geometry import (
    _NUMBER,
    _geometry_from_structure,
    _normalized,
    _number,
    _structure_from_geometry,
    _write_text,
)
from httk.atomistic.integrations.vasp.io._text import source_lines
from httk.atomistic.integrations.vasp.io.poscar_reader import _coordinate_precision
from httk.atomistic.io import _check_as_written, _check_load_precision

FORMAT = "abinit-input"
BOHR_ANGSTROM = "0.52917720859"

type Exact = fractions.Fraction | SurdScalar
type Variables = dict[str, list[str]]
type Units = dict[str, str]

# A quoted string, a ``#``/``!`` comment to the end of the line, a bare token, or a lone quote.
_TOKEN = re.compile(r'"[^"\n]*"|[#!].*|[^\s#!"]+|"')
_NAME = re.compile(r"[a-z_][\w:+?]*")
# A variable name split into its base and a dataset (digits, ``:``, ``+``, ``?``) or image suffix.
_SUFFIXED = re.compile(r"([a-z_]+?)([0-9:+?]*|_[0-9]+img|_lastimg)")
_SQRT = re.compile(r"(-?)sqrt\((.*)\)", re.IGNORECASE)
_INTEGER = re.compile(r"[+-]?\d+")
# ABINIT's unit words; only the length units are meaningful for the structure variables.
_LENGTH_UNITS = {"bohr": "bohr", "angstr": "angstrom", "angstrom": "angstrom"}
_UNITS = frozenset(
    {*_LENGTH_UNITS, "au", "nm", "ha", "hartree", "ry", "rydberg", "ev", "mev", "k", "kelvin", "t", "tesla", "s", "sec"}
)
_STRUCTURE = frozenset(
    {"acell", "angdeg", "natom", "ntypat", "rprim", "scalecart", "typat", "xangst", "xcart", "xred", "znucl"}
)
_REFUSED = {
    "spgroup": "symmetry-generated atoms (spgroup)",
    "spgroupma": "symmetry-generated atoms (spgroupma)",
    "natrd": "symmetry-generated atoms (natrd)",
    "nobj": "geometry built from objects (nobj)",
    "structure": "the structure variable (geometry read from another file)",
    "xyzfile": "geometry read from an xyz file (xyzfile)",
    "getcell": "a cell taken from another dataset (getcell)",
    "getxcart": "positions taken from another dataset (getxcart)",
    "getxred": "positions taken from another dataset (getxred)",
}
_TOLERANCE = fractions.Fraction(1, 10**12)


def _source_text(source: Any) -> tuple[str, str | None]:
    """Decode a path, stream or iterable of lines into text, plus the raw text when available."""
    try:
        with source_lines(source, preserve_path=True, capture_stream=True) as (lines, raw):
            if raw is not None:
                return raw, raw
            return "".join(line if line.endswith("\n") else line + "\n" for line in lines), None
    except UnicodeDecodeError as error:
        raise ValueError(f"ABINIT file is not UTF-8 text: {error}") from None


def _expand(token: str) -> list[str]:
    """Expand ABINIT's ``n*value`` repetition."""
    count, star, value = token.partition("*")
    if not star:
        return [token]
    if not _INTEGER.fullmatch(count) or int(count) < 1 or not value:
        raise ValueError(f"unsupported ABINIT repetition {token!r}; expected n*value with a positive count n")
    return [value] * int(count)


def _variables(text: str) -> tuple[Variables, Units]:
    """Parse ABINIT ``name value... [unit]`` text into value tokens and units per lower-case name.

    Repetitions are expanded; values stay unconverted tokens. A unit word attaches to the
    variable it follows.
    """
    values: Variables = {}
    units: Units = {}
    name: str | None = None
    for match in _TOKEN.finditer(text):
        token = match.group()
        if token[0] in "#!":
            continue
        if token == '"':
            raise ValueError("ABINIT input has an unterminated string")
        lower = token.lower()
        if lower in _UNITS:
            if name is None or not values[name] or name in units:
                raise ValueError(f"ABINIT unit {token!r} does not follow a variable's values")
            units[name] = lower
        elif _NAME.fullmatch(lower):
            if name is not None and not values[name]:
                raise ValueError(f"ABINIT variable {name!r} has no value (followed by {token!r})")
            if lower in values:
                raise ValueError(f"ABINIT variable {lower!r} appears more than once")
            name = lower
            values[name] = []
        elif name is None:
            raise ValueError(f"ABINIT value {token!r} precedes any variable name")
        elif name in units:
            raise ValueError(f"ABINIT variable {name!r} has a value {token!r} after its unit")
        else:
            values[name].extend(_expand(token))
    if name is not None and not values[name]:
        raise ValueError(f"ABINIT variable {name!r} has no value")
    return values, units


def _value(token: str, name: str) -> Exact:
    """One exact number: a decimal, ``a/b``, ``sqrt(x)`` or ``-sqrt(x)``."""
    try:
        root = _SQRT.fullmatch(token)
        if root is not None:
            value = exactmath.sqrt(_number(root.group(2)), exact=True)
            return -value if root.group(1) else value
        numerator, slash, denominator = token.partition("/")
        if slash:
            return _number(numerator) / _number(denominator)
        return _number(token)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"ABINIT {name} value {token!r} is not a supported number: {error}") from None


def _entry(token: str, name: str) -> str | Exact:
    """A payload entry: the token itself when it is a plain decimal (keeping its precision claim)."""
    return token if _NUMBER.fullmatch(token) else _value(token, name)


def _precision(token: str, *, coefficient: bool = False) -> fractions.Fraction:
    """A token's digit precision claim; ``0`` for exact expressions and digit-less integers.

    Dimensionless ``coefficient`` tokens (``rprim``, ``scalecart``) follow POSCAR's direct
    coordinates: ``0.0``, ``0.5`` and ``1.0`` spell exact fractions and claim nothing.
    """
    if not _NUMBER.fullmatch(token):
        return fractions.Fraction(0)
    return _coordinate_precision([[_normalized(token)]], cartesian=not coefficient) or fractions.Fraction(0)


def _size(value: Any) -> fractions.Fraction:
    """The magnitude of an exact number."""
    if isinstance(value, fractions.Fraction):
        return abs(value)
    # ponytail: surd magnitudes via float, ~1e-16 relative; only scales a precision bound.
    return fractions.Fraction(abs(float(value)))


def _get(values: Variables, name: str, count: int, default: list[str] | None = None) -> list[str]:
    """The value tokens of a variable, checked against the count ABINIT expects."""
    tokens = values.get(name, default)
    if tokens is None:
        raise ValueError(f"ABINIT structure needs the variable {name}")
    if len(tokens) != count:
        raise ValueError(f"ABINIT {name} needs {count} values, got {len(tokens)}")
    return tokens


def _integer(token: str, name: str) -> int:
    """One integer value; decimals with an integral value (as in output echoes) are accepted."""
    value = _number(token) if _NUMBER.fullmatch(token) else None
    if value is None or value.denominator != 1:
        raise ValueError(f"ABINIT {name} value {token!r} is not an integer")
    return int(value)


def _check_supported(values: Variables, units: Units) -> None:
    """Refuse the input constructs this reader does not model."""
    for name in values:
        match = _SUFFIXED.fullmatch(name)
        if match is None:
            continue
        base, suffix = match.groups()
        if base in _REFUSED:
            raise ValueError(f"ABINIT {_REFUSED[base]} is not supported; give the explicit geometry")
        if suffix and base in _STRUCTURE:
            raise ValueError(
                f"ABINIT dataset- or image-dependent structure variable {name!r} is not supported; "
                "a structure is read only from dataset-independent variables"
            )
    for name, unit in units.items():
        if name in _STRUCTURE and (name not in ("acell", "xcart") or unit not in _LENGTH_UNITS):
            raise ValueError(f"ABINIT unit {unit!r} on {name} is not supported (acell and xcart take Bohr or Angstrom)")


# Exact cosines of the angles (degrees) whose ABINIT angdeg cells this reader builds.
_HALF = fractions.Fraction(1, 2)
_COS: dict[int, Exact] = {
    0: fractions.Fraction(1),
    30: exactmath.sqrt(3, exact=True) * _HALF,
    45: exactmath.sqrt(2, exact=True) * _HALF,
    60: _HALF,
    90: fractions.Fraction(0),
    120: -_HALF,
    135: -exactmath.sqrt(2, exact=True) * _HALF,
    150: -exactmath.sqrt(3, exact=True) * _HALF,
}


class _Inexact(Exception):
    """ABINIT's angdeg construction has no exact form for these angles."""


def _cos(angle: fractions.Fraction) -> Any:
    if angle not in _COS:
        raise _Inexact
    return _COS[int(angle)]


def _sqrt(value: Any) -> Any:
    try:
        return exactmath.sqrt(value, exact=True)
    except ValueError:
        raise _Inexact from None


def _angdeg_rprim(tokens: list[str]) -> list[list[Any]] | None:
    """The rprim rows ABINIT builds from angdeg (ingeo.F90, as documented for angdeg), or ``None``.

    ``None`` means the construction is not exact: an angle outside 30, 45, 60, 90, 120, 135
    and 150 degrees, or a square root of an irrational value.
    """
    values = [_value(token, "angdeg") for token in tokens]
    angles = [angle for angle in values if isinstance(angle, fractions.Fraction) and 0 < angle < 180]
    if len(angles) != 3:
        raise ValueError(f"ABINIT angdeg angles must be rational degrees between 0 and 180, got {tokens}")
    try:
        return _exact_angdeg_rprim(*angles)
    except _Inexact:
        return None


def _exact_angdeg_rprim(a1: fractions.Fraction, a2: fractions.Fraction, a3: fractions.Fraction) -> list[list[Any]]:
    """ingeo.F90's angdeg rows, raising :class:`_Inexact` when they have no exact form."""
    zero = fractions.Fraction(0)
    if (
        abs(a1 - a2) < _TOLERANCE
        and abs(a2 - a3) < _TOLERANCE
        and abs(a1 - 90) + abs(a2 - 90) + abs(a3 - 90) > _TOLERANCE
    ):
        # Equal angles: the trigonal axis along z.
        square = fractions.Fraction(2, 3) * (1 - _cos(a1))
        aa, cc = _sqrt(square), _sqrt(1 - square)
        y = exactmath.sqrt(3, exact=True) * _HALF * aa
        return [[aa, zero, cc], [-_HALF * aa, y, cc], [-_HALF * aa, -y, cc]]
    cos1, cos2, cos3 = _cos(a1), _cos(a2), _cos(a3)
    sin3 = _cos(abs(90 - a3))
    y3 = (cos1 - cos3 * cos2) / sin3
    z3 = _sqrt(1 - cos2 * cos2 - y3 * y3)
    return [[fractions.Fraction(1), zero, zero], [cos3, sin3, zero], [cos2, y3, z3]]


def _geometry(values: Variables, units: Units) -> dict[str, Any]:
    """The neutral geometry payload (without ``format``) of parsed ABINIT structure variables."""
    natom = _integer(_get(values, "natom", 1, ["1"])[0], "natom")
    ntypat = _integer(_get(values, "ntypat", 1, ["1"])[0], "ntypat")
    if natom < 1 or ntypat < 1:
        raise ValueError(f"ABINIT natom and ntypat must be positive, got {natom} and {ntypat}")
    typat = [_integer(token, "typat") for token in _get(values, "typat", natom, ["1"] if natom == 1 else None)]
    znucl = [_integer(token, "znucl") for token in _get(values, "znucl", ntypat)]
    if any(not 1 <= kind <= ntypat for kind in typat):
        raise ValueError(f"ABINIT typat values must lie in 1..ntypat={ntypat}, got {typat}")

    acell_tokens = _get(values, "acell", 3, ["1", "1", "1"])
    acell = [_value(token, "acell") for token in acell_tokens]
    scalecart_tokens = _get(values, "scalecart", 3, ["1", "1", "1"])
    scalecart = [_value(token, "scalecart") for token in scalecart_tokens]
    given = [name for name in ("xred", "xcart", "xangst") if name in values]
    if len(given) > 1:
        raise ValueError(f"ABINIT positions are given more than once ({', '.join(given)})")
    payload: dict[str, Any] = {"cell_unit": _LENGTH_UNITS[units.get("acell", "bohr")], "bohr_angstrom": BOHR_ANGSTROM}

    if "rprim" in values and "angdeg" in values:
        raise ValueError("ABINIT rprim and angdeg are mutually exclusive")
    rprim: list[list[Any]] | None
    angdeg: list[str] = []
    rprim_tokens = ["0"] * 9  # angdeg rows are exact
    if "angdeg" in values:
        angdeg = _get(values, "angdeg", 3)
        rprim = _angdeg_rprim(angdeg)
    else:
        rprim_tokens = _get(values, "rprim", 9, ["1", "0", "0", "0", "1", "0", "0", "0", "1"])
        flat = [_value(token, "rprim") for token in rprim_tokens]
        rprim = [flat[3 * row : 3 * row + 3] for row in range(3)]
    if rprim is not None:
        # rprimd(i, j) = scalecart(i) * rprim(i, j) * acell(j): vector j's Cartesian component i.
        payload["cell"] = [
            [acell[row] * rprim[row][column] * scalecart[column] for column in range(3)] for row in range(3)
        ]
        # The products are exact numbers, so their digit claim is propagated explicitly:
        # |d(a r s)| <= p(a)|r||s| + |a|p(r)|s| + |a||r|p(s), maximized over the entries.
        bound = max(
            _precision(acell_tokens[row]) * _size(rprim[row][column]) * _size(scalecart[column])
            + _size(acell[row])
            * _precision(rprim_tokens[3 * row + column], coefficient=True)
            * _size(scalecart[column])
            + _size(acell[row]) * _size(rprim[row][column]) * _precision(scalecart_tokens[column], coefficient=True)
            for row in range(3)
            for column in range(3)
        )
        payload["cell_precision"] = bound or None
    elif given and given[0] != "xred":
        raise ValueError(
            f"ABINIT {given[0]} with angdeg {' '.join(angdeg)} needs an explicit rprim: these angles have no "
            "exact ABINIT frame, and Cartesian coordinates depend on that frame"
        )
    elif any(value != 1 for value in scalecart):
        raise ValueError(
            f"ABINIT scalecart with angdeg {' '.join(angdeg)} needs an explicit rprim: these angles have no "
            "exact ABINIT frame, and scalecart scales its Cartesian components"
        )
    else:
        # Same lengths and angles as ABINIT's cell, in the CellParams frame; fractional sites do not see the frame.
        payload["cell_parameters"] = [_entry(token, "acell") for token in acell_tokens] + [
            _entry(token, "angdeg") for token in angdeg
        ]
    if given:
        coordinates = given[0]
        flat_positions = [_entry(token, coordinates) for token in _get(values, coordinates, 3 * natom)]
        payload["positions"] = [flat_positions[3 * site : 3 * site + 3] for site in range(natom)]
        if coordinates == "xred":
            payload["positions_kind"] = "fractional"
        else:
            payload["positions_kind"] = "cartesian"
            payload["positions_unit"] = (
                "angstrom" if coordinates == "xangst" else _LENGTH_UNITS[units.get("xcart", "bohr")]
            )
    else:
        payload["positions"] = [[0, 0, 0] for _ in range(natom)]
        payload["positions_kind"] = "fractional"

    used = sorted(set(typat))
    elements = {kind: symbol_of(znucl[kind - 1]) for kind in used}
    totals = {element: list(elements.values()).count(element) for element in elements.values()}
    seen: dict[str, int] = {}
    names: dict[int, str] = {}
    for kind in used:
        element = elements[kind]
        seen[element] = seen.get(element, 0) + 1
        names[kind] = element if totals[element] == 1 else f"{element}{seen[element]}"
    payload["species"] = [{"name": names[kind], "element": elements[kind], "mass": None} for kind in used]
    payload["species_at_sites"] = [names[kind] for kind in typat]
    return payload


def read_abinit_input(source: Any, *, precision: float | None = None, as_written: bool = False) -> dict[str, Any]:
    """Read the structure of an ABINIT input file into a neutral geometry payload.

    The structure variables ``acell`` (with a ``Bohr``/``Angstrom`` unit), ``rprim`` or
    ``angdeg``, ``scalecart``, ``natom``, ``ntypat``, ``typat``, ``znucl`` and one of
    ``xred``, ``xcart`` or ``xangst`` are read exactly, with ABINIT's defaults; every other
    variable is ignored. An ``angdeg`` cell whose ABINIT construction is not exact is given
    as ``cell_parameters`` (lengths ``acell``, angles ``angdeg``) when the positions are
    fractional and ``scalecart`` is unity. Species are named by element (``Fe1``, ``Fe2`` when two types share
    an element).

    :param source: A file path (compressed files are decompressed), a text stream, or an
        iterable of lines.
    :param precision: Optional Cartesian precision in Å for the structure built from it.
    :param as_written: Already exact; accepted for a uniform load interface.
    :return: The ``abinit-input`` geometry payload, with the original text in ``raw``.
    :raises ValueError: If the text is not a supported ABINIT input: malformed values,
        missing ``znucl``, inconsistent counts, multi-dataset or image structure variables,
        symmetry-generated atoms, the ``structure`` variable, or ``xcart``/``xangst`` or a
        non-unit ``scalecart`` with ``angdeg`` angles that have no exact ABINIT frame.
    """
    _check_load_precision(precision, "read_abinit_input")
    _check_as_written(as_written, "read_abinit_input")
    text, raw = _source_text(source)
    values, units = _variables(text)
    _check_supported(values, units)
    payload = {"format": FORMAT, **_geometry(values, units), "raw": raw}
    if precision is not None:
        payload["precision_override"] = precision
    return payload


def _abinit_input_payload_from_structure(obj: Any) -> dict[str, Any]:
    """Serialize a structure to an ``abinit-input`` geometry payload.

    :param obj: Anything :class:`~httk.atomistic.UnitcellStructureView` accepts.
    :return: The geometry payload.
    """
    return _geometry_from_structure(obj, FORMAT)


def _render_abinit_input(payload: Mapping[str, Any]) -> str:
    """Render a geometry payload as the structure variables of an ABINIT input."""
    geometry = _geometry_from_structure(_structure_from_geometry({**payload, "format": FORMAT}), FORMAT)
    if not all(geometry["periodicity"]):
        raise ValueError("ABINIT cells are periodic along all three cell vectors; this structure is not")
    names = [item["name"] for item in geometry["species"]]
    typat = [str(names.index(name) + 1) for name in geometry["species_at_sites"]]
    lines = [
        "acell 1 1 1 Angstrom",
        "rprim",
        *(" " + " ".join(row) for row in geometry["cell"]),
        f"natom {len(typat)}",
        f"ntypat {len(names)}",
        "typat",
        *(" " + " ".join(typat[start : start + 20]) for start in range(0, len(typat), 20)),
        "znucl " + " ".join(str(atomic_number(item["element"])) for item in geometry["species"]),
        "xred",
        *(" " + " ".join(row) for row in geometry["positions"]),
    ]
    return "\n".join(lines) + "\n"


def _write_abinit_input_payload(
    destination: str | os.PathLike[str] | io.TextIOBase, payload: Mapping[str, Any], **kwargs: Any
) -> None:
    """Write an ``abinit-input`` payload: its ``raw`` text verbatim, or else its structure variables.

    :param destination: A filesystem path or an open text stream.
    :param payload: The geometry payload.
    :param \\*\\*kwargs: Writer options; none are accepted.
    :raises ValueError: If options are given with ``raw`` text, or the structure is not periodic.
    :raises TypeError: If options are given for a payload without ``raw`` text.
    """
    _write_text(destination, payload, kwargs, _render_abinit_input)


__all__ = ["read_abinit_input"]
