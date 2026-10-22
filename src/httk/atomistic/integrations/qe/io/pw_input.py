"""Read and write the geometry of Quantum ESPRESSO ``pw.x`` input files.

The reader interprets only the geometry of an input: the ``&system`` keys ``ibrav``,
``celldm(1..6)``, ``A``, ``B``, ``C``, ``cosAB``, ``cosAC``, ``cosBC``, ``nat`` and ``ntyp``
and the cards ``ATOMIC_SPECIES``, ``CELL_PARAMETERS`` and ``ATOMIC_POSITIONS``. Every other
namelist and card is parsed (namelists) or delimited (cards) and otherwise skipped. Lengths in
Bohr are converted to Å with *Quantum ESPRESSO*'s own constant, ``BOHR_RADIUS_ANGS =
0.52917720859`` (CODATA 2006, ``Modules/constants.f90`` of QE 7.x), recorded as
:data:`BOHR_ANGSTROM`.
"""

import fractions
import io
import math
import numbers
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from httk.core import combined_precision, exactmath

from httk.atomistic.integrations._geometry import (
    _element_from_label,
    _geometry_from_structure,
    _number,
    _structure_from_geometry,
    _write_text,
)
from httk.atomistic.integrations.vasp.io._text import source_lines
from httk.atomistic.io import _check_load_precision

FORMAT = "qe-pw-input"
BOHR_ANGSTROM = "0.52917720859"
"""The Bohr radius in Å used by *Quantum ESPRESSO* 7.x (CODATA 2006)."""

_LABEL_RULE = "an element symbol optionally followed by digits or by _x/-x, at most 3 characters"

# One namelist token: blanks/comments, a quoted string (doubled quotes escape), an indexed
# name such as ``celldm(1)`` or ``starting_ns_eigenvalue(1,2,1)``, punctuation, or a word.
_TOKEN = re.compile(
    r"""(?P<skip>\s+|!.*)
    |(?P<string>'(?:[^']|'')*'|"(?:[^"]|"")*")
    |(?P<name>[A-Za-z_]\w*\s*\([^)]*\))
    |(?P<punct>[=,/])
    |(?P<word>[^\s=,/'"!]+)""",
    re.VERBOSE,
)
_CARDS = frozenset(
    {
        "ATOMIC_SPECIES",
        "ATOMIC_POSITIONS",
        "K_POINTS",
        "ADDITIONAL_K_POINTS",
        "CELL_PARAMETERS",
        "CONSTRAINTS",
        "OCCUPATIONS",
        "ATOMIC_VELOCITIES",
        "ATOMIC_FORCES",
        "SOLVENTS",
        "HUBBARD",
    }
)
_CARD_HEAD = re.compile(r"([A-Za-z_]+)(.*)")
_INTEGER = re.compile(r"[+-]?\d+")
_CELLDM = re.compile(r"celldm(?:\((\d+)\))?")
_ABC_KEYS = ("a", "b", "c", "cosab", "cosac", "cosbc")
_HALF = fractions.Fraction(1, 2)

type _Rows = list[list[Any]]


def _source_text(source: Any, what: str) -> str:
    """The decoded text of a path, open stream or iterable of lines."""
    try:
        with source_lines(source, preserve_path=True, capture_stream=True) as (lines, raw):
            if raw is not None:
                return raw
            return "".join(line if line.endswith("\n") else line + "\n" for line in lines)
    except UnicodeDecodeError as error:
        raise ValueError(f"{what} is not UTF-8 text: {error}") from None


def _label_element(label: str, where: str) -> str:
    """The element a ``pw.x`` species label names, by QE's label rule."""
    element = _element_from_label(label) if len(label) <= 3 else None
    if element is None:
        raise ValueError(f"{where}: species label {label!r} does not name an element (pw.x labels are {_LABEL_RULE})")
    return element


def _lineno(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _namelists(text: str) -> tuple[dict[str, dict[str, list[str]]], int]:
    """Parse the leading Fortran namelists; return them and the offset where cards start."""
    namelists: dict[str, dict[str, list[str]]] = {}
    current: str | None = None
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None:
            raise ValueError(f"pw.x input line {_lineno(text, pos)}: unterminated string")
        kind, token = match.lastgroup or "", match.group()
        if kind == "skip":
            pos = match.end()
            continue
        if current is None:
            if not token.startswith("&"):
                break
            current = token[1:].lower()
            if not current or current in namelists:
                raise ValueError(f"pw.x input line {_lineno(text, pos)}: empty or repeated namelist {token!r}")
            tokens = []
        elif token == "/":
            namelists[current] = _assignments(current, tokens)
            current = None
        else:
            tokens.append((kind, token))
        pos = match.end()
    if current is not None:
        raise ValueError(f"pw.x input namelist &{current} is not terminated by '/'")
    return namelists, pos


def _assignments(namelist: str, tokens: list[tuple[str, str]]) -> dict[str, list[str]]:
    """The ``name = value[, value...]`` assignments of one namelist, names lower-cased."""
    values: dict[str, list[str]] = {}
    key: str | None = None
    index = 0
    while index < len(tokens):
        kind, token = tokens[index]
        if token == ",":
            index += 1
            continue
        if kind in ("name", "word") and index + 1 < len(tokens) and tokens[index + 1][1] == "=":
            key = re.sub(r"\s+", "", token).lower()
            if key in values:
                raise ValueError(f"pw.x input &{namelist}: {key} is assigned twice")
            values[key] = []
            index += 2
            continue
        if key is None or token == "=":
            raise ValueError(f"pw.x input &{namelist}: malformed assignment near {token!r}")
        values[key].append(token)
        index += 1
    empty = sorted(name for name, assigned in values.items() if not assigned)
    if empty:
        raise ValueError(f"pw.x input &{namelist}: no value assigned to {empty}")
    return values


def _cards(text: str, offset: int) -> dict[str, tuple[str, list[tuple[int, list[str]]]]]:
    """Split the card section into ``{CARD: (option, [(line number, fields), ...])}``."""
    cards: dict[str, tuple[str, list[tuple[int, list[str]]]]] = {}
    current: list[tuple[int, list[str]]] | None = None
    for lineno, line in enumerate(text[offset:].splitlines(), start=_lineno(text, offset)):
        content = re.split(r"[!#]", line, maxsplit=1)[0].strip()
        if not content:
            continue
        head = _CARD_HEAD.match(content)
        name = head.group(1).upper() if head else ""
        if head and name in _CARDS:
            if name in cards:
                raise ValueError(f"pw.x input line {lineno}: card {name} appears twice")
            current = []
            cards[name] = (head.group(2).strip().strip("{}()").strip().lower(), current)
        elif current is None:
            raise ValueError(f"pw.x input line {lineno}: expected a namelist or card, got {content!r}")
        else:
            current.append((lineno, content.split()))
    return cards


def _integer(token: str, key: str) -> int:
    if not _INTEGER.fullmatch(token):
        raise ValueError(f"pw.x input &system {key} must be an integer, got {token!r}")
    return int(token)


def _real(token: str, key: str) -> fractions.Fraction:
    try:
        return _number(token)
    except ValueError:
        raise ValueError(f"pw.x input &system {key} must be a decimal number, got {token!r}") from None


def _scalar(system: Mapping[str, list[str]], key: str) -> str | None:
    assigned = system.get(key)
    if assigned is None:
        return None
    if len(assigned) != 1:
        raise ValueError(f"pw.x input &system {key} takes one value, got {assigned}")
    return assigned[0]


def _celldm(system: Mapping[str, list[str]]) -> dict[int, str]:
    """``celldm(i)`` tokens, including list assignments and ``r*value`` repeats."""
    celldm: dict[int, str] = {}
    for key, assigned in system.items():
        match = _CELLDM.fullmatch(key)
        if match is None:
            continue
        expanded: list[str] = []
        for token in assigned:
            count, star, value = token.partition("*")
            expanded += [value] * int(count) if star and count.isdigit() else [token]
        for offset, token in enumerate(expanded):
            index = int(match.group(1) or 1) + offset
            if not 1 <= index <= 6 or index in celldm:
                raise ValueError(f"pw.x input &system assigns celldm({index}), outside celldm(1..6) or twice")
            celldm[index] = token
    return celldm


def _lattice_parameters(
    system: Mapping[str, list[str]], ibrav: int
) -> tuple[fractions.Fraction | None, str, fractions.Fraction | None, dict[int, fractions.Fraction]]:
    """Return ``(alat or None, alat unit, alat token precision, celldm)``.

    A/B/C/cos* are mapped to celldm as QE's ``abc2celldm`` does.
    """
    tokens = _celldm(system)
    celldm = {index: _real(token, f"celldm({index})") for index, token in tokens.items()}
    abc_tokens = {key: token for key in _ABC_KEYS if (token := _scalar(system, key)) is not None}
    abc = {key: _real(token, key) for key, token in abc_tokens.items()}
    if celldm and abc:
        raise ValueError("pw.x input &system mixes celldm with A/B/C/cosAB/cosAC/cosBC; pw.x refuses this")
    if abc:
        zero = fractions.Fraction(0)
        a = abc.get("a", zero)
        if a == 0:
            raise ValueError("pw.x input &system B/C/cosAB/cosAC/cosBC need A")
        celldm = {2: abc.get("b", zero) / a, 3: abc.get("c", zero) / a}
        if ibrav == 14:
            celldm |= {4: abc.get("cosbc", zero), 5: abc.get("cosac", zero), 6: abc.get("cosab", zero)}
        elif ibrav in (-12, -13):
            celldm[5] = abc.get("cosac", zero)
        else:
            celldm[4] = abc.get("cosab", zero)
        alat, unit, token = a, "angstrom", abc_tokens["a"]
    else:
        alat, unit, token = celldm.get(1, fractions.Fraction(0)), "bohr", tokens.get(1, "0")
    if alat < 0:
        raise ValueError(f"pw.x input lattice parameter must be positive, got {alat}")
    # pw.x reads celldm(1) = 0 and A = 0 (the defaults) as "not given".
    return (alat or None), unit, combined_precision([token.replace("d", "e").replace("D", "e")]), celldm


def _root(value: Any, what: str) -> tuple[Any, fractions.Fraction]:
    """A square root and an exact bound on its error (``0`` when the root is exact).

    ``exact=None`` is bounded best effort (``exact=True`` factorises arbitrarily large
    radicands: many seconds for 14-digit tokens). When the radicand is beyond the bound the
    rational approximation ``r`` is used, as CellParams does for inexact angles, with the
    error bound ``|r - sqrt(v)| = |r^2 - v| / (r + sqrt(v)) <= |r^2 - v| / r``.
    """
    if value <= 0:
        raise ValueError(f"pw.x input needs {what} > 0, got {value}")
    root = exactmath.sqrt(value)
    if root * root == value:
        return root, fractions.Fraction(0)
    approximation = fractions.Fraction(root)
    if approximation <= 0:
        raise ValueError(f"pw.x input: sqrt({what}) of {value} is too small to approximate")
    return approximation, abs(approximation * approximation - value) / approximation


def _scaled_precision(
    tokens: Sequence[Sequence[str]], factor: Any, factor_precision: fractions.Fraction | None
) -> fractions.Fraction | None:
    """The precision of ``token * factor`` entries: ``δt·|factor| + max|t|·δfactor``, or ``None``."""
    token_precision = combined_precision(token.replace("d", "e").replace("D", "e") for row in tokens for token in row)
    if token_precision is None and factor_precision is None:
        return None
    largest = max(abs(_number(token)) for row in tokens for token in row)
    # ponytail: a surd alat (|a1|) goes through a float; a ~1e-16 relative slack in a tolerance.
    exact = isinstance(factor, (int, fractions.Fraction))
    magnitude = abs(fractions.Fraction(factor if exact else float(factor)))
    zero = fractions.Fraction(0)
    # All-zero integer tokens times an alat claim nothing (a zero precision is no claim).
    return (token_precision or zero) * magnitude + largest * (factor_precision or zero) or None


def _lower(value: Any) -> fractions.Fraction:
    """A rational lower bound of an exact number (itself when rational)."""
    if isinstance(value, (int, fractions.Fraction)):
        return fractions.Fraction(value)
    slack = fractions.Fraction(1, 10**30)
    return value.to_fractions_approx(slack) - slack


def _ibrav_cell(ibrav: int, celldm: Mapping[int, fractions.Fraction]) -> tuple[_Rows, fractions.Fraction]:
    """The lattice vectors of QE's INPUT_PW ``ibrav`` table in units of alat.

    Return them with a bound on their entries' error, which is ``0`` unless a square root had
    to be approximated (see :func:`_root`).
    """

    def ratio(index: int) -> fractions.Fraction:
        value = celldm.get(index, fractions.Fraction(0))
        if value <= 0:
            raise ValueError(f"pw.x input ibrav = {ibrav} needs celldm({index}) > 0 (or B/C), got {value}")
        return value

    def cosine(index: int) -> fractions.Fraction:
        value = celldm.get(index, fractions.Fraction(0))
        if not -1 < value < 1:
            raise ValueError(f"pw.x input ibrav = {ibrav} needs -1 < celldm({index}) < 1, got {value}")
        return value

    h, no_error = _HALF, fractions.Fraction(0)
    root3, root2 = _root(3, "3")[0], _root(2, "2")[0]
    match ibrav:
        case 1:
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]], no_error
        case 2:
            return [[-h, 0, h], [0, h, h], [-h, h, 0]], no_error
        case 3:
            return [[h, h, h], [-h, h, h], [-h, -h, h]], no_error
        case -3:
            return [[-h, h, h], [h, -h, h], [h, h, -h]], no_error
        case 4:
            return [[1, 0, 0], [-h, root3 / 2, 0], [0, 0, ratio(3)]], no_error
        case 5 | -5:
            g = cosine(4)
            (tx, ex), (ty, ey), (tz, ez) = (
                _root((1 - g) / 2, "1 - cos"),
                _root((1 - g) / 6, "1 - cos"),
                _root((1 + 2 * g) / 3, "1 + 2cos"),
            )
            # Entries are tx, ty, 2ty, tz or (tz -/+ k ty)/sqrt(3) with k <= 2 sqrt(2), whose errors
            # are at most (1 + 2 sqrt(2))/sqrt(3) < 3 times the largest root error.
            error = 3 * max(ex, ey, ez)
            if ibrav == 5:
                return [[tx, -ty, tz], [0, 2 * ty, tz], [-tx, -ty, tz]], error
            u, v = (tz - 2 * root2 * ty) / root3, (tz + root2 * ty) / root3
            return [[u, v, v], [v, u, v], [v, v, u]], error
        case 6:
            return [[1, 0, 0], [0, 1, 0], [0, 0, ratio(3)]], no_error
        case 7:
            c = ratio(3) / 2
            return [[h, -h, c], [h, h, c], [-h, -h, c]], no_error
        case 8:
            return [[1, 0, 0], [0, ratio(2), 0], [0, 0, ratio(3)]], no_error
        case 9:
            b = ratio(2) / 2
            return [[h, b, 0], [-h, b, 0], [0, 0, ratio(3)]], no_error
        case -9:
            b = ratio(2) / 2
            return [[h, -b, 0], [h, b, 0], [0, 0, ratio(3)]], no_error
        case 91:
            b, c = ratio(2) / 2, ratio(3) / 2
            return [[1, 0, 0], [0, b, -c], [0, b, c]], no_error
        case 10:
            b, c = ratio(2) / 2, ratio(3) / 2
            return [[h, 0, c], [h, b, 0], [0, b, c]], no_error
        case 11:
            b, c = ratio(2) / 2, ratio(3) / 2
            return [[h, b, c], [-h, b, c], [-h, -b, c]], no_error
        case 12 | 13:
            b, c, g = ratio(2), ratio(3), cosine(4)
            sine, error = _root(1 - g * g, "1 - cos^2")
            v2 = [b * g, b * sine, 0]
            return ([[1, 0, 0], v2, [0, 0, c]] if ibrav == 12 else [[h, 0, -c / 2], v2, [h, 0, c / 2]]), b * error
        case -12 | -13:
            b, c, be = ratio(2), ratio(3), cosine(5)
            sine, error = _root(1 - be * be, "1 - cos^2")
            v3 = [c * be, 0, c * sine]
            return ([[1, 0, 0], [0, b, 0], v3] if ibrav == -12 else [[h, b / 2, 0], [-h, b / 2, 0], v3]), c * error
        case 14:
            b, c = ratio(2), ratio(3)
            al, be, ga = cosine(4), cosine(5), cosine(6)
            sg, e_sg = _root(1 - ga * ga, "1 - cos(gamma)^2")
            volume, e_volume = _root(1 + 2 * al * be * ga - al * al - be * be - ga * ga, "the triclinic volume factor")
            rows = [[1, 0, 0], [b * ga, b * sg, 0], [c * be, c * (al - be * ga) / sg, c * volume / sg]]
            if not e_sg and not e_volume:
                return rows, no_error
            # With the true sine s >= sg - e_sg = d > 0: |K/sg - K/s| <= |K| e_sg / (sg d) and
            # |V/sg - V*/s| <= e_volume / d + V e_sg / (sg d), evaluated on rational bounds
            # (lower for sg and d, upper for V) since either root may be an exact surd.
            sg_low, volume_high = _lower(sg), -_lower(-volume)
            d = sg_low - e_sg
            if d <= 0:
                raise ValueError("pw.x input ibrav = 14 cell is too close to degenerate to approximate")
            error = max(
                b * e_sg,
                c * abs(al - be * ga) * e_sg / (sg_low * d),
                c * (e_volume / d + volume_high * e_sg / (sg_low * d)),
            )
            return rows, error
    raise ValueError(
        f"pw.x input ibrav = {ibrav} is not supported "
        "(supported: 0, 1, 2, 3, -3, 4, 5, -5, 6, 7, 8, 9, -9, 91, 10, 11, 12, -12, 13, -13, 14)"
    )


def _scaled(rows: Sequence[Sequence[Any]], factor: Any) -> _Rows:
    return [[entry * factor for entry in row] for row in rows]


def _card_rows(
    card: str, rows: list[tuple[int, list[str]]], count: int, width: tuple[int, ...]
) -> list[tuple[int, list[str]]]:
    if len(rows) != count:
        raise ValueError(f"pw.x input card {card} must have {count} lines, got {len(rows)}")
    for lineno, fields in rows:
        if len(fields) not in width:
            raise ValueError(f"pw.x input line {lineno}: {card} line has {len(fields)} fields, expected {width}")
    return rows


def _decimal(token: str, lineno: int, card: str) -> str:
    try:
        _number(token)
    except ValueError:
        raise ValueError(
            f"pw.x input line {lineno}: {card} entry {token!r} is not a decimal number "
            "(algebraic expressions are not supported)"
        ) from None
    return token


def read_pw_input(source: Any, *, precision: float | None = None) -> dict[str, Any]:
    """Parse the geometry of a Quantum ESPRESSO ``pw.x`` input file into a neutral payload.

    Supported: ``ibrav = 0`` with ``CELL_PARAMETERS {alat|bohr|angstrom}`` and every lattice
    of QE's INPUT_PW ``ibrav`` table (1, 2, 3, -3, 4, 5, -5, 6, 7, 8, 9, -9, 91, 10, 11, 12,
    -12, 13, -13, 14), built exactly from ``celldm(1..6)`` (Bohr) or ``A, B, C, cosAB, cosAC,
    cosBC`` (Å), and ``ATOMIC_POSITIONS {alat|bohr|angstrom|crystal}`` (no option means
    ``alat``; trailing ``if_pos`` columns are ignored). The lattice parameter alat is
    ``celldm(1)`` or ``A``; for ``CELL_PARAMETERS bohr|angstrom`` it is the length of the
    first vector, and a ``CELL_PARAMETERS`` card without option is ``alat`` when a lattice
    parameter is given and ``bohr`` otherwise, as INPUT_PW documents. Species names, element
    symbols (by QE's label rule: an element symbol optionally followed by digits or by
    ``_x``/``-x``, at most 3 characters) and masses come from ``ATOMIC_SPECIES``. Decimal
    tokens are kept as strings; values scaled by alat or built from the ``ibrav`` table are
    exact numbers (surds where the table has square roots) whose ``cell_precision`` /
    ``positions_precision`` is that of the tokens and the alat token. Square roots are
    attempted with bounded effort only: one not exactly representable from the given digits
    is a rational approximation whose error bound is added to the claimed precision, and
    ``|a1|`` is computed only when an alat-unit card needs it.

    :param source: A filename (compressed files are decompressed), open text stream, or
        iterable of lines.
    :param precision: Cartesian coordinate precision as a length in Å, or ``None`` to derive
        it from the written digits. Must be a finite number greater than zero when given.
    :return: A ``qe-pw-input`` geometry payload with the original text in ``raw``.
    :raises ValueError: If the input is malformed or not UTF-8, or uses an unsupported or
        ambiguous construct: ``space_group``/``crystal_sg``, an unsupported ``ibrav``, both
        ``celldm`` and ``A``-style parameters, a lattice parameter together with
        ``CELL_PARAMETERS bohr|angstrom``, ``CELL_PARAMETERS`` with ``ibrav`` other than 0,
        algebraic expressions, or a species label that names no element.
    """
    _check_load_precision(precision, "read_pw_input")
    text = _source_text(source, "pw.x input")
    namelists, offset = _namelists(text)
    system = namelists.get("system")
    if system is None:
        raise ValueError("not a pw.x input: no &system namelist")
    cards = _cards(text, offset)

    def required(key: str) -> int:
        token = _scalar(system, key)
        if token is None:
            raise ValueError(f"pw.x input &system needs {key}")
        return _integer(token, key)

    space_group = _scalar(system, "space_group")
    if space_group is not None and _integer(space_group, "space_group") != 0:
        raise ValueError("pw.x input space_group (crystal_sg) geometry is not supported; use an ibrav/cell input")
    ibrav, nat, ntyp = required("ibrav"), required("nat"), required("ntyp")
    if nat <= 0 or ntyp <= 0:
        raise ValueError(f"pw.x input &system nat and ntyp must be positive, got {nat} and {ntyp}")
    alat, alat_unit, alat_precision, celldm = _lattice_parameters(system, ibrav)

    species: list[dict[str, Any]] = []
    for lineno, (label, mass, _pseudo) in _card_rows(
        "ATOMIC_SPECIES", cards.get("ATOMIC_SPECIES", ("", []))[1], ntyp, (3,)
    ):
        if any(item["name"] == label for item in species):
            raise ValueError(f"pw.x input line {lineno}: species {label!r} is declared twice")
        element = _label_element(label, f"pw.x input line {lineno}")
        # ponytail: a non-positive mass (pw.x then uses its own value) is read as no mass claim.
        positive = _number(_decimal(mass, lineno, "ATOMIC_SPECIES")) > 0
        species.append({"name": label, "element": element, "mass": mass if positive else None})

    cell_card = cards.get("CELL_PARAMETERS")
    cell_precision: fractions.Fraction | None = None  # None: derived from the cell tokens
    first_vector: list[str] | None = None  # a1 of bohr/angstrom vectors, whose length is alat
    if ibrav == 0:
        if cell_card is None:
            raise ValueError("pw.x input with ibrav = 0 needs a CELL_PARAMETERS card")
        option, rows = cell_card
        option = option or ("alat" if alat is not None else "bohr")
        tokens = [
            [_decimal(token, lineno, "CELL_PARAMETERS") for token in fields]
            for lineno, fields in _card_rows("CELL_PARAMETERS", rows, 3, (3,))
        ]
        if option == "alat":
            if alat is None:
                raise ValueError("pw.x input CELL_PARAMETERS alat needs celldm(1) or A")
            cell: _Rows = _scaled([[_number(token) for token in row] for row in tokens], alat)
            cell_unit, cell_precision = alat_unit, _scaled_precision(tokens, alat, alat_precision)
        elif option in ("bohr", "angstrom"):
            if alat is not None:
                raise ValueError(f"pw.x input gives a lattice parameter together with CELL_PARAMETERS {option}")
            cell, cell_unit, first_vector, alat_unit = tokens, option, tokens[0], option
        else:
            raise ValueError(f"pw.x input CELL_PARAMETERS option {option!r} is not alat, bohr or angstrom")
    else:
        if cell_card is not None:
            raise ValueError(f"pw.x input has CELL_PARAMETERS with ibrav = {ibrav}; the geometry is ambiguous")
        if alat is None:
            raise ValueError(f"pw.x input ibrav = {ibrav} needs celldm(1) or A")
        rows_in_alat, root_error = _ibrav_cell(ibrav, celldm)
        cell, cell_unit = _scaled(rows_in_alat, alat), alat_unit
        # ponytail: only the alat token's precision and any root approximation propagate; the
        # celldm(2..6)/B/C/cos* tokens' own digits do not (needs derivatives of the formulas).
        token_error = fractions.Fraction(0)
        if alat_precision is not None:
            token_error = alat_precision * max(abs(fractions.Fraction(float(x))) for row in rows_in_alat for x in row)
        cell_precision = (token_error + alat * root_error) or None

    if "ATOMIC_POSITIONS" not in cards:
        raise ValueError("pw.x input has no ATOMIC_POSITIONS card")
    option, rows = cards["ATOMIC_POSITIONS"]
    option = option or "alat"
    labels: list[str] = []
    positions: _Rows = []
    for lineno, fields in _card_rows("ATOMIC_POSITIONS", rows, nat, (4, 7)):
        if any(flag not in ("0", "1") for flag in fields[4:]):
            raise ValueError(f"pw.x input line {lineno}: if_pos flags must be 0 or 1, got {fields[4:]}")
        labels.append(fields[0])
        positions.append([_decimal(token, lineno, "ATOMIC_POSITIONS") for token in fields[1:4]])
    payload: dict[str, Any] = {"format": FORMAT, "cell": cell, "cell_unit": cell_unit, "bohr_angstrom": BOHR_ANGSTROM}
    if cell_precision is not None:
        payload["cell_precision"] = cell_precision
    if option == "crystal":
        payload |= {"positions": positions, "positions_kind": "fractional"}
    elif option == "alat":
        if alat is None:
            assert first_vector is not None  # every other branch above sets alat or raises
            # INPUT_PW: with bohr/angstrom vectors, alat is the length of the first one.
            alat, root_error = _root(sum(_number(token) ** 2 for token in first_vector), "|a1|^2")
            a1_precision = combined_precision(token.replace("d", "e").replace("D", "e") for token in first_vector)
            # |grad |a1|| <= sqrt(3) < 2, plus the error of an approximated root.
            alat_precision = (2 * (a1_precision or 0) + root_error) or None
        payload |= {
            "positions": _scaled([[_number(token) for token in row] for row in positions], alat),
            "positions_kind": "cartesian",
            "positions_unit": alat_unit,
        }
        positions_precision = _scaled_precision(positions, alat, alat_precision)
        if positions_precision is not None:
            payload["positions_precision"] = positions_precision
    elif option in ("bohr", "angstrom"):
        payload |= {"positions": positions, "positions_kind": "cartesian", "positions_unit": option}
    else:
        raise ValueError(
            f"pw.x input ATOMIC_POSITIONS option {option!r} is not supported (use alat, bohr, angstrom or crystal)"
        )
    payload |= {"species": species, "species_at_sites": labels, "raw": text}
    if precision is not None:
        payload["precision_override"] = precision
    return payload


def _pw_input_payload_from_structure(obj: Any) -> dict[str, Any]:
    """Serialize a structure to a ``qe-pw-input`` geometry payload.

    :param obj: Anything :class:`~httk.atomistic.UnitcellStructureView` accepts.
    :return: The geometry payload (Å cell, fractional positions).
    """
    return _geometry_from_structure(obj, FORMAT)


def _write_pw_input_payload(
    destination: str | os.PathLike[str] | io.TextIOBase, payload: Mapping[str, Any], **kwargs: Any
) -> None:
    r"""Write a ``qe-pw-input`` payload: its ``raw`` text verbatim, else the rendered geometry.

    :param destination: A filesystem path or an open text stream.
    :param payload: The geometry payload.
    :param \**kwargs: The ``masses`` and ``pseudopotentials`` mappings of the renderer.
    """
    _write_text(destination, payload, kwargs, _render_pw_input)


def _render_pw_input(
    payload: Mapping[str, Any],
    *,
    masses: Mapping[str, float] | None = None,
    pseudopotentials: Mapping[str, str] | None = None,
) -> str:
    """Render the geometry part of a ``pw.x`` input.

    The text is ``&system`` (``ibrav = 0``, ``nat``, ``ntyp``), ``ATOMIC_SPECIES``,
    ``CELL_PARAMETERS angstrom`` and ``ATOMIC_POSITIONS crystal``.

    :param payload: A geometry payload in any units.
    :param masses: Species name to mass in atomic mass units; otherwise the species mass, else
        ``1.0`` (pw.x needs a number).
    :param pseudopotentials: Species name to pseudopotential file name; otherwise
        ``<Element>.UPF``.
    :return: The input text.
    :raises ValueError: If the structure is not periodic in all directions, a species name is
        not a valid QE label for its element, or a keyword names unknown species or holds an
        invalid value.
    """
    data = _geometry_from_structure(_structure_from_geometry(payload), FORMAT)
    if not all(data["periodicity"]):
        raise ValueError("pw.x cells are periodic in all three directions; this structure is not")
    names = [item["name"] for item in data["species"]]
    options = {"masses": masses or {}, "pseudopotentials": pseudopotentials or {}}
    for keyword, mapping in options.items():
        if not isinstance(mapping, Mapping):
            raise ValueError(f"{keyword} must be a mapping from species name, got {mapping!r}")
        unknown = sorted(set(mapping) - set(names))
        if unknown:
            raise ValueError(f"{keyword} names species {unknown} not in the structure {names}")
    lines = ["&system", "  ibrav = 0", f"  nat = {len(data['species_at_sites'])}", f"  ntyp = {len(names)}", "/"]
    lines.append("ATOMIC_SPECIES")
    for item in data["species"]:
        name, element = item["name"], item["element"]
        if len(name) > 3 or _element_from_label(name) != element:
            raise ValueError(
                f"species name {name!r} ({element}) is not a valid pw.x label ({_LABEL_RULE}); "
                "rename the species, e.g. to its element symbol"
            )
        mass = options["masses"].get(name)
        if mass is None:
            rendered = item["mass"] or "1.0"
        elif isinstance(mass, bool) or not isinstance(mass, numbers.Real) or not math.isfinite(mass) or mass <= 0:
            raise ValueError(f"masses[{name!r}] must be a finite positive number, got {mass!r}")
        else:
            rendered = repr(float(mass))
        pseudo = options["pseudopotentials"].get(name, f"{element}.UPF")
        if not isinstance(pseudo, str) or not pseudo or re.search(r"[\s!#]", pseudo):
            raise ValueError(
                f"pseudopotentials[{name!r}] must be a file name without blanks, '!' or '#', got {pseudo!r}"
            )
        lines.append(f"{name} {rendered} {pseudo}")
    lines.append("CELL_PARAMETERS angstrom")
    lines += [" ".join(row) for row in data["cell"]]
    lines.append("ATOMIC_POSITIONS crystal")
    lines += [f"{name} {' '.join(row)}" for name, row in zip(data["species_at_sites"], data["positions"], strict=True)]
    return "\n".join(lines) + "\n"
