"""Read the geometries a Quantum ESPRESSO ``pw.x`` run prints, as trajectory frames."""

import fractions
import re
from typing import Any

from httk.core import combined_precision

from httk.atomistic.integrations._geometry import _number
from httk.atomistic.io import _check_load_precision

from .pw_input import BOHR_ANGSTROM, _label_element, _scaled_precision, _source_text

FORMAT = "qe-pw-output"

_PARENTHESIZED = re.compile(r"=\s*\(([^)]*)\)")
_TAU = re.compile(r"\s*\d+\s+(\S+)\s+tau\(\s*\d+\)\s*=\s*\(([^)]*)\)")
_CELL_HEAD = re.compile(r"CELL_PARAMETERS\s*\(\s*(?:alat\s*=\s*(\S+)|(bohr|angstrom))\s*\)")
_POSITIONS_HEAD = re.compile(r"ATOMIC_POSITIONS\s*\(\s*(alat|bohr|angstrom|crystal)\s*\)")


def _triple(text: str, lineno: int) -> list[str]:
    tokens = text.split()
    if len(tokens) != 3:
        raise ValueError(f"pw.x output line {lineno}: expected three numbers, got {text.strip()!r}")
    for token in tokens:
        try:
            _number(token)
        except ValueError:
            raise ValueError(f"pw.x output line {lineno}: {token!r} is not a decimal number") from None
    return tokens


def _times(rows: list[list[str]], factor: fractions.Fraction) -> list[list[Any]]:
    return [[_number(token) * factor for token in row] for row in rows]


def read_pw_output(source: Any, *, precision: float | None = None) -> dict[str, Any]:
    """Parse the geometries of a Quantum ESPRESSO ``pw.x`` output into a frames payload.

    Frame 0 is the header geometry (``lattice parameter (alat)``, ``crystal axes`` and the
    ``site n. ... positions (alat units)`` table); it is printed with only 4 to 7 decimals and
    is therefore lossy: the ``pw.x`` input holds the exact initial geometry. Values printed in
    alat units claim the precision of their digits and of the printed alat. Its site count
    must match ``number of atoms/cell``. Every later
    ``ATOMIC_POSITIONS`` block is one more frame, on the most recent ``CELL_PARAMETERS``
    block's cell (``vc-relax``) or else the header cell. The ``Begin/End final coordinates``
    block is a frame only when it differs from the last one. Species names and masses come from
    the header's atomic species table, elements from QE's label rule.

    :param source: A filename (compressed files are decompressed), open text stream, or
        iterable of lines.
    :param precision: Cartesian coordinate precision as a length in Å for every frame, or
        ``None`` to derive it from the written digits. Must be a finite number greater than
        zero when given.
    :return: A ``qe-pw-output`` frames payload.
    :raises ValueError: If the text is not UTF-8 or not a ``pw.x`` output with a header
        geometry, or a geometry block is malformed or truncated.
    """
    _check_load_precision(precision, "read_pw_output")
    lines = _source_text(source, "pw.x output").splitlines()

    def line(index: int, what: str) -> str:
        if index >= len(lines):
            raise ValueError(f"pw.x output ends inside {what}")
        return lines[index]

    alat: fractions.Fraction | None = None
    alat_precision: fractions.Fraction | None = None
    nat: int | None = None
    axes: list[list[str]] | None = None
    species: list[dict[str, Any]] | None = None
    cell: tuple[list[list[Any]], str, fractions.Fraction | None] | None = None  # rows, unit, precision
    frames: list[dict[str, Any]] = []
    final_start: int | None = None

    def frame(
        labels: list[str],
        positions: list[list[Any]],
        kind: str,
        unit: str | None,
        positions_precision: fractions.Fraction | None = None,
    ) -> None:
        if cell is None or species is None:
            raise ValueError("not a pw.x output: no header cell and atomic species table before the positions")
        item: dict[str, Any] = {
            "cell": cell[0],
            "cell_unit": cell[1],
            "bohr_angstrom": BOHR_ANGSTROM,
            "positions": positions,
            "positions_kind": kind,
            "species": species,
            "species_at_sites": labels,
        }
        if unit is not None:
            item["positions_unit"] = unit
        if cell[2] is not None:
            item["cell_precision"] = cell[2]
        if positions_precision is not None:
            item["positions_precision"] = positions_precision
        if precision is not None:
            item["precision_override"] = precision
        frames.append(item)

    index = 0
    while index < len(lines):
        text = lines[index].strip()
        if alat is None and text.startswith("lattice parameter (alat)"):
            match = re.search(r"=\s*(\S+)", text)
            alat = _number(match.group(1)) if match else None
            if not alat or alat <= 0 or match is None:
                raise ValueError(f"pw.x output line {index + 1}: invalid lattice parameter {text!r}")
            alat_precision = combined_precision([match.group(1)])
        elif nat is None and text.startswith("number of atoms/cell"):
            match = re.fullmatch(r"number of atoms/cell\s*=\s*(\d+)", text)
            if match is None:
                raise ValueError(f"pw.x output line {index + 1}: invalid atom count {text!r}")
            nat = int(match.group(1))
        elif axes is None and text.startswith("crystal axes:"):
            axes = []
            for k in range(3):
                match = _PARENTHESIZED.search(line(index + 1 + k, "the crystal axes"))
                if match is None:
                    raise ValueError(f"pw.x output line {index + 2 + k}: expected a crystal axis")
                axes.append(_triple(match.group(1), index + 2 + k))
            if alat is None:
                raise ValueError("pw.x output prints crystal axes before the lattice parameter")
            cell = (_times(axes, alat), "bohr", _scaled_precision(axes, alat, alat_precision))
            index += 3
        elif species is None and text.startswith("atomic species") and "mass" in text:
            species = []
            while (row := line(index + 1, "the atomic species table").split()) and len(row) >= 3:
                index += 1
                where = f"pw.x output line {index + 1}"
                species.append({"name": row[0], "element": _label_element(row[0], where), "mass": row[2]})
        elif not frames and text.startswith("site n.") and "positions (alat units)" in text:
            if alat is None:
                raise ValueError("pw.x output prints positions before the lattice parameter")
            labels, positions = [], []
            while match := _TAU.match(line(index + 1, "the site table")):
                index += 1
                labels.append(match.group(1))
                positions.append(_triple(match.group(2), index + 1))
            if len(labels) != nat:
                raise ValueError(f"pw.x output line {index + 2}: the site table lists {len(labels)} sites, not {nat}")
            frame(
                labels, _times(positions, alat), "cartesian", "bohr", _scaled_precision(positions, alat, alat_precision)
            )
        elif text.startswith("CELL_PARAMETERS"):
            match = _CELL_HEAD.fullmatch(text)
            if match is None:
                raise ValueError(f"pw.x output line {index + 1}: unsupported {text!r}")
            rows = [_triple(line(index + 1 + k, "CELL_PARAMETERS"), index + 2 + k) for k in range(3)]
            if match.group(1) is not None:
                alat, alat_precision = _number(match.group(1)), combined_precision([match.group(1)])
                cell = (_times(rows, alat), "bohr", _scaled_precision(rows, alat, alat_precision))
            else:
                cell = (list(rows), match.group(2), None)
            index += 3
        elif text.startswith("ATOMIC_POSITIONS"):
            match = _POSITIONS_HEAD.fullmatch(text)
            if match is None:
                raise ValueError(f"pw.x output line {index + 1}: unsupported {text!r}")
            if not frames:
                raise ValueError("not a pw.x output: ATOMIC_POSITIONS before the header geometry")
            labels, positions = [], []
            for k in range(len(frames[0]["species_at_sites"])):
                fields = line(index + 1 + k, "ATOMIC_POSITIONS").split()
                if len(fields) not in (4, 7):
                    raise ValueError(f"pw.x output line {index + 2 + k}: malformed ATOMIC_POSITIONS line")
                labels.append(fields[0])
                positions.append(_triple(" ".join(fields[1:4]), index + 2 + k))
            index += len(labels)
            option = match.group(1)
            if option == "crystal":
                frame(labels, positions, "fractional", None)
            elif option == "alat":
                assert alat is not None  # frames exist, so the header set alat
                frame(
                    labels,
                    _times(positions, alat),
                    "cartesian",
                    "bohr",
                    _scaled_precision(positions, alat, alat_precision),
                )
            else:
                frame(labels, positions, "cartesian", option)
        elif text == "Begin final coordinates":
            final_start = len(frames)
        elif text == "End final coordinates" and final_start is not None:
            # The final block normally repeats the last step's geometry: keep one copy.
            if len(frames) == final_start + 1 and final_start > 0 and frames[-1] == frames[-2]:
                frames.pop()
            final_start = None
        index += 1
    if not frames:
        raise ValueError("not a pw.x output: no header geometry (lattice parameter, crystal axes, site table)")
    return {"format": FORMAT, "frames": frames}
