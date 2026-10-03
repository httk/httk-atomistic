"""Read native text atom/custom dumps without rounding decimal tokens."""

import os
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from httk.core import TextstreamFileView

__all__ = ["LammpsDumpFile", "LammpsDumpFrame", "read_lammps_dump"]

_COORDINATES = {
    "x": ("x", "y", "z"),
    "xs": ("xs", "ys", "zs"),
    "xu": ("xu", "yu", "zu"),
    "xsu": ("xsu", "ysu", "zsu"),
}


@dataclass(frozen=True, slots=True)
class LammpsDumpFrame:
    """One exact, ID-sorted snapshot in the dump's source units and axes.

    Positions include the laboratory box origin. Unwrapped positions are absent
    unless supplied directly or reconstructible from all three image flags.
    Segment numbers increase whenever printed steps fail to increase.

    :param step: Printed integer step.
    :param segment: Zero-based segment identified by non-increasing steps.
    :param time: Printed time, when present.
    :param units: Printed unit style, when present.
    :param cell: Three row lattice vectors.
    :param origin: Laboratory box origin.
    :param periodicity: Periodic axes, or absent when the header omits flags.
    :param atom_ids: Sorted persistent atom IDs.
    :param atom_types: Types aligned with IDs.
    :param positions: Cartesian positions in the printed coordinate convention.
    :param unwrapped_positions: Unwrapped laboratory positions, if known.
    :param velocities: Cartesian velocities, if printed.
    :param forces: Cartesian forces, if printed.
    """

    step: int
    segment: int
    time: Fraction | None
    units: str | None
    cell: tuple[tuple[Fraction, ...], ...]
    origin: tuple[Fraction, ...]
    periodicity: tuple[bool, ...] | None
    atom_ids: tuple[int, ...]
    atom_types: tuple[int, ...]
    positions: tuple[tuple[Fraction, ...], ...]
    unwrapped_positions: tuple[tuple[Fraction, ...], ...] | None
    velocities: tuple[tuple[Fraction, ...], ...] | None
    forces: tuple[tuple[Fraction, ...], ...] | None


@dataclass(frozen=True, slots=True)
class LammpsDumpFile:
    """Reopen and stream a native text atom/custom dump on every traversal.

    Atom IDs and their types must remain constant within each segment. Geometry
    and vector values retain exact decimal values. No species or units are inferred.
    Orthogonal, restricted triclinic and general ``abc origin`` boxes are supported.

    :param path: Dump path, optionally compressed by a registered codec.
    :param coordinates: ``auto`` requires one complete coordinate representation;
        otherwise select ``x``, ``xs``, ``xu`` or ``xsu`` explicitly.
    :raises ValueError: If the coordinate selector is invalid.
    """

    path: str | os.PathLike[str]
    coordinates: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.coordinates not in ("auto", *_COORDINATES):
            raise ValueError("coordinates must be auto, x, xs, xu or xsu")

    def frames(self) -> Iterator[LammpsDumpFrame]:
        """Yield snapshots with memory bounded by the current atom count.

        :yields: Exact snapshots in source order.
        :raises ValueError: If syntax, geometry, atom identity or numeric fields are invalid.
        """
        stream = TextstreamFileView(Path(self.path))
        try:
            lines = iter(stream)
            previous_step: int | None = None
            segment = 0
            identity: tuple[tuple[int, int], ...] | None = None
            coordinate_style: str | None = None
            units: str | None = None
            time: Fraction | None = None
            for line in lines:
                header = line.strip()
                if not header:
                    continue
                if header == "ITEM: UNITS":
                    units = _line(lines)
                    continue
                if header == "ITEM: TIME":
                    time = _number(_line(lines))
                    continue
                if header != "ITEM: TIMESTEP":
                    raise ValueError(f"expected ITEM: TIMESTEP, found {header!r}")
                step = _integer(_line(lines))
                if previous_step is not None and step <= previous_step:
                    segment += 1
                    identity = None
                    coordinate_style = None
                previous_step = step
                header = _line(lines)
                if header == "ITEM: TIME":
                    time = _number(_line(lines))
                    header = _line(lines)
                if header != "ITEM: NUMBER OF ATOMS":
                    raise ValueError("expected ITEM: NUMBER OF ATOMS")
                count = _integer(_line(lines))
                if count <= 0:
                    raise ValueError("a dump frame must contain at least one atom")
                box = _line(lines).split()
                if box[:3] != ["ITEM:", "BOX", "BOUNDS"]:
                    raise ValueError("expected ITEM: BOX BOUNDS")
                rows = tuple(tuple(_number(value) for value in _line(lines).split()) for _ in range(3))
                cell, origin, periodicity = _box(box[3:], rows)
                atoms = _line(lines).split()
                if atoms[:2] != ["ITEM:", "ATOMS"]:
                    raise ValueError("expected ITEM: ATOMS")
                columns = atoms[2:]
                if len(set(columns)) != len(columns) or "id" not in columns or "type" not in columns:
                    raise ValueError("atom columns must be unique and contain id and type")
                available = [name for name, triplet in _COORDINATES.items() if _triplet(columns, triplet)]
                coordinate = self.coordinates
                if coordinate == "auto":
                    if len(available) != 1:
                        raise ValueError("select coordinates explicitly when the dump has multiple representations")
                    coordinate = available[0]
                if coordinate not in available:
                    raise ValueError(f"dump does not contain the selected {coordinate!r} coordinates")
                if coordinate_style is not None and coordinate != coordinate_style:
                    raise ValueError("coordinate representation changes within a dump segment")
                coordinate_style = coordinate
                image = _triplet(columns, ("ix", "iy", "iz"))
                velocity = _triplet(columns, ("vx", "vy", "vz"))
                force = _triplet(columns, ("fx", "fy", "fz"))
                records: list[dict[str, str]] = []
                for _ in range(count):
                    values = _line(lines).split()
                    if len(values) != len(columns):
                        raise ValueError("atom row length does not match its header")
                    records.append(dict(zip(columns, values, strict=True)))
                records.sort(key=lambda row: _integer(row["id"]))
                ids = tuple(_integer(row["id"]) for row in records)
                types = tuple(_integer(row["type"]) for row in records)
                if len(set(ids)) != count or min(ids) <= 0 or min(types) <= 0:
                    raise ValueError("atom IDs and types must be positive; IDs must be unique")
                current_identity = tuple(zip(ids, types, strict=True))
                if identity is not None and current_identity != identity:
                    raise ValueError("atom IDs or types change within a dump segment")
                identity = current_identity
                coords = tuple(tuple(_number(row[name]) for name in _COORDINATES[coordinate]) for row in records)
                positions = (
                    tuple(_cartesian(row, cell, origin) for row in coords) if coordinate in {"xs", "xsu"} else coords
                )
                unwrapped = positions if coordinate in {"xu", "xsu"} else None
                if unwrapped is None and image:
                    flags = tuple(
                        tuple(Fraction(_integer(row[name])) for name in ("ix", "iy", "iz")) for row in records
                    )
                    unwrapped = tuple(
                        tuple(p[j] + sum(flags[i][k] * cell[k][j] for k in range(3)) for j in range(3))
                        for i, p in enumerate(positions)
                    )
                yield LammpsDumpFrame(
                    step,
                    segment,
                    time,
                    units,
                    cell,
                    origin,
                    periodicity,
                    ids,
                    types,
                    positions,
                    unwrapped,
                    tuple(tuple(_number(row[name]) for name in ("vx", "vy", "vz")) for row in records)
                    if velocity
                    else None,
                    tuple(tuple(_number(row[name]) for name in ("fx", "fy", "fz")) for row in records)
                    if force
                    else None,
                )
                time = None
        finally:
            stream.close()


def read_lammps_dump(source: str | os.PathLike[str], **kwargs: object) -> dict[str, object]:
    r"""Return a lazy neutral payload for a native LAMMPS text dump.

    Reader keyword arguments are retained for the trajectory adapter; the file is
    not opened or scanned here.

    :param source: Filesystem dump path, optionally compressed.
    :param \**kwargs: Explicit trajectory options such as species and units.
    :return: A neutral ``lammps-dump`` payload.
    :raises TypeError: If ``source`` is not path-like.
    :raises FileNotFoundError: If the path is not a file.
    """
    if not isinstance(source, str | os.PathLike):
        raise TypeError("LAMMPS dump source must be path-like")
    path = Path(os.fsdecode(os.fspath(source)))
    if not path.is_file():
        raise FileNotFoundError(f"LAMMPS dump source does not exist: {path!s}")
    return {"format": "lammps-dump", "source": path, "options": dict(kwargs)}


def _line(lines: Iterator[str]) -> str:
    try:
        return next(lines).strip()
    except StopIteration as exc:
        raise ValueError("truncated LAMMPS dump frame") from exc


def _number(token: str) -> Fraction:
    try:
        return Fraction(token)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid finite numeric token {token!r}") from exc


def _integer(token: str) -> int:
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(f"invalid integer token {token!r}") from exc


def _triplet(columns: list[str], names: tuple[str, str, str]) -> bool:
    present = sum(name in columns for name in names)
    if present not in (0, 3):
        raise ValueError(f"incomplete atom column triplet {names!r}")
    return present == 3


def _cartesian(
    scaled: tuple[Fraction, ...], cell: tuple[tuple[Fraction, ...], ...], origin: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    return tuple(origin[j] + sum(scaled[k] * cell[k][j] for k in range(3)) for j in range(3))


def _box(
    fields: list[str], rows: tuple[tuple[Fraction, ...], ...]
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...], tuple[bool, ...] | None]:
    zero = Fraction(0)
    if fields[:2] == ["abc", "origin"]:
        if any(len(row) != 4 for row in rows):
            raise ValueError("general triclinic box requires three vector-plus-origin rows")
        cell = tuple(row[:3] for row in rows)
        origin = tuple(row[3] for row in rows)
        flags = fields[2:]
        # Native general-triclinic dumps do not print boundary flags.
        periodicity = None if not flags else _periodicity(flags)
    elif fields[:3] == ["xy", "xz", "yz"]:
        if any(len(row) != 3 for row in rows):
            raise ValueError("restricted triclinic box requires three bounds-plus-tilt rows")
        xy, xz, yz = (row[2] for row in rows)
        xlo = rows[0][0] - min(zero, xy, xz, xy + xz)
        xhi = rows[0][1] - max(zero, xy, xz, xy + xz)
        ylo = rows[1][0] - min(zero, yz)
        yhi = rows[1][1] - max(zero, yz)
        origin = (xlo, ylo, rows[2][0])
        cell = ((xhi - xlo, zero, zero), (xy, yhi - ylo, zero), (xz, yz, rows[2][1] - rows[2][0]))
        periodicity = _periodicity(fields[3:])
        if any(cell[i][i] <= 0 for i in range(3)):
            raise ValueError("box lengths must be positive")
    else:
        if any(len(row) != 2 for row in rows):
            raise ValueError("orthogonal box requires three lower/upper bounds")
        origin = tuple(row[0] for row in rows)
        lengths = tuple(row[1] - row[0] for row in rows)
        if any(length <= 0 for length in lengths):
            raise ValueError("box lengths must be positive")
        cell = tuple(tuple(lengths[i] if i == j else zero for j in range(3)) for i in range(3))
        periodicity = _periodicity(fields)
    a, b, c = cell
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if determinant <= 0:
        raise ValueError("cell must be nonsingular and right-handed")
    return cell, origin, periodicity


def _periodicity(flags: list[str]) -> tuple[bool, ...]:
    if len(flags) != 3 or any(len(flag) != 2 or set(flag) - set("pfsm") for flag in flags):
        raise ValueError("box must declare three valid boundary flags")
    if any("p" in flag and flag != "pp" for flag in flags):
        raise ValueError("periodic boundaries must occur on both faces")
    return tuple(flag == "pp" for flag in flags)
