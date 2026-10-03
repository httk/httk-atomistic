"""Stream exact LAMMPS dump trajectories into canonical structures and observables."""

import os
from collections.abc import Iterator, Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar, Self

from httk.core import SurdVector

from httk.atomistic.integrations.lammps.io.dump import LammpsDumpFile, LammpsDumpFrame
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.trajectory.backend import TrajectoryBackend

__all__ = ["LammpsTrajectory"]

_OBSERVABLES = (
    "step",
    "time",
    "atom_ids",
    "cartesian_positions",
    "unwrapped_positions",
    "velocities",
    "forces",
    "cell_origin",
)
_EV_JOULES = Fraction("1.602176634e-19")
_AVOGADRO = Fraction("6.02214076e23")
_KCAL_MOL_EV = Fraction(4184) / (_AVOGADRO * _EV_JOULES)


def _fraction(value: Any, name: str) -> Fraction:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        return Fraction(str(value)) if isinstance(value, float) else Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc


def _periodicity(value: Sequence[bool] | None) -> tuple[bool, bool, bool] | None:
    if value is None:
        return None
    try:
        result = tuple(value)
    except TypeError as exc:
        raise ValueError("periodicity must contain exactly three booleans") from exc
    if len(result) != 3 or any(not isinstance(item, bool) for item in result):
        raise ValueError("periodicity must contain exactly three booleans")
    return result[0], result[1], result[2]


def _scale_rows(rows: tuple[tuple[Fraction, ...], ...], factor: Fraction) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(value * factor for value in row) for row in rows)


class LammpsTrajectory(TrajectoryBackend):
    r"""Read a native LAMMPS atom/custom text dump as a bounded stream.

    Atom types are mapped through the required ``species`` mapping. Geometry and
    observables are converted exactly to ångström, picosecond and electronvolt
    units. ``real`` and ``metal`` use their defined LAMMPS units; ``lj`` requires
    explicit length, time and energy scales in those canonical units.

    A non-increasing printed step starts a new segment. If the source contains
    more than one segment, ``segment`` must select one. General triclinic
    ``abc origin`` dumps omit boundary flags, so they require ``periodicity``.

    :param source: Dump path, optionally compressed by a registered codec.
    :param species: Explicit positive atom-type to chemical-symbol mapping.
    :param units: LAMMPS unit style: ``metal``, ``real`` or ``lj``.
    :param coordinates: Coordinate representation: ``auto``, ``x``, ``xs``,
        ``xu`` or ``xsu``.
    :param timestep: Constant source-unit time per printed step, if printed times
        are absent.
    :param segment: Zero-based segment to select, or ``None`` for a single-segment
        source.
    :param periodicity: Explicit periodic axes; required when box flags are absent.
    :param lj_scales: Length (Å), time (ps) and energy (eV) per source unit for
        ``lj``.
    :raises TypeError: If ``source`` is not path-like.
    :raises ValueError: If metadata is invalid or the selected segment violates
        the trajectory contract.
    """

    kind: ClassVar[str] = "lammps"

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt only an existing backend or an explicitly selected LAMMPS path.

        :param obj: Candidate backend or source path.
        :param \**hints: Backend-selection and constructor hints.
        :return: An initialized backend, or ``None`` when this backend declines.
        """
        kind = hints.get("kind")
        if isinstance(obj, cls):
            return obj if kind in (None, cls.kind) else None
        if kind != cls.kind or not isinstance(obj, os.PathLike | str):
            return None
        options = dict(hints)
        options.pop("kind")
        return cls(obj, **options)

    def __init__(
        self,
        source: os.PathLike[str] | str,
        *,
        species: Mapping[int, str],
        units: str,
        coordinates: str = "auto",
        timestep: Fraction | float | Decimal | str | None = None,
        segment: int | None = None,
        periodicity: Sequence[bool] | None = None,
        lj_scales: Sequence[Fraction | float | Decimal | str] | None = None,
    ) -> None:
        if not isinstance(source, os.PathLike | str):
            raise TypeError("LammpsTrajectory source must be path-like")
        if segment is not None and (not isinstance(segment, int) or isinstance(segment, bool) or segment < 0):
            raise ValueError("segment must be a non-negative integer or None")
        if not isinstance(species, Mapping):
            raise TypeError("species must be an atom-type mapping")
        mapped: list[tuple[int, str]] = []
        for atom_type, symbol in species.items():
            if not isinstance(atom_type, int) or isinstance(atom_type, bool) or atom_type <= 0:
                raise ValueError("species keys must be positive integer atom types")
            if not isinstance(symbol, str):
                raise ValueError("species values must be chemical-symbol strings")
            mapped.append((atom_type, symbol))
        if not mapped:
            raise ValueError("species must map at least one atom type")
        mapped.sort()
        requested_periodicity = _periodicity(periodicity)
        scales: tuple[Fraction, Fraction, Fraction]
        if units == "metal":
            scales = (Fraction(1), Fraction(1), Fraction(1))
        elif units == "real":
            scales = (Fraction(1), Fraction(1, 1000), _KCAL_MOL_EV)
        elif units == "lj":
            if lj_scales is None or isinstance(lj_scales, str | bytes) or len(lj_scales) != 3:
                raise ValueError("lj units require three explicit length, time and energy scales")
            scales = (
                _fraction(lj_scales[0], "lj scale"),
                _fraction(lj_scales[1], "lj scale"),
                _fraction(lj_scales[2], "lj scale"),
            )
            if any(value <= 0 for value in scales):
                raise ValueError("lj scales must be strictly positive")
        else:
            raise ValueError("units must be 'metal', 'real' or 'lj'")
        if units != "lj" and lj_scales is not None:
            raise ValueError("lj_scales is only valid with units='lj'")
        step_time = None if timestep is None else _fraction(timestep, "timestep")
        if step_time is not None and step_time <= 0:
            raise ValueError("timestep must be strictly positive")

        self._source = Path(os.fsdecode(os.fspath(source)))
        self._dump = LammpsDumpFile(self._source, coordinates)
        self._species_map = tuple(mapped)
        self._units = units
        self._scales = scales
        self._timestep = step_time
        self._segment = segment
        self._periodicity_override = requested_periodicity

        first = next(self._selected_frames(), None)
        if first is None:
            target = "any segment" if segment is None else f"segment {segment}"
            raise ValueError(f"LAMMPS dump contains no frames in {target}")
        self._atom_ids = first.atom_ids
        self._atom_types = first.atom_types
        by_type = dict(self._species_map)
        missing = sorted(set(first.atom_types) - by_type.keys())
        if missing:
            raise ValueError(f"species mapping is missing LAMMPS atom types {missing!r}")
        self._species_at_sites = tuple(by_type[atom_type] for atom_type in first.atom_types)
        distinct_names = tuple(dict.fromkeys(self._species_at_sites))
        self._species = tuple(Species.from_object(name) for name in distinct_names)
        self._printed_units = first.units
        self._periodicity = self._frame_periodicity(first)
        self._availability = (
            first.time is not None or self._timestep is not None,
            first.unwrapped_positions is not None,
            first.velocities is not None,
            first.forces is not None,
        )
        self._observable_names = tuple(
            name
            for name in _OBSERVABLES
            if name != "time" or self._availability[0]
            if name != "unwrapped_positions" or self._availability[1]
            if name != "velocities" or self._availability[2]
            if name != "forces" or self._availability[3]
        )
        self._validate_frame(first)

    def _selected_frames(self) -> Iterator[LammpsDumpFrame]:
        found = False
        for frame in self._dump.frames():
            if self._segment is None:
                if frame.segment != 0:
                    raise ValueError("LAMMPS dump contains multiple segments; select segment explicitly")
                found = True
                yield frame
            elif frame.segment == self._segment:
                found = True
                yield frame
            elif found and frame.segment > self._segment:
                return

    def _frame_periodicity(self, frame: LammpsDumpFrame) -> tuple[bool, bool, bool]:
        override = self._periodicity_override
        if frame.periodicity is None:
            if override is None:
                raise ValueError("LAMMPS box omits boundary flags; periodicity must be supplied explicitly")
            return override
        parsed = frame.periodicity[0], frame.periodicity[1], frame.periodicity[2]
        if override is not None and parsed != override:
            raise ValueError("explicit periodicity disagrees with LAMMPS box boundary flags")
        return parsed

    def _validate_frame(self, frame: LammpsDumpFrame) -> None:
        if frame.atom_ids != self._atom_ids or frame.atom_types != self._atom_types:
            raise ValueError("LAMMPS atom IDs or types change within the selected segment")
        if frame.units != self._printed_units:
            raise ValueError("LAMMPS printed unit metadata changes within the selected segment")
        if frame.units is not None and frame.units != self._units:
            raise ValueError(f"LAMMPS dump declares units {frame.units!r}, not requested {self._units!r}")
        if self._frame_periodicity(frame) != self._periodicity:
            raise ValueError("LAMMPS periodicity changes within the selected segment")
        availability = (
            frame.time is not None or self._timestep is not None,
            frame.unwrapped_positions is not None,
            frame.velocities is not None,
            frame.forces is not None,
        )
        if availability != self._availability:
            raise ValueError("LAMMPS field availability changes within the selected segment")

    def _structure(self, frame: LammpsDumpFrame) -> UnitcellStructure:
        length, _, _ = self._scales
        cell = Cell(_scale_rows(frame.cell, length), periodicity=self._periodicity)
        relative = tuple(
            tuple(position[axis] - frame.origin[axis] for axis in range(3)) for position in frame.positions
        )
        reduced = SurdVector(relative) * SurdVector(frame.cell).inv()
        return UnitcellStructure(cell, Sites(reduced), self._species, self._species_at_sites)

    def _value(self, frame: LammpsDumpFrame, name: str) -> Any:
        length, time, energy = self._scales
        if name == "step":
            return frame.step
        if name == "time":
            if frame.time is None:
                assert self._timestep is not None
                source_time = self._timestep * frame.step
            else:
                source_time = frame.time
            return source_time * time
        if name == "atom_ids":
            return frame.atom_ids
        if name == "cartesian_positions":
            return _scale_rows(frame.positions, length)
        if name == "unwrapped_positions":
            assert frame.unwrapped_positions is not None
            return _scale_rows(frame.unwrapped_positions, length)
        if name == "velocities":
            assert frame.velocities is not None
            return _scale_rows(frame.velocities, length / time)
        if name == "forces":
            assert frame.forces is not None
            return _scale_rows(frame.forces, energy / length)
        if name == "cell_origin":
            return tuple(value * length for value in frame.origin)
        raise KeyError(name)

    @property
    def species(self) -> tuple[Species, ...]:
        """Return the constant distinct species in first-site order."""
        return self._species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Return constant species names aligned with sorted atom IDs."""
        return self._species_at_sites

    @property
    def observable_names(self) -> tuple[str, ...]:
        """Return available canonical observable names."""
        return self._observable_names

    def frames(self) -> Iterator[UnitcellStructure]:
        """Stream structures from the selected segment.

        :return: A fresh iterator of exact unit-cell structures.
        """
        return (structure for structure, _ in self.samples())

    def observable(self, name: str) -> tuple[Any, ...]:
        """Materialize one canonical observable in frame order.

        :param name: Available observable name.
        :return: Values in frame order.
        :raises KeyError: If the observable is unavailable.
        """
        if name not in self._observable_names:
            raise KeyError(name)
        return tuple(values[0] for _, values in self.samples(name))

    def samples(self, *names: str) -> Iterator[tuple[UnitcellStructure, tuple[Any, ...]]]:
        r"""Stream frames and requested observables together in one pass.

        :param \*names: Observable names in the requested result order.
        :return: A fresh iterator of aligned structure/value pairs.
        :raises KeyError: If an observable is unavailable.
        :raises ValueError: If selected frames violate the trajectory contract.
        """
        for name in names:
            if name not in self._observable_names:
                raise KeyError(name)

        def generate() -> Iterator[tuple[UnitcellStructure, tuple[Any, ...]]]:
            for frame in self._selected_frames():
                self._validate_frame(frame)
                yield self._structure(frame), tuple(self._value(frame, name) for name in names)

        return generate()

    def unwrap(self) -> Path:
        """Return the normalized dump path."""
        return self._source

    @property
    def source_locator(self) -> str:
        """Return the dump path as a storage locator."""
        return os.fsdecode(os.fspath(self._source))
