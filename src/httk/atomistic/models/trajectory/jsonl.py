"""Lazy atomistic adapter for the neutral trajectory JSONL payload."""

import os
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

import httk.core

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.trajectory.backend import TrajectoryBackend


class JsonlTrajectory(TrajectoryBackend):
    """A lazy trajectory backed by a neutral ``httk-trajectory-jsonl`` payload or path."""

    kind: ClassVar[str] = "jsonl"

    def __new__(cls, source: Any, **hints: Any) -> Any:
        if hints.get("kind", cls.kind) != cls.kind:
            return None
        if isinstance(source, cls):
            return source
        if isinstance(source, Mapping) and source.get("format") not in (None, "httk-trajectory-jsonl"):
            return None
        if not isinstance(source, (Mapping, str, os.PathLike)):
            return None
        return super().__new__(cls)

    def __init__(self, source: Any, **hints: Any) -> None:
        if getattr(self, "_jsonl_initialized", False):
            return
        self._source = source
        if isinstance(source, Mapping):
            if source.get("format") != "httk-trajectory-jsonl":
                raise ValueError("JsonlTrajectory payload must have format 'httk-trajectory-jsonl'.")
            self._file = source["trajectory_jsonl"]
        elif isinstance(source, (str, os.PathLike)):
            payload = httk.core.load(os.fsdecode(os.fspath(source)), raw=True)
            if payload.get("format") != "httk-trajectory-jsonl":
                raise ValueError("JsonlTrajectory path did not load as a trajectory JSONL payload.")
            self._file = payload["trajectory_jsonl"]
        else:
            raise TypeError("JsonlTrajectory expects a trajectory JSONL path or payload")
        self._jsonl_initialized = True

    @property
    def _info(self) -> Mapping[str, Any]:
        return self._file.header["x-httk-trajectory"]

    @property
    def nframes(self) -> int:
        return self._file.nframes

    @property
    def header(self) -> Mapping[str, Any]:
        return self._file.header

    @property
    def species(self) -> tuple[Species, ...]:
        return tuple(Species.create(value) for value in self._info["species"])

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return tuple(self._info["species_at_sites"])

    @property
    def reference_frames(self) -> tuple[int, ...] | None:
        references = self._info["reference_frames"]
        return None if references is None else tuple(references)

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(self._info["observable_names"])

    def observable(self, name: str) -> tuple[Any, ...]:
        if name not in self.observable_names:
            raise KeyError(name)
        return tuple(frame["observables"][name] for frame in self._file.frames())

    def _structure(self, frame: Mapping[str, Any]) -> UnitcellStructure:
        cell = frame.get("lattice_vectors", self._info.get("constant_cell"))
        if cell is None:
            raise ValueError("trajectory JSONL frame has no cell")
        return UnitcellStructure(
            Cell(cell),
            Sites(frame["fractional_site_positions"]),
            self.species,
            self.species_at_sites,
        )

    def frame(self, i: int) -> UnitcellStructure:
        return self._structure(self._file.frame(i))

    def frames(self) -> Iterator[UnitcellStructure]:
        yield from (self._structure(frame) for frame in self._file.frames())

    def unwrap(self) -> Any:
        return self._source

    @property
    def source_locator(self) -> str | None:
        path = getattr(self._file, "path", None)
        if isinstance(path, str):
            return path
        return os.fsdecode(os.fspath(self._source)) if isinstance(self._source, os.PathLike | str) else None
