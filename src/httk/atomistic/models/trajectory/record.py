"""Trajectory backend for bounded trajectory storage records."""

from collections.abc import Iterator
from functools import cached_property
from typing import Any, cast

from httk.atomistic.models.species.species import Species
from httk.atomistic.models.species.view import SpeciesView
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.models.trajectory.backend import TrajectoryBackend
from httk.atomistic.storage.records import (
    ObservableSummaryRecord,
    TrajectoryRecord,
)


class RecordTrajectory(TrajectoryBackend):
    """Expose only the bounded summary and stored reference frames.

    Full frame data is deliberately absent from
    :class:`httk.atomistic.storage.records.TrajectoryRecord`.
    Callers must use ``source_locator`` to reopen the original source for any
    frame that is not one of the stored reference frames.
    """

    _record: TrajectoryRecord

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, TrajectoryRecord):
            return None
        return super().__new__(cls)

    def __init__(self, obj: TrajectoryRecord, **hints: Any) -> None:
        self._record = obj

    @property
    def nframes(self) -> int:
        return self._record.nframes

    @cached_property
    def species(self) -> tuple[Species, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return tuple(SpeciesView(cast(Any, value), kind="record") for value in self._record.species)

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return self._record.species_at_sites

    @property
    def reference_frames(self) -> tuple[int, ...]:
        return self._record.reference_frame_indexes

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(value.name for value in self._record.observable_summaries)

    @property
    def observable_summaries(self) -> tuple[ObservableSummaryRecord, ...]:
        return self._record.observable_summaries

    def _stored_index(self, i: int) -> int:
        if not isinstance(i, int):
            raise TypeError("Trajectory frame index must be an integer")
        normalized = i + self.nframes if i < 0 else i
        if not 0 <= normalized < self.nframes:
            raise IndexError(
                f"Trajectory frame index {i} out of range; frame data is not stored, "
                f"use source_locator={self._record.source_locator!r} to recover it"
            )
        if normalized not in self.reference_frames:
            raise IndexError(
                f"Trajectory frame {normalized} is not stored in the bounded record; "
                f"use source_locator={self._record.source_locator!r} to recover it"
            )
        return self.reference_frames.index(normalized)

    def frame(self, i: int) -> UnitcellStructure:
        return UnitcellStructureView(self._record.reference_frame_structures[self._stored_index(i)])

    def frames(self) -> Iterator[UnitcellStructure]:
        raise RuntimeError(
            "Trajectory frame data is not stored in the record; use "
            f"source_locator={self._record.source_locator!r} to recover the full trajectory"
        )

    def observable(self, name: str) -> tuple[Any, ...]:
        if name not in self.observable_names:
            raise KeyError(name)
        raise RuntimeError(
            "Trajectory observable values are summarized but not stored per frame; use "
            f"source_locator={self._record.source_locator!r} to recover them"
        )

    def unwrap(self) -> TrajectoryRecord:
        return self._record
