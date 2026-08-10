"""Expose bounded trajectory storage records as backends."""

from collections.abc import Iterator
from functools import cached_property
from typing import Any, Self, cast

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
    r"""Expose only the bounded summary and stored reference frames.

    Full frame data is deliberately absent from
    :class:`httk.atomistic.storage.records.TrajectoryRecord`.
    Callers must use ``source_locator`` to reopen the original source for any
    frame that is not one of the stored reference frames.

    :param obj: The bounded trajectory storage record.
    :param \**hints: Backend-selection hints.
    """

    _record: TrajectoryRecord

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a trajectory record.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, TrajectoryRecord):
            return None
        return cls(obj, **hints)

    def __init__(self, obj: TrajectoryRecord, **hints: Any) -> None:
        self._record = obj

    @property
    def nframes(self) -> int:
        """Return the total number of source frames."""
        return self._record.nframes

    @cached_property
    def species(self) -> tuple[Species, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return the stored distinct species."""
        return tuple(SpeciesView(cast(Any, value), kind="record") for value in self._record.species)

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        """Return the stored species name at each site."""
        return self._record.species_at_sites

    @property
    def reference_frames(self) -> tuple[int, ...]:
        """Return the indexes of frames stored in the bounded record."""
        return self._record.reference_frame_indexes

    @property
    def observable_names(self) -> tuple[str, ...]:
        """Return the names of summarized observables."""
        return tuple(value.name for value in self._record.observable_summaries)

    @property
    def observable_summaries(self) -> tuple[ObservableSummaryRecord, ...]:
        """Return the stored per-observable summaries."""
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
        """Return a stored reference frame by source index.

        :param i: Source frame index.
        :return: The stored reference frame.
        :raises IndexError: If the frame is not stored in the bounded record.
        :raises TypeError: If the index is not an integer.
        """
        return UnitcellStructureView(self._record.reference_frame_structures[self._stored_index(i)])

    def frames(self) -> Iterator[UnitcellStructure]:
        """Reject iteration because full frames are not stored.

        :return: Never; reopen the source at ``source_locator`` instead.
        :raises RuntimeError: Always, because the record stores no full frame sequence.
        """
        raise RuntimeError(
            "Trajectory frame data is not stored in the record; use "
            f"source_locator={self._record.source_locator!r} to recover the full trajectory"
        )

    def observable(self, name: str) -> tuple[Any, ...]:
        """Reject per-frame access to summarized observable values.

        :param name: Observable name.
        :return: Never; reopen the source at ``source_locator`` instead.
        :raises KeyError: If the observable is unavailable.
        :raises RuntimeError: If the observable is summarized but not stored per frame.
        """
        if name not in self.observable_names:
            raise KeyError(name)
        raise RuntimeError(
            "Trajectory observable values are summarized but not stored per frame; use "
            f"source_locator={self._record.source_locator!r} to recover them"
        )

    def unwrap(self) -> TrajectoryRecord:
        """Return the bounded storage record."""
        return self._record
