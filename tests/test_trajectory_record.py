"""Bounded trajectory storage-record tests."""

from dataclasses import fields

import pytest
from httk.core.storage import content_id, project_storage_record

from httk.atomistic import (
    Cell,
    RecordTrajectory,
    Sites,
    Species,
    Trajectory,
    TrajectoryRecord,
    TrajectoryView,
    UnitcellStructure,
)
from httk.atomistic.models.trajectory.record import ObservableSummaryRecord


def _frame(x: int) -> UnitcellStructure:
    return UnitcellStructure(
        Cell([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        Sites([[x, 0, 0], [0, x, 0]]),
        [Species.from_object("Na"), Species.from_object("Cl")],
        ["Na", "Cl"],
    )


def _trajectory(reference_frames: list[int] | None = None) -> Trajectory:
    return Trajectory(
        [_frame(0), _frame(1), _frame(2)],
        {"energy": (3, -2, 4), "label": ("start", "middle", "end")},
        reference_frames=reference_frames,
    )


def _record(trajectory: Trajectory, **metadata: object) -> TrajectoryRecord:
    values = dict(project_storage_record(TrajectoryRecord, trajectory))
    values.update(metadata)
    return TrajectoryRecord(**values)


def test_projection_keeps_only_bounded_reference_and_observable_data() -> None:
    record = _record(_trajectory([1]), source_locator="runs/md.extxyz")
    assert record.nframes == 3
    assert record.reference_frame_indexes == (1,)
    assert len(record.reference_frame_structures) == 1
    assert record.observable_summaries == (
        ObservableSummaryRecord("energy", 3.0, 4.0, -2.0, 4.0),
        ObservableSummaryRecord("label"),
    )
    assert "frames" not in {value.name for value in fields(record)}
    assert record.type == "trajectories"
    assert record.id is None
    assert record.immutable_id is None


def test_projection_defaults_to_first_and_last_reference_frames() -> None:
    record = _record(_trajectory())
    assert record.reference_frame_indexes == (0, 2)
    assert len(record.reference_frame_structures) == 2


def test_projection_accepts_trajectory_views() -> None:
    record = TrajectoryRecord(**project_storage_record(TrajectoryRecord, TrajectoryView(_trajectory([1]))))
    assert record.reference_frame_indexes == (1,)


def test_record_trajectory_round_trip_exposes_only_stored_frames() -> None:
    record = _record(_trajectory([1]), source_locator="runs/md.extxyz")
    backend = RecordTrajectory(record)
    view = TrajectoryView(record)
    assert backend.nframes == 3
    assert backend.species_at_sites == ("Na", "Cl")
    assert backend.observable_names == ("energy", "label")
    assert backend.observable_summaries[0].minimum == -2.0
    assert backend.frame(1).species_at_sites == ("Na", "Cl")
    assert view.frame(1).species_at_sites == ("Na", "Cl")
    assert view.unwrap() is record
    with pytest.raises(IndexError, match="not stored.*source_locator"):
        backend.frame(0)
    with pytest.raises(RuntimeError, match="not stored.*source_locator"):
        backend.frames()


def test_record_frame_out_of_range_explains_source_recovery() -> None:
    backend = RecordTrajectory(_record(_trajectory([1]), source_locator="runs/md.extxyz"))
    with pytest.raises(IndexError, match="out of range.*source_locator"):
        backend.frame(8)


def test_locator_and_metadata_are_identity_skipped() -> None:
    source = _trajectory([1])
    first = _record(source, source_locator="first.extxyz", id="logical-a", immutable_id="remote-a")
    second = _record(source, source_locator="second.extxyz", id="logical-b", immutable_id="remote-b")
    assert first == second
    assert content_id(first) == content_id(second)
    assert first.id == "logical-a"
    assert second.id == "logical-b"


def test_observable_summary_order_does_not_change_content_id() -> None:
    frames = tuple(_trajectory([1]).frames())
    first = Trajectory(frames, {"z": (1, 2, 3), "a": (3, 2, 1)}, reference_frames=[1])
    second = Trajectory(frames, {"a": (3, 2, 1), "z": (1, 2, 3)}, reference_frames=[1])
    assert content_id(TrajectoryRecord(**project_storage_record(TrajectoryRecord, first))) == content_id(
        TrajectoryRecord(**project_storage_record(TrajectoryRecord, second))
    )


def test_trajectory_record_content_id_pin() -> None:
    # A changed value means a deliberate storage-identity break: update this pin only with migration intent.
    assert content_id(_record(_trajectory([1]))) == "5974ac56bcb370de5ca3e131b0dca28af753a0661151b45060f92f9e8986902c"
