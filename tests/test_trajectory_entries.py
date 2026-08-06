"""OPTIMADE trajectories entry-family and provider checks."""

import datetime

import pytest
from httk.core.storage import project_storage_record

from httk.atomistic import (
    Cell,
    RecordTrajectory,
    Sites,
    Species,
    Trajectory,
    TrajectoryEntry,
    TrajectoryEntryProvider,
    TrajectoryRecord,
    UnitcellStructure,
)
from httk.atomistic.entries.trajectories import TRAJECTORY_FRAME_MATERIALIZATION_LIMIT


def _frame(index: int) -> UnitcellStructure:
    return UnitcellStructure(
        Cell([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        Sites([[index, 0, 0]]),
        [Species.create("Si")],
        ["Si"],
    )


def _trajectory(nframes: int = 3) -> Trajectory:
    return Trajectory(
        [_frame(index) for index in range(nframes)],
        {
            "_httk_frame_total_energies": tuple(float(index) for index in range(nframes)),
            "_httk_frame_temperatures": tuple(300.0 + index for index in range(nframes)),
        },
        reference_frames=(0, nframes - 1),
    )


def _served(provider: TrajectoryEntryProvider) -> dict[str, object]:
    row = next(iter(provider.records("trajectories")))
    return {name: row[key] for name, key in provider.property_keys("trajectories").items()}


def test_trajectory_entry_provider_emits_compact_and_frame_wrapped_values() -> None:
    provider = TrajectoryEntryProvider(
        {"md": _trajectory()},
        properties={"md": {"_httk_time_step": 1.0}},
    )
    row = _served(provider)

    assert row["type"] == "trajectories"
    assert row["nframes"] == 3
    assert row["reference_frames"] == [0, 2]
    assert row["lattice_vectors"] == [[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]]
    assert row["species_at_sites"] == [["Si"]]
    assert row["species"] == [[{"name": "Si", "chemical_symbols": ["Si"], "concentration": [1.0]}]]
    assert row["fractional_site_positions"] == [
        [[0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
        [[2.0, 0.0, 0.0]],
    ]
    assert row["_httk_frame_total_energies"] == [0.0, 1.0, 2.0]
    assert row["_httk_frame_temperatures"] == [300.0, 301.0, 302.0]
    assert row["_httk_time_step"] == 1.0


def test_trajectory_entry_provider_does_not_materialize_over_bound_frames() -> None:
    provider = TrajectoryEntryProvider({"large": _trajectory(TRAJECTORY_FRAME_MATERIALIZATION_LIMIT + 1)})
    row = _served(provider)
    assert row["nframes"] == TRAJECTORY_FRAME_MATERIALIZATION_LIMIT + 1
    assert row["fractional_site_positions"] is None
    assert row["_httk_frame_total_energies"] is None


def test_record_trajectory_serves_summary_without_frame_lists() -> None:
    native = _trajectory()
    record = TrajectoryRecord(**project_storage_record(TrajectoryRecord, native))
    provider = TrajectoryEntryProvider({"recorded": RecordTrajectory(record)})
    row = _served(provider)
    assert row["nframes"] == 3
    assert row["reference_frames"] == [0, 2]
    assert row["fractional_site_positions"] is None
    assert row["_httk_frame_total_energies"] is None


def test_record_trajectory_serves_identity_metadata() -> None:
    native = _trajectory()
    values = dict(project_storage_record(TrajectoryRecord, native))
    values.update(
        immutable_id="immutable-md",
        last_modified=datetime.datetime(2026, 1, 2, tzinfo=datetime.UTC),
    )
    record = TrajectoryRecord(**values)
    row = _served(TrajectoryEntryProvider({"recorded": RecordTrajectory(record)}))
    assert row["immutable_id"] == "immutable-md"
    assert row["last_modified"] == "2026-01-02T00:00:00+00:00"


def test_trajectory_provider_rows_validate_against_extended_definition() -> None:
    pytest.importorskip("httk.data")
    from httk.data.validation import validate_record

    provider = TrajectoryEntryProvider({"md": _trajectory()}, properties={"md": {"_httk_time_step": 1.0}})
    validate_record(provider.entry_types()["trajectories"], _served(provider))


def test_trajectory_entry_family_and_shadow_guards() -> None:
    with pytest.raises(TypeError, match="logical entry family"):
        TrajectoryEntry()
    with pytest.raises(ValueError, match="standard OPTIMADE trajectory"):
        TrajectoryEntryProvider({"md": _trajectory()}, properties={"md": {"nframes": 3}})
