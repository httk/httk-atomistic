"""Atomistic trajectory JSONL round trips."""

from pathlib import Path

import httk.core
import pytest

from httk.atomistic import Cell, JsonlTrajectory, Sites, Species, Trajectory, UnitcellStructure


def _structure(x: float) -> UnitcellStructure:
    return UnitcellStructure(
        Cell([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        Sites([[x, 0, 0]]),
        [Species.create("Al")],
        ["Al"],
    )


def test_native_trajectory_jsonl_round_trip(tmp_path: Path) -> None:
    source = Trajectory([_structure(0), _structure(0.25), _structure(0.5)], {"energy": [1, 2, 3]}, [0, 2])
    path = tmp_path / "native.traj.jsonl.gz"
    httk.core.save(source, path)
    loaded = httk.core.load(path)
    assert isinstance(loaded, JsonlTrajectory)
    assert loaded.nframes == source.nframes
    assert loaded.reference_frames == (0, 2)
    assert loaded.observable("energy") == pytest.approx((1, 2, 3))
    for expected, actual in zip(source.frames(), loaded.frames()):
        for actual_row, expected_row in zip(
            actual.sites.reduced_coords.to_floats(), expected.sites.reduced_coords.to_floats()
        ):
            assert actual_row == pytest.approx(expected_row)
    assert loaded.unwrap()["format"] == "httk-trajectory-jsonl"
