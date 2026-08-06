"""httk-serve integration for the trajectory entry provider."""

import pytest

pytest.importorskip("httk.serve.optimade")

from httk.serve.optimade import adapter_from_providers
from httk.serve.optimade.backend import execute_query
from httk.serve.optimade.filter import parse_optimade_filter

from httk.atomistic import Cell, Sites, Species, Trajectory, TrajectoryEntryProvider, UnitcellStructure


def _trajectory(nframes: int) -> Trajectory:
    frames = tuple(
        UnitcellStructure(
            Cell([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
            Sites([[index, 0, 0]]),
            [Species.create("Si")],
            ["Si"],
        )
        for index in range(nframes)
    )
    return Trajectory(frames)


def test_adapter_from_providers_filters_trajectory_nframes() -> None:
    adapter = adapter_from_providers(
        [TrajectoryEntryProvider({"short": _trajectory(2), "long": _trajectory(3)})]
    )
    results = list(
        execute_query(
            adapter,
            ["trajectories"],
            ["id", "type", "nframes"],
            [],
            100,
            0,
            parse_optimade_filter("nframes >= 3"),
        )
    )
    assert [result.values["id"] for result in results] == ["long"]
    assert results[0].values["nframes"] == 3
