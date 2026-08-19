"""Trajectory family and OPTIMADE frame-mapping checks."""

from pathlib import Path

import pytest
from httk.core import load_entry_type_definition

from httk.atomistic import (
    Cell,
    PlainTrajectory,
    Sites,
    Species,
    Trajectory,
    TrajectoryView,
    UnitcellStructure,
)


def _frame(x: int = 0) -> UnitcellStructure:
    return UnitcellStructure(
        Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        Sites([[x, 0, 0]]),
        [Species.from_object("Si")],
        ["Si"],
    )


def _plain(**extra: object) -> dict[str, object]:
    result: dict[str, object] = {
        "nframes": 2,
        "lattice_vectors": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        "fractional_site_positions": [[[0, 0, 0]], [[1, 0, 0]]],
        "species_at_sites": [["Si"]],
        "species": [[{"name": "Si", "chemical_symbols": ["Si"], "concentration": [1]}]],
    }
    result.update(extra)
    return result


def test_native_trajectory_coerces_frames_and_delegates() -> None:
    trajectory = Trajectory([_frame(), _frame(1)], {"energy": [1, 2]}, [1, 0, 1])
    assert trajectory.nframes == 2
    assert trajectory.frame(-1).sites[0][0] == 1
    assert list(trajectory.frames()) == [trajectory.frame(0), trajectory.frame(1)]
    assert trajectory.reference_frames == (0, 1)
    assert trajectory.observable_names == ("energy",)
    assert trajectory.observable("energy") == (1, 2)
    with pytest.raises(KeyError):
        trajectory.observable("missing")
    view = TrajectoryView(trajectory)
    assert view.frame(0) is trajectory.frame(0)
    assert view.observable("energy") == (1, 2)
    assert view.unwrap() is trajectory


def test_native_trajectory_validates_shape_and_composition() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Trajectory([])
    with pytest.raises(ValueError, match="observable 'energy'"):
        Trajectory([_frame()], {"energy": []})
    with pytest.raises(ValueError, match="reference frame 2"):
        Trajectory([_frame()], reference_frames=[2])
    with pytest.raises(ValueError, match="varying composition"):
        Trajectory(
            [
                _frame(),
                UnitcellStructure(
                    Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), Sites([[0, 0, 0]]), [Species.from_object("O")], ["O"]
                ),
            ]
        )


def test_plain_trajectory_supports_compact_properties_and_identity_unwrap() -> None:
    raw = _plain(energy=[3, 4])
    trajectory = PlainTrajectory(raw)
    assert isinstance(trajectory, PlainTrajectory)
    assert trajectory.frame(-1).sites[0][0] == 1
    assert trajectory.species_at_sites == ("Si",)
    assert trajectory.observable("energy") == (3, 4)
    assert trajectory.unwrap() is raw
    assert trajectory.reference_frames is None


def test_plain_trajectory_rejects_axis_and_reference_errors() -> None:
    with pytest.raises(ValueError, match="energy.*length 1"):
        PlainTrajectory(_plain(energy=[1]))
    with pytest.raises(ValueError, match="energy.*frame axis"):
        PlainTrajectory(_plain(energy=1))
    with pytest.raises(ValueError, match="reference frame 2"):
        PlainTrajectory(_plain(reference_frames=[2]))


def test_plain_trajectory_requires_constant_composition() -> None:
    with pytest.raises(ValueError, match="species_at_sites.*varies"):
        PlainTrajectory(_plain(species_at_sites=[["Si"], ["O"]]))


def test_vendored_trajectory_definition_when_fetched() -> None:
    path = Path(__file__).parents[1] / "src/httk/registry/schemas/atomistic/trajectories.json"
    if not path.exists():
        pytest.skip("OPTIMADE trajectories definition was not fetched")
    definition = load_entry_type_definition("https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/trajectories")
    assert definition.definition_id == "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/trajectories"
