"""Streaming LAMMPS trajectory backend checks."""

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import httk.core
import pytest

from httk.atomistic import Cell, Sites, Species, Trajectory, TrajectoryView, UnitcellStructure
from httk.atomistic.integrations.lammps import LammpsTrajectory
from httk.atomistic.integrations.lammps.io import LammpsDumpFile
from httk.atomistic.models.trajectory.backend import TrajectoryBackend


def _dump_frame(
    step: int,
    *,
    box: str = "ITEM: BOX BOUNDS pp ff pp\n-1 3\n2 6\n0 8",
    columns: str = "id type xs ys zs ix iy iz vx vy vz fx fy fz",
    atoms: str = "2 2 1/2 1/4 3/4 -1 0 1 1 2 3 4 5 6\n1 1 0 0 0 0 0 0 -1 -2 -3 -4 -5 -6",
    time: str | None = None,
    units: str | None = None,
) -> str:
    prefix = "" if units is None else f"ITEM: UNITS\n{units}\n"
    prefix += "" if time is None else f"ITEM: TIME\n{time}\n"
    return f"{prefix}ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n2\n{box}\nITEM: ATOMS {columns}\n{atoms}\n"


def _write(path: Path, *frames: str) -> Path:
    path.write_text("".join(frames), encoding="utf-8")
    return path


def test_metal_trajectory_builds_exact_origin_relative_structures_and_observables(tmp_path: Path) -> None:
    source = _write(tmp_path / "atoms.dump", _dump_frame(10, units="metal"), _dump_frame(20))
    trajectory = LammpsTrajectory(source, species={1: "Si", 2: "O"}, units="metal", timestep="0.002")

    assert trajectory.species == (Species.from_object("Si"), Species.from_object("O"))
    assert trajectory.species_at_sites == ("Si", "O")
    assert trajectory.observable_names == (
        "step",
        "time",
        "atom_ids",
        "cartesian_positions",
        "unwrapped_positions",
        "velocities",
        "forces",
        "cell_origin",
    )
    first = trajectory.frame(0)
    assert first.cell.periodicity == (True, False, True)
    assert first.cell.basis.to_floats() == [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 8.0]]
    assert first.sites.reduced_coords.to_floats() == [[0.0, 0.0, 0.0], [0.5, 0.25, 0.75]]
    assert trajectory.observable("step") == (10, 20)
    assert trajectory.observable("time") == (Fraction(1, 50), Fraction(1, 25))
    assert trajectory.observable("atom_ids") == ((1, 2), (1, 2))
    assert trajectory.observable("cell_origin")[0] == (Fraction(-1), Fraction(2), Fraction(0))
    assert trajectory.observable("cartesian_positions")[0][1] == (Fraction(1), Fraction(3), Fraction(6))
    assert trajectory.observable("unwrapped_positions")[0][1] == (Fraction(-3), Fraction(3), Fraction(14))
    assert trajectory.unwrap() == source
    assert trajectory.source_locator == str(source)


def test_real_and_lj_units_convert_vectors_exactly(tmp_path: Path) -> None:
    source = _write(tmp_path / "units.dump", _dump_frame(2))
    real = LammpsTrajectory(source, species={1: "Si", 2: "O"}, units="real", timestep=2)
    kcal_mol_ev = Fraction(4184) / (Fraction("6.02214076e23") * Fraction("1.602176634e-19"))
    assert real.observable("time") == (Fraction(1, 250),)
    assert real.observable("velocities")[0][0] == (Fraction(-1000), Fraction(-2000), Fraction(-3000))
    assert real.observable("forces")[0][1] == (
        4 * kcal_mol_ev,
        5 * kcal_mol_ev,
        6 * kcal_mol_ev,
    )

    lj = LammpsTrajectory(
        source,
        species={1: "Si", 2: "O"},
        units="lj",
        timestep="1/2",
        lj_scales=("2", "3", "5"),
    )
    assert lj.observable("time") == (Fraction(3),)
    assert lj.observable("cell_origin")[0] == (Fraction(-2), Fraction(4), Fraction(0))
    assert lj.observable("velocities")[0][1] == (Fraction(2, 3), Fraction(4, 3), Fraction(2))
    assert lj.observable("forces")[0][1] == (Fraction(10), Fraction(25, 2), Fraction(15))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"units": "si"}, "units must be"),
        ({"units": "lj"}, "three explicit"),
        ({"units": "lj", "lj_scales": (1, 2)}, "three explicit"),
        ({"units": "lj", "lj_scales": (1, 0, 2)}, "strictly positive"),
        ({"units": "metal", "lj_scales": (1, 1, 1)}, "only valid"),
        ({"units": "metal", "timestep": 0}, "strictly positive"),
        ({"units": "metal", "segment": -1}, "non-negative"),
    ],
)
def test_unit_scale_and_selection_errors(tmp_path: Path, kwargs: dict[str, Any], message: str) -> None:
    source = _write(tmp_path / "bad-metadata.dump", _dump_frame(0))
    with pytest.raises(ValueError, match=message):
        LammpsTrajectory(source, species={1: "Si", 2: "O"}, **kwargs)


def test_printed_units_must_match_and_printed_times_take_precedence(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "time.dump",
        _dump_frame(0, units="real", time="0"),
        _dump_frame(4, time="0.003"),
        _dump_frame(9, time="0.01"),
    )
    trajectory = LammpsTrajectory(source, species={1: "Si", 2: "O"}, units="real", timestep=100)
    assert trajectory.observable("time") == (Fraction(0), Fraction(3, 1_000_000), Fraction(1, 100_000))
    with pytest.raises(ValueError, match="not requested"):
        LammpsTrajectory(source, species={1: "Si", 2: "O"}, units="metal")


def test_general_box_requires_explicit_periodicity(tmp_path: Path) -> None:
    general = "ITEM: BOX BOUNDS abc origin\n2 0 0 -1\n1 3 0 4\n0 1 5 -2"
    source = _write(tmp_path / "general.dump", _dump_frame(0, box=general))
    with pytest.raises(ValueError, match="periodicity must be supplied"):
        LammpsTrajectory(source, species={1: "Si", 2: "O"}, units="metal")
    trajectory = LammpsTrajectory(
        source,
        species={1: "Si", 2: "O"},
        units="metal",
        periodicity=(True, False, True),
    )
    assert trajectory.frame(0).cell.periodicity == (True, False, True)


def test_species_are_explicit_and_type_mapping_is_complete(tmp_path: Path) -> None:
    source = _write(tmp_path / "species.dump", _dump_frame(0))
    with pytest.raises(ValueError, match="missing LAMMPS atom types"):
        LammpsTrajectory(source, species={1: "Si"}, units="metal")
    with pytest.raises(ValueError, match="chemical symbol"):
        LammpsTrajectory(source, species={1: "Si", 2: "NotAnElement"}, units="metal")
    with pytest.raises(ValueError, match="positive integer"):
        LammpsTrajectory(source, species={0: "Si", 2: "O"}, units="metal")


def test_segments_require_selection_and_selected_segment_streams_alone(tmp_path: Path) -> None:
    source = _write(tmp_path / "restart.dump", _dump_frame(0), _dump_frame(1), _dump_frame(0), _dump_frame(2))
    implicit = LammpsTrajectory(source, species={1: "Si", 2: "O"}, units="metal")
    with pytest.raises(ValueError, match="multiple segments"):
        tuple(implicit.frames())
    selected = LammpsTrajectory(source, species={1: "Si", 2: "O"}, units="metal", segment=1)
    assert selected.observable("step") == (0, 2)
    with pytest.raises(ValueError, match="no frames in segment 2"):
        LammpsTrajectory(source, species={1: "Si", 2: "O"}, units="metal", segment=2)


def test_selected_segment_contract_cannot_change(tmp_path: Path) -> None:
    without_vectors = _dump_frame(1, columns="id type xs ys zs", atoms="1 1 0 0 0\n2 2 1/2 1/4 3/4")
    source = _write(tmp_path / "fields.dump", _dump_frame(0), without_vectors)
    trajectory = LammpsTrajectory(source, species={1: "Si", 2: "O"}, units="metal")
    with pytest.raises(ValueError, match="field availability changes"):
        tuple(trajectory.frames())


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        ((_dump_frame(0, units="metal"), _dump_frame(1, units="real")), "unit metadata changes"),
        (
            (
                _dump_frame(0),
                _dump_frame(1, box="ITEM: BOX BOUNDS pp pp pp\n-1 3\n2 6\n0 8"),
            ),
            "periodicity changes",
        ),
        ((_dump_frame(0, time="0"), _dump_frame(1)), "field availability changes"),
    ],
)
def test_unit_periodicity_and_time_availability_are_segment_contracts(
    tmp_path: Path, frames: tuple[str, ...], message: str
) -> None:
    source = _write(tmp_path / "changing-contract.dump", *frames)
    trajectory = LammpsTrajectory(source, species={1: "Si", 2: "O"}, units="metal")
    with pytest.raises(ValueError, match=message):
        tuple(trajectory.frames())


def test_samples_stream_once_preserve_order_and_view_delegates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _write(tmp_path / "samples.dump", _dump_frame(0), _dump_frame(1))
    trajectory = LammpsTrajectory(source, species={1: "Si", 2: "O"}, units="metal")
    traversals = 0
    original = LammpsDumpFile.frames

    def counted(dump: LammpsDumpFile) -> Any:
        nonlocal traversals
        traversals += 1
        return original(dump)

    monkeypatch.setattr(LammpsDumpFile, "frames", counted)
    samples = tuple(TrajectoryView(trajectory).samples("atom_ids", "step", "atom_ids"))
    assert traversals == 1
    assert [values for _, values in samples] == [((1, 2), 0, (1, 2)), ((1, 2), 1, (1, 2))]
    with pytest.raises(KeyError):
        tuple(trajectory.samples("missing"))


def test_backend_adoption_is_explicit_only(tmp_path: Path) -> None:
    source = _write(tmp_path / "adopt.dump", _dump_frame(0))
    assert LammpsTrajectory._backend_adopt(source) is None
    adopted = LammpsTrajectory._backend_adopt(source, kind="lammps", species={1: "Si", 2: "O"}, units="metal")
    assert isinstance(adopted, LammpsTrajectory)
    assert LammpsTrajectory._backend_adopt(adopted) is adopted
    assert LammpsTrajectory._backend_adopt(adopted, kind="other") is None


def test_registered_load_keeps_raw_payload_and_requires_explicit_metadata(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "registered.lammpstrj",
        _dump_frame(
            0,
            atoms="2 1 1/2 1/4 3/4 -1 0 1 1 2 3 4 5 6\n1 1 0 0 0 0 0 0 -1 -2 -3 -4 -5 -6",
        ),
    )
    raw = httk.core.load(source, raw=True)
    assert raw == {"format": "lammps-dump", "source": source, "options": {}}
    with pytest.raises(TypeError, match="species.*units|units.*species"):
        httk.core.load(source)
    loaded = httk.core.load(source, species={1: "Ar"}, units="metal", segment=0)
    assert isinstance(loaded, LammpsTrajectory)
    assert loaded.observable("step") == (0,)
    selected = httk.core.load_source(
        source,
        "explicit.lammpsdump",
        species={1: "Ar"},
        units="metal",
        segment=0,
    )
    assert isinstance(selected, LammpsTrajectory)
    assert selected.species_at_sites == ("Ar", "Ar")


def test_base_samples_materializes_selected_observables_and_checks_alignment() -> None:
    frame = UnitcellStructure(
        Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        Sites([[0, 0, 0]]),
        [Species.from_object("Si")],
        ["Si"],
    )
    trajectory = Trajectory([frame, frame], {"energy": [1, 2]})
    assert list(trajectory.samples("energy")) == [(frame, (1,)), (frame, (2,))]

    class Misaligned(TrajectoryBackend):
        def __init__(self) -> None:
            pass

        def frames(self) -> Iterator[UnitcellStructure]:
            return iter((frame, frame))

        @property
        def species(self) -> tuple[Species, ...]:
            return frame.species

        @property
        def species_at_sites(self) -> tuple[str, ...]:
            return frame.species_at_sites

        @property
        def observable_names(self) -> tuple[str, ...]:
            return ("short",)

        def observable(self, name: str) -> tuple[Any, ...]:
            if name != "short":
                raise KeyError(name)
            return (1,)

    with pytest.raises(ValueError, match="shorter"):
        tuple(Misaligned().samples("short"))
