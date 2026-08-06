"""VASP trajectory to JSONL conversion."""

from pathlib import Path
from time import perf_counter

import httk.core
import pytest
from httk.core.storage import project_storage_record

from httk.atomistic import JsonlTrajectory, TrajectoryRecord, VASPTrajectory

POSCAR = """Synthetic POSCAR
1.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
Si
1
Direct
0.0 0.0 0.0
"""

XDATCAR = """Synthetic XDATCAR
1.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
Si
1
Direct configuration= 1
0.1 0.2 0.3
Direct configuration= 2
0.2 0.3 0.4
Direct configuration= 3
0.3 0.4 0.5
"""


def test_vasp_trajectory_jsonl_round_trip(tmp_path: Path) -> None:
    source_dir = tmp_path / "vasp"
    source_dir.mkdir()
    (source_dir / "POSCAR").write_text(POSCAR, encoding="utf-8")
    (source_dir / "XDATCAR").write_text(XDATCAR, encoding="utf-8")
    source = VASPTrajectory(source_dir)
    destination = tmp_path / "vasp.traj.jsonl"
    httk.core.save(source, destination)
    loaded = httk.core.load(destination)
    assert isinstance(loaded, JsonlTrajectory)
    assert loaded.nframes == 3
    assert loaded.frame(1).sites.reduced_coords.to_floats() == [[0.2, 0.3, 0.4]]
    assert loaded.header["x-httk-trajectory"]["constant_cell"] is not None
    record = TrajectoryRecord(**project_storage_record(TrajectoryRecord, JsonlTrajectory(destination)))
    assert record.source_locator == str(destination)
    adapted_record = TrajectoryRecord(**project_storage_record(TrajectoryRecord, loaded))
    assert adapted_record.source_locator == str(destination)


AL_300K = Path(__file__).parents[2] / "electronic-structure-example-data" / "MD" / "VASP" / "Al_300K"


@pytest.mark.extended
@pytest.mark.skipif(not AL_300K.is_dir(), reason="workspace Al_300K fixture is unavailable")
def test_workspace_al_300k_jsonl_probe(tmp_path: Path) -> None:
    start = perf_counter()
    source = VASPTrajectory(AL_300K)
    output = tmp_path / "Al_300K.traj.jsonl"
    httk.core.save(source, output)
    loaded = httk.core.load(output)
    elapsed = perf_counter() - start
    print(f"Al_300K JSONL conversion: {loaded.nframes} frames in {elapsed:.2f}s")
    assert loaded.nframes == 10000
