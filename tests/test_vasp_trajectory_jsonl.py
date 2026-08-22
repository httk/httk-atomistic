"""VASP trajectory to JSONL conversion."""

from pathlib import Path

import httk.core
import pytest
from httk.core.storage import project_storage_record

from httk.atomistic import JsonlTrajectory, TrajectoryRecord, UnitcellStructureView, VASPStructure, VASPTrajectory

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


@pytest.mark.parametrize("backend", [VASPTrajectory, JsonlTrajectory])
def test_trajectory_identity_adoption_does_not_reinitialize(
    backend: type, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "vasp"
    source_dir.mkdir()
    (source_dir / "POSCAR").write_text(POSCAR, encoding="utf-8")
    (source_dir / "XDATCAR").write_text(XDATCAR, encoding="utf-8")
    source = VASPTrajectory(source_dir)
    destination = tmp_path / "vasp.traj.jsonl"
    httk.core.save(source, destination)

    if backend is VASPTrajectory:
        existing = source
        initialized = "_vasp_trajectory_initialized"
    else:
        existing = JsonlTrajectory(destination)
        initialized = "_jsonl_initialized"
    nframes = existing.nframes

    assert getattr(existing, initialized) is True
    assert backend(existing) is existing
    assert backend(existing, kind="something-else") is existing
    assert getattr(existing, initialized) is True
    assert existing.nframes == nframes

    init_calls: list[None] = []
    original_init = backend.__init__

    def counting_init(self: object, *args: object, **kwargs: object) -> None:
        init_calls.append(None)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(backend, "__init__", counting_init)
    assert backend._backend_adopt(existing) is existing
    assert init_calls == []
    assert backend._backend_adopt(existing, kind="other") is None


def test_vasp_structure_view_kind_dispatch(tmp_path: Path) -> None:
    source = tmp_path / "POSCAR"
    source.write_text(POSCAR, encoding="utf-8")
    backend = VASPStructure(source)

    with pytest.raises(TypeError):
        UnitcellStructureView(backend, kind="plain")

    view = UnitcellStructureView(backend, kind=VASPStructure.kind)
    assert view._backend is backend
    assert view.unwrap() is source
