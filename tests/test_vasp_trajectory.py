"""Lazy VASP trajectory adapter checks."""

from pathlib import Path

import httk.core
import pytest
from httk.core.storage import project_storage_record

from httk.atomistic import TrajectoryRecord, TrajectoryView, VASPTrajectory

POSCAR = """Synthetic POSCAR
1.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
Si O
1 1
Direct
0.0 0.0 0.0
0.0 0.0 0.0
"""

XDATCAR = """Synthetic XDATCAR
1.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
Si O
1 1
Direct configuration= 1
0.1 0.2 0.3
0.4 0.5 0.6
Direct configuration= 2
0.2 0.3 0.4
0.5 0.6 0.7
Direct configuration= 3
0.3 0.4 0.5
0.6 0.7 0.8
"""


def _outcar_frame(number: int, energy: str, temperature: str) -> str:
    return f""" ----------------------------------------- Iteration {number:4d}(   1)  ---------------------------------------
  in kB       1 2 3 4 5 6
       direct lattice vectors                 reciprocal lattice vectors
     2.000000000 0.000000000 0.000000000  0.1 0.0 0.0
     0.000000000 2.000000000 0.000000000  0.0 0.1 0.0
     0.000000000 0.000000000 2.000000000  0.0 0.0 0.1
  POSITION                                       TOTAL-FORCE (eV/Angst)
  -----------------------------------------------------------------------------------
     {number / 10:.1f}00000 0.000000 0.000000  0.0 0.0 0.0
     0.000000 {number / 20:.2f}0000 0.000000  0.0 0.0 0.0
    total drift: 0.0 0.0 0.0
  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
  free  energy   TOTEN  =       {energy} eV
  energy  without entropy=      {energy}  energy(sigma->0) =      {energy}
  kin. lattice  EKIN_LAT= 0.000000 (temperature {temperature} K)
"""


OUTCAR = (
    "vasp.5.2.12 synthetic\n"
    + "".join(_outcar_frame(number, f"-{number}.1", f"{300 + number}") for number in (1, 2, 3))
    + " General timing and accounting informations\n"
)

OUTCAR_STANDALONE = (
    "vasp.5.2.12 synthetic\n"
    " TITEL  = PAW_PBE Si 06Sep2000\n"
    " TITEL  = PAW_PBE O 08Apr2002\n"
    " ions per type = 1 1\n" + OUTCAR.split("\n", 1)[1]
)

XDATCAR_CARTESIAN = XDATCAR.replace(
    "Direct configuration= 1\n0.1 0.2 0.3\n0.4 0.5 0.6",
    "Cartesian configuration= 1\n0.2 0.4 0.6\n0.8 1.0 1.2",
)

XDATCAR_CARTESIAN_SCALED = XDATCAR_CARTESIAN.replace(
    "1.0\n2.0 0.0 0.0\n0.0 2.0 0.0\n0.0 0.0 2.0",
    "2.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0",
).replace("0.2 0.4 0.6\n0.8 1.0 1.2", "1.0 0.0 0.0\n0.0 1.0 0.0")

XDATCAR_NPT_SCALE = """Synthetic XDATCAR
1.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
1
Direct configuration= 1
0.0 0.0 0.0
NPT step 2
2.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
1
Direct configuration= 2
0.0 0.0 0.0
"""

POSCAR_NPT = """Synthetic POSCAR
1.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
Si
1
Direct
0.0 0.0 0.0
"""


def _write_directory(path: Path, *, outcar: bool = True, xdatcar: bool = True) -> None:
    (path / "POSCAR").write_text(POSCAR, encoding="utf-8")
    if outcar:
        (path / "OUTCAR").write_text(OUTCAR, encoding="utf-8")
    if xdatcar:
        (path / "XDATCAR").write_text(XDATCAR, encoding="utf-8")


def test_directory_prefers_xdatcar_geometry_and_reads_outcar_observables(tmp_path: Path) -> None:
    _write_directory(tmp_path)
    trajectory = VASPTrajectory(tmp_path)

    assert trajectory.nframes == 3
    assert trajectory.frame(1).sites.reduced_coords.to_floats() == [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]
    assert trajectory.observable("_httk_frame_total_energies") == (-1.1, -2.1, -3.1)
    assert trajectory.observable("_httk_frame_temperatures") == (301.0, 302.0, 303.0)
    assert trajectory.observable("_httk_frame_stresses")[0] == pytest.approx((-0.1, -0.2, -0.3, -0.5, -0.6, -0.4))
    assert trajectory.unwrap() is tmp_path
    assert TrajectoryView(trajectory).frame(0) == trajectory.frame(0)


def test_outcar_only_converts_cartesian_positions_to_reduced(tmp_path: Path) -> None:
    _write_directory(tmp_path, xdatcar=False)
    trajectory = VASPTrajectory(tmp_path)
    assert trajectory.nframes == 3
    assert trajectory.frame(0).sites.reduced_coords.to_floats() == [[0.05, 0.0, 0.0], [0.0, 0.025, 0.0]]


def test_xdatcar_only_has_empty_observables_and_supports_negative_index(tmp_path: Path) -> None:
    _write_directory(tmp_path, outcar=False)
    trajectory = VASPTrajectory(tmp_path)
    assert trajectory.observable_names == ()
    assert float(trajectory.frame(-1).sites[0][0]) == pytest.approx(0.3)


def test_standalone_outcar_derives_species_from_ions_and_potcar_titles(tmp_path: Path) -> None:
    source = tmp_path / "OUTCAR"
    source.write_text(OUTCAR_STANDALONE, encoding="utf-8")
    trajectory = VASPTrajectory(source)
    assert trajectory.frame(0).species_at_sites == ("Si", "O")


def test_standalone_outcar_rejects_species_count_disagreement(tmp_path: Path) -> None:
    source = tmp_path / "OUTCAR"
    source.write_text(OUTCAR_STANDALONE.replace("TITEL  = PAW_PBE O 08Apr2002\n", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="ions_per_type and potcar_titles disagree"):
        VASPTrajectory(source).frame(0)


def test_cartesian_xdatcar_coordinates_are_reduced_by_the_cell(tmp_path: Path) -> None:
    source = tmp_path / "XDATCAR"
    source.write_text(XDATCAR_CARTESIAN, encoding="utf-8")
    payload = httk.core.load(source, raw=True)
    first = next(payload["xdatcar"].frames())
    assert "cartesian" in first
    assert VASPTrajectory(source).frame(0).sites.reduced_coords.to_floats() == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]


def test_cartesian_xdatcar_uses_raw_lattice_before_universal_scale(tmp_path: Path) -> None:
    source = tmp_path / "XDATCAR"
    source.write_text(XDATCAR_CARTESIAN_SCALED, encoding="utf-8")
    payload = httk.core.load(source, raw=True)
    first = next(payload["xdatcar"].frames())
    assert "cartesian" in first
    assert VASPTrajectory(source).frame(0).sites.reduced_coords.to_floats() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_npt_xdatcar_uses_repeated_header_scale(tmp_path: Path) -> None:
    source = tmp_path / "vasp"
    source.mkdir()
    (source / "POSCAR").write_text(POSCAR_NPT, encoding="utf-8")
    xdatcar = source / "XDATCAR"
    xdatcar.write_text(XDATCAR_NPT_SCALE, encoding="utf-8")
    payload = httk.core.load(xdatcar, raw=True)
    frames = tuple(payload["xdatcar"].frames())
    assert "scale" in frames[1]
    assert VASPTrajectory(source).frame(1).cell.basis.to_floats() == [
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ]


def test_core_load_outcar_locator_survives_view_projection(tmp_path: Path) -> None:
    source = tmp_path / "OUTCAR"
    source.write_text(OUTCAR_STANDALONE, encoding="utf-8")
    loaded = TrajectoryView(httk.core.load(source))
    record = TrajectoryRecord(**project_storage_record(TrajectoryRecord, loaded))
    assert record.source_locator == str(source)


def test_vasp_path_locator_survives_trajectory_view_projection(tmp_path: Path) -> None:
    _write_directory(tmp_path)
    trajectory = TrajectoryView(VASPTrajectory(tmp_path))
    record = TrajectoryRecord(**project_storage_record(TrajectoryRecord, trajectory))
    assert record.source_locator == str(tmp_path)


def test_mismatched_frame_counts_are_rejected(tmp_path: Path) -> None:
    _write_directory(tmp_path)
    (tmp_path / "XDATCAR").write_text(XDATCAR.rsplit("Direct configuration= 3", 1)[0], encoding="utf-8")
    trajectory = VASPTrajectory(tmp_path)
    with pytest.raises(ValueError, match="XDATCAR=2, OUTCAR=3"):
        _ = trajectory.nframes


def test_core_load_adapts_outcar_to_a_trajectory(tmp_path: Path) -> None:
    source = tmp_path / "OUTCAR"
    source.write_text(OUTCAR, encoding="utf-8")
    raw = httk.core.load(source, raw=True)
    loaded = httk.core.load(source)
    assert raw["format"] == "vasp-outcar"
    assert isinstance(loaded, VASPTrajectory)


def test_loaded_payload_construction_is_lazy() -> None:
    class Unread:
        @property
        def nframes(self) -> int:
            raise AssertionError("payload was touched during construction")

    payload = {"format": "vasp-xdatcar", "xdatcar": Unread()}
    trajectory = VASPTrajectory(payload)
    assert trajectory.unwrap() is payload


AL_300K = Path(__file__).parents[2] / "electronic-structure-example-data" / "MD" / "VASP" / "Al_300K"


@pytest.mark.extended
@pytest.mark.skipif(not AL_300K.is_dir(), reason="workspace Al_300K fixture is unavailable")
def test_workspace_al_300k_trajectory() -> None:
    trajectory = VASPTrajectory(AL_300K)
    assert trajectory.nframes == 10000
    assert trajectory.frame(0).species_at_sites[0] == "Al"
    assert len(trajectory.observable("_httk_frame_temperatures")) == 10000
