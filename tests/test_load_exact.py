"""Every structure and trajectory reader reads its numbers exactly as written."""

import io
from fractions import Fraction
from pathlib import Path

import httk.core
import pytest
from httk.core.loading import load_source

from httk.atomistic import (
    Cell,
    Sites,
    Species,
    Trajectory,
    UnitcellStructure,
    UnitcellStructureView,
    asu_structures_from_cif,
)

FIXTURES = Path(__file__).parent / "fixtures"

# P3 with a site written at the rounded special position (1/3, 2/3, z), with and without an uncertainty.
TRIGONAL_CIF = """data_test
_cell_length_a 3.567
_cell_length_b 3.567
_cell_length_c 4.1
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
-y,x-y,z
-x+y,-x,z
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
B1 B {x} {y} 0.25
"""

OUTCAR = """ vasp.5.2.12 synthetic complex
   TITEL  = PAW_PBE Na_pv 05Jan2001
   number of ions     NIONS =      1
   ions per type =               1
 ----------------------------------------- Iteration    1(   1)  ---------------------------------------
       direct lattice vectors                 reciprocal lattice vectors
     3.567000000  0.000000000  0.000000000     0.1 0.0 0.0
     0.000000000  3.567000000  0.000000000     0.0 0.1 0.0
     0.000000000  0.000000000  3.567000000     0.0 0.0 0.1
  POSITION                                       TOTAL-FORCE (eV/Angst)
  -----------------------------------------------------------------------------------
     0.333333  0.000000  0.000000        0.000000  0.000000  0.000000
    total drift:                              0.000000 0.000000 0.000000
  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
  free  energy   TOTEN  =       -1.100 eV
  energy  without entropy=      -1.100  energy(sigma->0) =      -1.100
 General timing and accounting informations for this job:
"""

XDATCAR = """diamond
3.567
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
C
2
Direct configuration= 1
0.00 0.00 0.00
0.25 0.25 0.25
"""


@pytest.mark.parametrize(("x", "y"), [("0.3333", "0.6667"), ("0.3333(2)", "0.6667(2)")])
def test_cif_maps_special_positions_and_zero_tolerance_keeps_written_coordinates(
    tmp_path: Path, x: str, y: str
) -> None:
    path = tmp_path / "a.cif"
    path.write_text(TRIGONAL_CIF.format(x=x, y=y), encoding="utf-8")

    snapped = UnitcellStructureView(httk.core.load(path))
    assert snapped.sites.reduced_coords.to_fractions() == [[Fraction(1, 3), Fraction(2, 3), Fraction(1, 4)]]
    assert snapped.cell.lengths[0] == Fraction(3567, 1000)
    (asu,) = asu_structures_from_cif(httk.core.load(path, raw=True), tolerance=0.0)
    written = UnitcellStructureView(asu)
    assert [Fraction(3333, 10000), Fraction(6667, 10000), Fraction(1, 4)] in written.sites.reduced_coords.to_fractions()


def test_xdatcar_reads_exact_decimals(tmp_path: Path) -> None:
    path = tmp_path / "XDATCAR"
    path.write_text(XDATCAR, encoding="utf-8")
    assert httk.core.load(path).frame(0).cell.scale.fractions == Fraction(3567, 1000)


def test_outcar_reads_exact_decimals(tmp_path: Path) -> None:
    path = tmp_path / "OUTCAR"
    path.write_text(OUTCAR, encoding="utf-8")
    x = httk.core.load(path).frame(0).sites.reduced_coords.to_fractions()[0][0]
    assert x == Fraction(333333, 10**6) / Fraction(3567, 1000)


def test_trajectory_jsonl_reads_shortest_decimals(tmp_path: Path) -> None:
    structure = UnitcellStructure(
        Cell([[Fraction(3567, 1000), 0, 0], [0, 2, 0], [0, 0, 2]]),
        Sites([[Fraction(1, 10), 0, 0]]),
        [Species.from_object("Al")],
        ["Al"],
    )
    path = tmp_path / "t.traj.jsonl"
    httk.core.save(Trajectory([structure], {"energy": [1]}, None), path)

    frame = httk.core.load(path).frame(0)
    assert frame.cell.lengths[0] == Fraction(3567, 1000)
    assert frame.sites.reduced_coords.to_fractions()[0][0] == Fraction(1, 10)


@pytest.mark.parametrize("name", ["x.cif", "x.mcif"])
def test_cif_reads_from_a_binary_stream(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    if name == "x.cif":
        path.write_text(TRIGONAL_CIF.format(x="0.3333", y="0.6667"), encoding="utf-8")
    else:
        path.write_bytes((FIXTURES / "magnetic_cartesian.mcif").read_bytes())
    streamed = load_source(io.BytesIO(path.read_bytes()), name, raw=True)
    assert streamed == httk.core.load(path, raw=True)
    if name == "x.cif":
        assert load_source(io.BytesIO(path.read_bytes()), name) == httk.core.load(path)
