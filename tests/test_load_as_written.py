"""``load(path, as_written=True)`` is accepted by every structure reader and never snaps a written decimal."""

import io
from fractions import Fraction
from pathlib import Path

import httk.core
import pytest
from httk.core.loading import load_source

from httk.atomistic import Cell, Sites, Species, Trajectory, UnitcellStructure, UnitcellStructureView

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
def test_cif_as_written_keeps_written_coordinates(tmp_path: Path, x: str, y: str) -> None:
    path = tmp_path / "a.cif"
    path.write_text(TRIGONAL_CIF.format(x=x, y=y), encoding="utf-8")

    snapped = UnitcellStructureView(httk.core.load(path))
    assert snapped.sites.reduced_coords.to_fractions() == [[Fraction(1, 3), Fraction(2, 3), Fraction(1, 4)]]
    written = UnitcellStructureView(httk.core.load(path, as_written=True))
    assert [Fraction(3333, 10000), Fraction(6667, 10000), Fraction(1, 4)] in written.sites.reduced_coords.to_fractions()
    assert written.cell.lengths[0] == Fraction(3567, 1000)
    with pytest.raises(ValueError, match="as_written must be a bool"):
        httk.core.load(path, as_written="yes")


def test_xdatcar_as_written_reads_exact_decimals(tmp_path: Path) -> None:
    path = tmp_path / "XDATCAR"
    path.write_text(XDATCAR, encoding="utf-8")
    assert httk.core.load(path).frame(0).cell.scale.fractions == Fraction(346, 97)
    assert httk.core.load(path, as_written=True).frame(0).cell.scale.fractions == Fraction(3567, 1000)


def test_outcar_as_written_reads_exact_decimals(tmp_path: Path) -> None:
    path = tmp_path / "OUTCAR"
    path.write_text(OUTCAR, encoding="utf-8")
    # Six-decimal OUTCAR positions snap to the simplest consistent rational by default.
    x = httk.core.load(path).frame(0).sites.reduced_coords.to_fractions()[0][0]
    assert x == Fraction(1, 3) / Fraction(3567, 1000)
    x = httk.core.load(path, as_written=True).frame(0).sites.reduced_coords.to_fractions()[0][0]
    assert x == Fraction(333333, 10**6) / Fraction(3567, 1000)


def test_trajectory_jsonl_as_written_reads_written_decimals(tmp_path: Path) -> None:
    structure = UnitcellStructure(
        Cell([[Fraction(3567, 1000), 0, 0], [0, 2, 0], [0, 0, 2]]),
        Sites([[Fraction(1, 10), 0, 0]]),
        [Species.from_object("Al")],
        ["Al"],
    )
    path = tmp_path / "t.traj.jsonl"
    httk.core.save(Trajectory([structure], {"energy": [1]}, None), path)

    assert httk.core.load(path).frame(0).cell.lengths[0] != Fraction(3567, 1000)
    frame = httk.core.load(path, as_written=True).frame(0)
    assert frame.cell.lengths[0] == Fraction(3567, 1000)
    assert frame.sites.reduced_coords.to_fractions()[0][0] == Fraction(1, 10)


@pytest.mark.parametrize(
    ("path", "fmt"),
    [
        (FIXTURES / "magnetic_cartesian.mcif", None),
        (FIXTURES / "qe" / "si.in", "qe-pw-input"),
        (FIXTURES / "qe" / "si.out", "qe-pw-output"),
        (FIXTURES / "abinit" / "si.abi", None),
        (FIXTURES / "abinit" / "si.abo", None),
        (FIXTURES / "cp2k" / "si.inp", "cp2k-input"),
    ],
)
def test_exact_readers_accept_as_written(path: Path, fmt: str | None) -> None:
    assert httk.core.load(path, format=fmt, raw=True, as_written=True) == httk.core.load(path, format=fmt, raw=True)
    with pytest.raises(ValueError, match="as_written must be a bool"):
        httk.core.load(path, format=fmt, as_written=1)


def test_lammps_and_httk_structure_accept_as_written(tmp_path: Path) -> None:
    structure = UnitcellStructure(
        Cell([[Fraction(3567, 1000), 0, 0], [0, 4, 0], [0, 0, 4]]),
        Sites([[Fraction(1234567, 10**7), 0, 0]]),
        [Species.from_object("Si")],
        ["Si"],
    )
    for name, fmt in (("a.lmp", "lammps-data"), ("a.httkstruct", None)):
        path = tmp_path / name
        httk.core.save(structure, path)
        assert UnitcellStructureView(httk.core.load(path, format=fmt, as_written=True)) == UnitcellStructureView(
            httk.core.load(path, format=fmt)
        )


def test_lammps_dump_accepts_as_written_without_forwarding_it(tmp_path: Path) -> None:
    path = tmp_path / "atoms.lammpstrj"
    path.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0 3.567\n0 4\n0 4\n"
        "ITEM: ATOMS id type x y z\n1 1 0.5 0 0\n",
        encoding="utf-8",
    )
    assert httk.core.load(path, raw=True, as_written=True)["options"] == {}


@pytest.mark.parametrize("as_written", [False, True])
@pytest.mark.parametrize("name", ["x.cif", "x.mcif"])
def test_cif_reads_from_a_binary_stream(tmp_path: Path, name: str, as_written: bool) -> None:
    path = tmp_path / name
    if name == "x.cif":
        path.write_text(TRIGONAL_CIF.format(x="0.3333", y="0.6667"), encoding="utf-8")
    else:
        path.write_bytes((FIXTURES / "magnetic_cartesian.mcif").read_bytes())
    streamed = load_source(io.BytesIO(path.read_bytes()), name, raw=True, as_written=as_written)
    assert streamed == httk.core.load(path, raw=True, as_written=as_written)
    if name == "x.cif":
        assert load_source(io.BytesIO(path.read_bytes()), name, as_written=as_written) == httk.core.load(
            path, as_written=as_written
        )
