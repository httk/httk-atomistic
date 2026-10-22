"""Tests for the ABINIT input reader/writer and main-output reader."""

import gzip
import io
import math
from fractions import Fraction
from pathlib import Path

import pytest
from httk.core import SurdVector, exactmath, load, save

from httk.atomistic import Species, Trajectory, UnitcellStructure
from httk.atomistic.integrations._geometry import _structure_from_geometry
from httk.atomistic.integrations.abinit.io import read_abinit_input, read_abinit_output
from httk.atomistic.integrations.abinit.io.inputs import _abinit_input_payload_from_structure, _render_abinit_input

FIXTURES = Path(__file__).with_name("fixtures") / "abinit"
BOHR = Fraction("0.52917720859")
SI_CELL = [
    [0, Fraction("5.13"), Fraction("5.13")],
    [Fraction("5.13"), 0, Fraction("5.13")],
    [Fraction("5.13")] * 2 + [0],
]


def _load_text(text: str) -> UnitcellStructure:
    return _structure_from_geometry(read_abinit_input(io.StringIO(text)))


def _bohr(rows: list[list[Fraction | int]]) -> SurdVector:
    return SurdVector([[BOHR * value for value in row] for row in rows])


def test_si_input_geometry_is_exact() -> None:
    structure = load(str(FIXTURES / "si.abi"))
    assert structure.cell.basis == _bohr(SI_CELL)
    assert structure.sites.reduced_coords.to_fractions() == [[0, 0, 0], [Fraction(1, 4)] * 3]
    assert [item.name for item in structure.species] == ["Si"]
    assert list(structure.species_at_sites) == ["Si", "Si"]
    payload = load(str(FIXTURES / "si.abi"), raw=True)
    assert payload["format"] == "abinit-input"
    assert payload["raw"] == (FIXTURES / "si.abi").read_text()


def test_si_output_is_one_frame_equal_to_input() -> None:
    trajectory = load(str(FIXTURES / "si.abo"))
    assert isinstance(trajectory, Trajectory)
    assert trajectory.nframes == 1
    frame = trajectory.frame(0)
    structure = load(str(FIXTURES / "si.abi"))
    assert frame.cell.basis == structure.cell.basis
    assert frame.sites.reduced_coords == structure.sites.reduced_coords
    assert list(frame.species_at_sites) == ["Si", "Si"]


def test_output_frames_after_computation_when_geometry_changes() -> None:
    text = (FIXTURES / "si.abo").read_text()
    head, marker, tail = text.partition("-outvars: echo values of variables after computation")
    tail = tail.replace(
        "2.5000000000E-01  2.5000000000E-01  2.5000000000E-01", "2.6000000000E-01  2.5000000000E-01  2.5000000000E-01"
    )
    payload = read_abinit_output(io.StringIO(head + marker + tail), precision=1e-3)
    assert len(payload["frames"]) == 2
    assert payload["frames"][1]["positions"][1][0] == "2.6000000000E-01"
    assert all(frame["precision_override"] == 1e-3 for frame in payload["frames"])
    # xangst is never read: corrupting it changes nothing.
    assert (
        read_abinit_output(io.StringIO(text.replace("1.3573395400E+00", "9.9E+00")))["frames"]
        == read_abinit_output(io.StringIO(text))["frames"]
    )


def test_output_refusals() -> None:
    with pytest.raises(ValueError, match="not an ABINIT main output"):
        read_abinit_output(io.StringIO("acell 1 1 1\n"))
    text = (FIXTURES / "si.abo").read_text()
    with pytest.raises(ValueError, match="multi-dataset"):
        read_abinit_output(io.StringIO(text.replace("            acell ", "           acell1 ", 1)))
    with pytest.raises(ValueError, match="multi-dataset"):
        read_abinit_output(
            io.StringIO(text.replace("            natom ", "           ndtset 2\n            natom ", 1))
        )


SYNTAX = """
# comment ! more
ACELL 2*1.0d1 5.0 angstr   ! unit word, repetition, Fortran exponent
rprim 1 0 0  0 1 0  0 0 1
natom 2 ntypat 2 typat 2 1 znucl 8 26
xred 0 0 0  1/3 -sqrt(0.25) sqrt(0.0625)
pseudos "a # b.psp8, c.psp8"
ecut 10 Ha
"""


def test_syntax_units_repetition_and_expressions() -> None:
    structure = _load_text(SYNTAX)
    assert structure.cell.basis == SurdVector([[10, 0, 0], [0, 10, 0], [0, 0, 5]])
    assert structure.sites.reduced_coords.to_fractions()[1] == [Fraction(1, 3), Fraction(-1, 2), Fraction(1, 4)]
    assert [(item.name, item.chemical_symbols) for item in structure.species] == [("O", ("O",)), ("Fe", ("Fe",))]
    assert list(structure.species_at_sites) == ["Fe", "O"]


def test_irrational_sqrt_value_stays_exact() -> None:
    structure = _load_text("acell 3*2 Angstrom rprim 1 0 0 -1/2 sqrt(0.75) 0 0 0 1 znucl 6 typat 1 xred 0 0 0")
    assert structure.cell.basis == SurdVector.from_radicand_map(
        {1: [[2, 0, 0], [-1, 0, 0], [0, 0, 2]], 3: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]}
    )


def test_huge_sqrt_radicand_is_approximated_and_claimed() -> None:
    # Untrusted radicands beyond httk-core's exact bound are approximated, never factored.
    huge = 2**61 - 1  # a prime, far beyond the exact-surd radicand bound
    payload = read_abinit_input(
        io.StringIO(f"acell 3*1 rprim sqrt({huge}) 0 0 0 1 0 0 0 1 znucl 6 xcart sqrt({huge}) 0 0")
    )
    root = payload["cell"][0][0]
    assert isinstance(root, Fraction) and abs(root * root - huge) < 1
    assert payload["cell_precision"] == exactmath.default_accuracy
    assert payload["positions_precision"] == exactmath.default_accuracy
    angdeg = read_abinit_input(io.StringIO(f"acell sqrt({huge}) 1 1 angdeg 90 90 90 znucl 6"))
    assert angdeg["cell_precision"] == exactmath.default_accuracy
    # sqrt(2)-class tokens stay exact surds and claim nothing.
    exact = read_abinit_input(io.StringIO("acell 3*1 rprim sqrt(2) 0 0 0 sqrt(3) 0 0 0 1 znucl 6 xred 0 0 0"))
    assert exact["cell"][0][0] * exact["cell"][0][0] == 2
    assert exact["cell_precision"] is None and "positions_precision" not in exact


def test_defaults_and_xcart_xangst_scalecart() -> None:
    structure = _load_text("znucl 14 xcart 0.5 0.25 0")
    assert structure.cell.basis == SurdVector([[BOHR, 0, 0], [0, BOHR, 0], [0, 0, BOHR]])
    assert structure.sites.reduced_coords.to_fractions() == [[Fraction(1, 2), Fraction(1, 4), 0]]
    assert _load_text("znucl 14").sites.reduced_coords.to_fractions() == [[0, 0, 0]]
    angstrom = _load_text("acell 3*4 xangst 1 0 0 znucl 14")
    assert angstrom.sites.reduced_coords.to_fractions() == [[1 / (4 * BOHR), 0, 0]]
    xcart_angstrom = _load_text("acell 3*2 Angstrom xcart 1 0 0 Angstrom znucl 14")
    assert xcart_angstrom.sites.reduced_coords.to_fractions() == [[Fraction(1, 2), 0, 0]]
    scaled = _load_text("acell 1 2 3 Angstrom rprim 1 1 0 0 1 1 1 0 1 scalecart 2 3 5 znucl 14")
    assert scaled.cell.basis == SurdVector([[2, 3, 0], [0, 6, 10], [6, 0, 15]])


def test_same_element_types_are_numbered() -> None:
    structure = _load_text("natom 3 ntypat 3 typat 1 2 3 znucl 26 26 8 xred 0 0 0 0.5 0.5 0.5 0.25 0.25 0.25")
    assert list(structure.species_at_sites) == ["Fe1", "Fe2", "O"]


def test_angdeg_matches_documented_construction() -> None:
    hexagonal = _load_text("acell 3 3 5 Angstrom angdeg 90 90 120 znucl 6")
    assert hexagonal.cell.basis == SurdVector.from_radicand_map(
        {1: [[3, 0, 0], [Fraction(-3, 2), 0, 0], [0, 0, 5]], 3: [[0, 0, 0], [0, Fraction(3, 2), 0], [0, 0, 0]]}
    )
    trigonal = _load_text("acell 3*1 Angstrom angdeg 3*60 znucl 6")
    # FCC primitive vectors: every pair at 60 degrees, unit length, trigonal axis along z.
    basis = trigonal.cell.basis
    gram = basis * basis.T()
    assert gram == SurdVector(
        [[1, Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 2), 1, Fraction(1, 2)], [Fraction(1, 2)] * 2 + [1]]
    )


def test_inexact_angdeg_uses_lattice_parameters() -> None:
    payload = read_abinit_input(
        io.StringIO("acell 4.0 5.00 6.000 Angstrom angdeg 90 104.5 90 znucl 6 xred 0.1 0.2 0.3")
    )
    assert "cell" not in payload
    assert payload["cell_parameters"] == ["4.0", "5.00", "6.000", "90", "104.5", "90"]
    structure = _structure_from_geometry(payload)
    assert structure.cell.lengths == (4, 5, 6)
    # Precision from the length tokens only (CIF's rule); the fractional sites keep theirs.
    assert structure.cell.precision == Fraction(1, 10)
    assert structure.sites.reduced_coords.to_fractions() == [[Fraction(1, 10), Fraction(1, 5), Fraction(3, 10)]]
    basis = structure.cell.basis.to_floats()
    cos_beta = sum(x * y for x, y in zip(basis[0], basis[2])) / (4 * 6)
    assert math.degrees(math.acos(cos_beta)) == pytest.approx(104.5, abs=1e-8)
    assert _load_text("acell 3*10 Angstrom angdeg 3*70 znucl 6").cell.lengths == (10, 10, 10)
    # In Bohr the CellParams approximation reaches the b and c lengths (~1e-11 Å; see the report).
    bohr = [float(length) for length in _load_text("acell 3*10 angdeg 3*70 znucl 6").cell.lengths]
    assert bohr == pytest.approx([10 * float(BOHR)] * 3, abs=1e-9)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("angdeg 90 104.5 90 znucl 6 xcart 0 0 0", "xcart with angdeg 90 104.5 90 needs an explicit rprim"),
        ("angdeg 3*70 znucl 6 xangst 0 0 0", "xangst with angdeg .* needs an explicit rprim"),
        ("angdeg 90 104.5 90 scalecart 1 1 2 znucl 6", "scalecart with angdeg .* needs an explicit rprim"),
    ],
)
def test_inexact_angdeg_refusals(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _load_text(text)


def test_exact_angdeg_keeps_abinit_frame_with_cartesian_positions() -> None:
    structure = _load_text("acell 3*2 Angstrom angdeg 90 90 120 scalecart 1 1 2 znucl 6 xcart 1 0 0 Angstrom")
    assert structure.sites.reduced_coords.to_fractions() == [[Fraction(1, 2), 0, 0]]


def test_angdeg_agrees_with_pymatgen_port_of_ingeo() -> None:
    abiobjects = pytest.importorskip("pymatgen.io.abinit.abiobjects")
    from pymatgen.core.units import bohr_to_ang

    for angles in ([90, 90, 120], [60, 60, 60], [90, 90, 60], [90, 60, 120], [60, 90, 90], [90, 90, 90]):
        ours = _load_text("angdeg " + " ".join(map(str, angles)) + " znucl 6").cell.basis.to_floats()
        theirs = abiobjects.lattice_from_abivars(acell=[1, 1, 1], angdeg=angles).matrix / bohr_to_ang
        for our_row, their_row in zip(ours, theirs):
            assert our_row == pytest.approx([float(BOHR) * value for value in their_row], abs=1e-9)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("ndtset 2 acell1 3*1 znucl 14", "dataset"),
        ("ndtset 2 xred: 0 0 0 znucl 14", "dataset"),
        ("xred+ 0 0 0 znucl 14", "dataset"),
        ("xred_lastimg 0 0 0 znucl 14", "image"),
        ("spgroup 227 natrd 1 znucl 14", "symmetry"),
        ("natrd 1 znucl 14", "symmetry"),
        ('structure "poscar:POSCAR"', "structure variable"),
        ("getxred 1 znucl 14", "another dataset"),
        ("natom 2 typat 1 1 znucl 14 xred 0 0 0", "xred needs 6"),
        ("natom 2 znucl 14", "typat"),
        ("acell 3*1", "znucl"),
        ("ntypat 2 typat 1 znucl 14", "znucl needs 2"),
        ("typat 2 znucl 14", "typat values"),
        ("znucl 14.5", "not an integer"),
        ("acell 3*1 nm znucl 14", "unit 'nm'"),
        ("acell 3*1 Angstroms znucl 14", "has no value"),
        ("xred 0 0 0 Bohr znucl 14", "unit 'bohr' on xred"),
        ("acell 3*1 acell 3*2 znucl 14", "more than once"),
        ("xred 0 0 0 xcart 0 0 0 znucl 14", "more than once"),
        ("rprim 1 0 0 0 1 0 0 0 1 angdeg 3*90 znucl 14", "mutually exclusive"),
        ("acell *1 znucl 14", "repetition"),
        ("acell 3*1 znucl 14 xred 0 0 sqrt(2)/2", "not a supported number"),
        ("acell 3*1 znucl 14 xred 0 0 1/0", "not a supported number"),
        ('pseudos "unterminated znucl 14', "unterminated"),
        ("1.0 acell 3*1", "precedes"),
        ("rprim 1 0 0 1 0 0 0 0 1 znucl 14", "degenerate"),
    ],
)
def test_input_refusals(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _load_text(text)


def test_non_utf8_input_is_a_value_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.abi"
    path.write_bytes(b"acell 3*1 \xff\xfe znucl 14\n")
    with pytest.raises(ValueError, match="UTF-8"):
        load(str(path))


def _sample() -> UnitcellStructure:
    return UnitcellStructure(
        [["3.5", 0, 0], [0, "3.25", 0], ["0.5", 0, "4.125"]],
        [[0, 0, 0], ["0.5", "0.25", "0.125"], ["0.75", "0.5", "0.5"]],
        [Species("Fe1", ("Fe",), (1,)), Species("O", ("O",), (1,)), Species("Fe2", ("Fe",), (1,))],
        ["Fe1", "O", "Fe2"],
    )


def test_writer_output_and_round_trip(tmp_path: Path) -> None:
    structure = _sample()
    text = _render_abinit_input(_abinit_input_payload_from_structure(structure))
    assert text.splitlines()[0] == "acell 1 1 1 Angstrom"
    assert "typat\n 1 2 3\nznucl 26 8 26\nxred\n" in text
    path = tmp_path / "x.abi"
    save(structure, str(path))
    assert path.read_text() == text
    back = load(str(path))
    assert back.cell.basis == structure.cell.basis
    assert back.sites.reduced_coords == structure.sites.reduced_coords
    # acell 1 1 1 claims nothing: the re-read cell precision comes from the rprim digits only
    si = load(str(FIXTURES / "si.abi"))
    save(si, str(tmp_path / "si.abi"))
    si_back = load(str(tmp_path / "si.abi"))
    assert si_back.cell.basis == si.cell.basis
    assert si_back.cell.precision == Fraction(1, 10**13)  # rprim tokens 2.7146790800667 Å
    # typat/znucl carry no names: same-element species are renumbered by type.
    assert list(back.species_at_sites) == ["Fe1", "O", "Fe2"]


def test_raw_round_trip_is_byte_identical_and_refuses_options(tmp_path: Path) -> None:
    payload = load(str(FIXTURES / "si.abi"), raw=True)
    target = tmp_path / "copy.abi"
    save(payload, str(target))
    assert target.read_bytes() == (FIXTURES / "si.abi").read_bytes()
    with pytest.raises(ValueError, match="raw"):
        save(payload, str(tmp_path / "other.abi"), comment="x")


def test_compressed_input(tmp_path: Path) -> None:
    path = tmp_path / "si.abi.gz"
    path.write_bytes(gzip.compress((FIXTURES / "si.abi").read_bytes()))
    assert load(str(path)).cell.basis == load(str(FIXTURES / "si.abi")).cell.basis


def test_generic_names_load_with_format(tmp_path: Path) -> None:
    path = tmp_path / "si.in"
    path.write_bytes((FIXTURES / "si.abi").read_bytes())
    assert load(str(path), format="abinit-input").cell.basis == _bohr(SI_CELL)
    out = tmp_path / "run.out"
    out.write_bytes((FIXTURES / "si.abo").read_bytes())
    assert load(str(out), format="abinit-output").nframes == 1


def test_writer_refuses_non_periodic(tmp_path: Path) -> None:
    from httk.atomistic import Cell

    structure = _sample()
    flat = UnitcellStructure(
        Cell(structure.cell, periodicity=(True, True, False)),
        structure.sites,
        list(structure.species),
        list(structure.species_at_sites),
    )
    with pytest.raises(ValueError, match="periodic"):
        save(flat, str(tmp_path / "flat.abi"))


def test_ase_reads_written_input(tmp_path: Path) -> None:
    ase_io = pytest.importorskip("ase.io")
    import numpy

    structure = _sample()
    path = tmp_path / "x.abi"
    save(structure, str(path))
    atoms = ase_io.read(str(path), format="abinit-in")
    assert atoms.cell[:] == pytest.approx(numpy.array(structure.cell.basis.to_floats()), abs=1e-8)
    cartesian = (structure.sites.reduced_coords * structure.cell.basis).to_floats()
    assert atoms.positions == pytest.approx(numpy.array(cartesian), abs=1e-8)
    assert list(atoms.get_chemical_symbols()) == ["Fe", "O", "Fe"]


def test_cell_precision_propagates_token_digits() -> None:
    # si.abi: rprim 0.0/0.5 are exact fractions (POSCAR's rule), so only acell 10.26 (1/100) claims: 1/100 * 0.5 Bohr.
    assert load(str(FIXTURES / "si.abi"), raw=True)["cell_precision"] == Fraction(1, 200)
    assert load(str(FIXTURES / "si.abi")).cell.precision == Fraction(1, 200) * BOHR
    # si.abo echo: acell 1.0260000000E+01 (1e-9) times rprim 0.0000000000E+00 (1e-10).
    trajectory = load(str(FIXTURES / "si.abo"))
    assert trajectory.frame(0).cell.precision == Fraction(1026, 10**12) * BOHR
    # Integer acell and exact rprim claim nothing; exact angdeg rows leave only acell's digits.
    exact = read_abinit_input(io.StringIO("acell 3*10 rprim 0 1/2 1/2 1/2 0 1/2 1/2 1/2 0 znucl 14"))
    assert exact["cell_precision"] is None
    assert _structure_from_geometry(exact).cell.precision is None
    angdeg = read_abinit_input(io.StringIO("acell 3*10.0 angdeg 90 90 120 scalecart 1 1 2 znucl 14"))
    assert angdeg["cell_precision"] == Fraction(2, 10)
    # A scalecart 1.0 is exact too; a 1.5 claims 1/10.
    assert read_abinit_input(io.StringIO("acell 3*10 scalecart 1.0 1.0 1.0 znucl 14"))["cell_precision"] is None
    assert read_abinit_input(io.StringIO("acell 3*10 scalecart 1.0 1.0 1.5 znucl 14"))["cell_precision"] == 1
