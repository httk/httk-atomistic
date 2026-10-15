"""CP2K input/restart geometry reading and ``&SUBSYS`` writing (format ``cp2k-input``)."""

import gzip
import io
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from httk.core import SurdVector, load, save

from httk.atomistic.integrations._geometry import _structure_from_geometry
from httk.atomistic.integrations.cp2k.io import read_cp2k_input

FIXTURES = Path(__file__).parent / "fixtures" / "cp2k"
SI = FIXTURES / "si.inp"
RESTART = FIXTURES / "feo-1.restart"
DIAMOND = [
    (0, 0, 0),
    (0, Fraction(1, 2), Fraction(1, 2)),
    (Fraction(1, 2), 0, Fraction(1, 2)),
    (Fraction(1, 2), Fraction(1, 2), 0),
    (Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)),
    (Fraction(1, 4), Fraction(3, 4), Fraction(3, 4)),
    (Fraction(3, 4), Fraction(1, 4), Fraction(3, 4)),
    (Fraction(3, 4), Fraction(3, 4), Fraction(1, 4)),
]


def _input(cell: str = "A 4 0 0\nB 0 4 0\nC 0 0 4", coord: str = "Na 0 0 0\nCl 2 2 2", extra: str = "") -> str:
    """A minimal full CP2K input around the given &CELL and &COORD bodies."""
    return (
        "&GLOBAL\n  PROJECT test\n&END GLOBAL\n&FORCE_EVAL\n  METHOD Quickstep\n  &SUBSYS\n"
        f"    &CELL\n{cell}\n    &END CELL\n    &COORD\n{coord}\n    &END COORD\n{extra}\n"
        "  &END SUBSYS\n&END FORCE_EVAL\n"
    )


def _load(tmp_path: Path, text: str, **kwargs: Any) -> Any:
    path = tmp_path / "test.inp"
    path.write_text(text)
    return load(str(path), format="cp2k-input", **kwargs)


def test_si_inp_exact_geometry() -> None:
    structure = load(str(SI), format="cp2k-input")
    assert structure.cell.basis == SurdVector(
        [[Fraction("5.43"), 0, 0], [0, Fraction("5.43"), 0], [0, 0, Fraction("5.43")]]
    )
    assert [tuple(row) for row in structure.sites.reduced_coords.to_fractions()] == DIAMOND
    assert [item.name for item in structure.species] == ["Si"]
    assert list(structure.species_at_sites) == ["Si"] * 8
    assert structure.cell.periodicity == (True, True, True)
    payload = load(str(SI), format="cp2k-input", raw=True)
    assert payload["format"] == "cp2k-input"
    assert payload["raw"] == SI.read_text()
    assert payload["positions_kind"] == "fractional"


def test_restart_fixture_cartesian_and_kinds() -> None:
    structure = load(str(RESTART), format="cp2k-input")
    assert structure.cell.basis == SurdVector([[3, 0, 0], [0, 3, 0], [0, 0, 4]])
    assert structure.sites.reduced_coords.to_fractions() == [[0, 0, 0], [Fraction(1, 2)] * 3]
    assert [(item.name, item.chemical_symbols) for item in structure.species] == [("Fe1", ("Fe",)), ("O", ("O",))]


def test_abc_hexagonal_is_exact(tmp_path: Path) -> None:
    text = _input("ABC 3 3 5\nALPHA_BETA_GAMMA 90 90 120", "Si 0 0 0\nSi 0.5 0 0")
    text = text.replace("&COORD\n", "&COORD\nSCALED T\n")
    structure = _load(tmp_path, text)
    half = Fraction(1, 2)
    expected = SurdVector.from_radicand_map(
        {1: [[3, 0, 0], [-3 * half, 0, 0], [0, 0, 5]], 3: [[0, 0, 0], [0, 3 * half, 0], [0, 0, 0]]}
    )
    assert structure.cell.basis == expected
    assert structure.sites.reduced_coords.to_fractions() == [[0, 0, 0], [half, 0, 0]]


def test_abc_rhombohedral_and_default_angles(tmp_path: Path) -> None:
    rhombohedral = _load(tmp_path, _input("ABC 2 2 2\nALPHA_BETA_GAMMA 60 60 60", "C 0 0 0"))
    height = rhombohedral.cell.basis._element((2, 2))
    assert height * height == Fraction(8, 3)
    volume = rhombohedral.cell.volume
    assert volume * volume == 32
    cubic = _load(tmp_path, _input("ABC [angstrom] 2 3 4", "C 0 0 0"))
    assert cubic.cell.basis == SurdVector([[2, 0, 0], [0, 3, 0], [0, 0, 4]])


def test_abc_monoclinic_cartesian_round_trip(tmp_path: Path) -> None:
    # 104.5 degrees has no exact cosine: CellParams' rational approximation of the basis, in
    # CP2K's frame (a along x, b in the xy plane), with the site precision covering it.
    text = _input("ABC 3.00 4.00 5.00\nALPHA_BETA_GAMMA 90 104.5 90", "Na 0 0 0\nCl 1.25 0.5 2.0")
    structure = _load(tmp_path, text)
    beta = math.radians(104.5)
    expected = [[3, 0, 0], [0, 4, 0], [5 * math.cos(beta), 0, 5 * math.sin(beta)]]
    basis = structure.cell.basis.to_floats()
    assert all(abs(basis[i][j] - expected[i][j]) < 1e-9 for i in range(3) for j in range(3))
    reduced = structure.sites.reduced_coords.to_fractions()[1]
    exact_basis = structure.cell.basis.coefficient(1).to_fractions()
    cartesian = [sum(reduced[i] * exact_basis[i][j] for i in range(3)) for j in range(3)]
    assert cartesian == [Fraction("1.25"), Fraction("0.5"), 2]
    assert structure.cell.precision == Fraction(1, 100)
    path = tmp_path / "geom.inp"
    save(structure, str(path), format="cp2k-input")
    back = load(str(path), format="cp2k-input")
    # The writer renders the rational basis and coordinates to 16 significant digits.
    for row, original in zip(back.cell.basis.to_floats(), basis):
        assert row == pytest.approx(original, abs=1e-14)
    for row, original in zip(back.sites.reduced_coords.to_floats(), structure.sites.reduced_coords.to_floats()):
        assert row == pytest.approx(original, abs=1e-14)


@pytest.mark.parametrize(
    ("cell", "match"),
    [
        ("ABC 3 3 3\nALPHA_BETA_GAMMA 60 60 120", "degenerate"),
        ("ABC 3 0 3", "positive"),
        ("ABC 3 3 3\nALPHA_BETA_GAMMA [rad] 1 1 1", r"unit \[rad\]"),
        ("A [bohr] 4 0 0\nB 0 4 0\nC 0 0 4", r"unit \[bohr\]"),
        ("ABC 3 3 3\nA 3 0 0", "both ABC and A/B/C"),
        ("A 4 0 0\nB 0 4 0", "needs A, B and C"),
        ("A 4 0 0\nA 4 0 0\nB 0 4 0\nC 0 0 4", "repeated"),
        ("A 4 0\nB 0 4 0\nC 0 0 4", "three numbers"),
        ("A 4 0 x\nB 0 4 0\nC 0 0 4", "decimal number"),
        ("ABC 4 4 4\nMULTIPLE_UNIT_CELL 2 2 2", "MULTIPLE_UNIT_CELL 2 2 2"),
        ("ABC 4 4 4\nSYMMETRY CUBIC", "SYMMETRY CUBIC"),
        ("CELL_FILE_NAME cell.cell", "CELL_FILE_NAME"),
        ("ABC 4 4 4\n&CELL_REF\nABC 4 4 4\n&END CELL_REF", "CELL_REF"),
        ("ABC 4 4 4\nPERIODIC XX", "PERIODIC"),
    ],
)
def test_cell_refusals(tmp_path: Path, cell: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, _input(cell, "Na 0 0 0"))


@pytest.mark.parametrize(
    ("periodic", "expected"),
    [
        ("NONE", (False, False, False)),
        ("XY", (True, True, False)),
        ("xz", (True, False, True)),
        ("Z", (False, False, True)),
    ],
)
def test_periodic(tmp_path: Path, periodic: str, expected: tuple[bool, ...]) -> None:
    structure = _load(tmp_path, _input(f"ABC 4 4 4\nPERIODIC {periodic}"))
    assert structure.cell.periodicity == expected


def test_units_comments_case_and_continuation(tmp_path: Path) -> None:
    text = (
        "# leading comment\n&global\n&end\n&force_eval ! inline\n &subsys\n  &cell\n"
        "   a [Angstrom] 4 0 0 ! vector a\n   B 0 4 \\\n     0\n   c 0 0 4 # vector c\n  &end cell\n"
        "  &coord\tignored\n   unit angstrom\n   Na 0 0 0 MOL1\n   Cl 2 2 2 MOL1\n  &end coord\n &end subsys\n&end force_eval\n"
    )
    structure = _load(tmp_path, text)
    assert structure.cell.basis == SurdVector([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    assert structure.sites.reduced_coords.to_fractions() == [[0, 0, 0], [Fraction(1, 2)] * 3]
    with pytest.raises(ValueError, match="UNIT bohr"):
        _load(tmp_path, _input(coord="UNIT bohr\nNa 0 0 0"))
    with pytest.raises(ValueError, match="kind x y z"):
        _load(tmp_path, _input(coord="Na 0 0 0 MOL RES"))
    with pytest.raises(ValueError, match="logical"):
        _load(tmp_path, _input(coord="SCALED maybe\nNa 0 0 0"))
    with pytest.raises(ValueError, match="no atoms"):
        _load(tmp_path, _input(coord="SCALED F"))


def test_kind_names_are_case_insensitive(tmp_path: Path) -> None:
    structure = _load(tmp_path, _input(coord="si 0 0 0\nSI 2 2 2", extra="&KIND Si\n ELEMENT Si\n&END KIND"))
    assert [item.name for item in structure.species] == ["si"]
    assert list(structure.species_at_sites) == ["si", "si"]
    clash = _structure_from_geometry(
        {
            "format": "cp2k-input",
            "cell": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            "cell_unit": "angstrom",
            "positions": [[0, 0, 0], ["0.5", "0.5", "0.5"]],
            "positions_kind": "fractional",
            "species": [{"name": "x1", "element": "Fe", "mass": None}, {"name": "X1", "element": "Ni", "mass": None}],
            "species_at_sites": ["x1", "X1"],
        }
    )
    with pytest.raises(ValueError, match="letter case"):
        save(clash, str(tmp_path / "geom.inp"), format="cp2k-input")


def test_kind_elements(tmp_path: Path) -> None:
    kinds = "&KIND Fe1\n ELEMENT Fe\n&END KIND\n&KIND si_up\n BASIS_SET DZVP\n&END KIND"
    structure = _load(tmp_path, _input(coord="Fe1 0 0 0\nSi_up 2 2 2\nNa 1 1 1", extra=kinds))
    assert [(item.name, item.chemical_symbols[0]) for item in structure.species] == [
        ("Fe1", "Fe"),
        ("Si_up", "Si"),
        ("Na", "Na"),
    ]
    assert list(structure.species_at_sites) == ["Fe1", "Si_up", "Na"]
    with pytest.raises(ValueError, match="cannot determine the element of kind 'Qq1'"):
        _load(tmp_path, _input(coord="Qq1 0 0 0", extra="&KIND Qq1\n&END KIND"))
    with pytest.raises(ValueError, match="ELEMENT 'Qq' is not an element"):
        _load(tmp_path, _input(coord="A1 0 0 0", extra="&KIND A1\n ELEMENT Qq\n&END KIND"))
    with pytest.raises(ValueError, match="ghost"):
        _load(tmp_path, _input(coord="H 0 0 0", extra="&KIND H\n GHOST\n&END KIND"))
    with pytest.raises(ValueError, match="unique kind name"):
        _load(tmp_path, _input(coord="H 0 0 0", extra="&KIND H\n&END KIND\n&KIND h\n&END KIND"))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("@INCLUDE 'subsys.inc'\n" + _input(), "@INCLUDE"),
        ("@SET A 4\n" + _input(), "@SET"),
        (_input("ABC ${A} 4 4"), r"variables"),
        ("@IF 1\n" + _input() + "@ENDIF\n", "@IF"),
        (_input(extra="&TOPOLOGY\n COORD_FILE_NAME si.xyz\n COORD_FILE_FORMAT XYZ\n&END TOPOLOGY"), "COORD_FILE_NAME"),
        (_input(extra="&TOPOLOGY\n COORD_FILE_FORMAT XYZ\n&END TOPOLOGY"), "COORD_FILE_FORMAT"),
        (_input(extra="&TOPOLOGY\n MULTIPLE_UNIT_CELL 1 1 2\n&END TOPOLOGY"), "MULTIPLE_UNIT_CELL 1 1 2"),
        (_input() + _input().split("&END GLOBAL\n")[1], "several &FORCE_EVAL"),
        (_input(extra="&END SUBSYS\n&SUBSYS\n&CELL\nABC 1 1 1\n&END CELL"), "more than one &SUBSYS"),
        ("&GLOBAL\n PROJECT x\n&END GLOBAL\n", "not a CP2K input: no &FORCE_EVAL/&SUBSYS"),
        (_input().replace("&END FORCE_EVAL\n", ""), "never closed"),
        (_input().replace("&END COORD", "&END CELL"), "closes &COORD"),
        ("&END\n", "without an open section"),
        ("stray\n" + _input(), "outside any"),
    ],
)
def test_input_refusals(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, text)


def test_topology_without_external_coordinates_is_accepted(tmp_path: Path) -> None:
    extra = "&TOPOLOGY\n NUMBER_OF_ATOMS 2\n COORD_FILE_FORMAT OFF\n MULTIPLE_UNIT_CELL 1 1 1\n&END TOPOLOGY"
    assert len(_load(tmp_path, _input(extra=extra)).species_at_sites) == 2


def test_non_cp2k_content_is_refused(tmp_path: Path) -> None:
    orca = "! B3LYP def2-SVP Opt\n%pal nprocs 4 end\n* xyz 0 1\nH 0 0 0\nH 0 0 0.74\n*\n$new_job\n"
    with pytest.raises(ValueError, match="not a CP2K input"):
        _load(tmp_path, orca)
    binary = tmp_path / "lammps.restart"
    binary.write_bytes(bytes(range(256)) * 8)
    with pytest.raises(ValueError, match="not a CP2K input"):
        load(str(binary), format="cp2k-input")
    with pytest.raises(TypeError, match="path or a stream"):
        read_cp2k_input(["&SUBSYS\n"])


def test_streams_precision_and_compression(tmp_path: Path) -> None:
    payload = read_cp2k_input(io.StringIO(SI.read_text()), precision=1e-3)
    assert payload["precision_override"] == 1e-3
    assert payload["raw"] == SI.read_text()
    assert read_cp2k_input(io.BytesIO(SI.read_bytes()))["raw"] == SI.read_text()
    with pytest.raises(ValueError):
        read_cp2k_input(SI, precision=-1.0)
    compressed = tmp_path / "si.inp.gz"
    compressed.write_bytes(gzip.compress(SI.read_bytes()))
    structure = load(str(compressed), format="cp2k-input")
    assert [tuple(row) for row in structure.sites.reduced_coords.to_fractions()] == DIAMOND


def test_save_load_round_trip_is_exact(tmp_path: Path) -> None:
    source = _load(
        tmp_path, _input("ABC 4 4 5\nPERIODIC XY", "Fe1 0 0 0\nO 2 2 2.5", "&KIND Fe1\n ELEMENT Fe\n&END KIND")
    )
    path = tmp_path / "geom.inp"
    save(source, str(path), format="cp2k-input")
    text = path.read_text()
    assert text.startswith("&SUBSYS\n") and text.endswith("&END SUBSYS\n")
    assert "    PERIODIC XY\n" in text and "    SCALED .TRUE.\n" in text
    assert "  &KIND Fe1\n    ELEMENT Fe\n  &END KIND\n" in text and "&KIND O" not in text
    back = load(str(path), format="cp2k-input")
    assert back.cell.basis == source.cell.basis
    assert back.sites.reduced_coords == source.sites.reduced_coords
    assert back.cell.periodicity == (True, True, False)
    assert [(item.name, item.chemical_symbols) for item in back.species] == [("Fe1", ("Fe",)), ("O", ("O",))]
    assert list(back.species_at_sites) == ["Fe1", "O"]
    si = load(str(SI), format="cp2k-input")
    save(si, str(path), format="cp2k-input")
    assert load(str(path), format="cp2k-input").sites.reduced_coords == si.sites.reduced_coords


def test_writer_normalizes_cartesian_payload_and_refuses_bad_kind_names(tmp_path: Path) -> None:
    payload = load(str(RESTART), format="cp2k-input", raw=True)
    del payload["raw"]
    path = tmp_path / "geom.inp"
    save(payload, str(path), format="cp2k-input")
    assert "    O 0.5 0.5 0.5\n" in path.read_text()
    payload["species"] = [{"name": "Fe 1", "element": "Fe", "mass": None}, {"name": "O", "element": "O", "mass": None}]
    payload["species_at_sites"] = ["Fe 1", "O"]
    with pytest.raises(ValueError, match="kind name"):
        save(payload, str(path), format="cp2k-input")


def test_raw_round_trip_is_byte_identical(tmp_path: Path) -> None:
    for source in (SI, RESTART):
        payload = load(str(source), format="cp2k-input", raw=True)
        path = tmp_path / source.name
        save(payload, str(path), format="cp2k-input")
        assert path.read_bytes() == source.read_bytes()
        with pytest.raises(ValueError, match="raw"):
            save(payload, str(path), format="cp2k-input", indent=2)


def test_ase_reads_the_restart_fixture_alike(tmp_path: Path) -> None:
    ase_io = pytest.importorskip("ase.io")
    numpy = pytest.importorskip("numpy")
    # ASE 3.29's cp2k-restart reader takes every &COORD line as Cartesian Å, so it cannot read
    # the writer's SCALED output; compare on Cartesian restart-style files instead.
    triclinic = _input("A 4 0 0\nB 1 3.5 0\nC 0.5 0.25 5", "Si 0.1 0.2 0.3\nO 1.25 2.5 3.75")
    path = tmp_path / "tri.restart"
    path.write_text(triclinic)
    for source in (RESTART, path):
        atoms = ase_io.read(str(source), format="cp2k-restart")
        structure = load(str(source), format="cp2k-input")
        assert numpy.allclose(atoms.cell[:], structure.cell.basis.to_floats(), rtol=0, atol=1e-8)
        cartesian = (structure.sites.reduced_coords * structure.cell.basis).to_floats()
        assert numpy.allclose(atoms.positions, cartesian, rtol=0, atol=1e-8)
        assert atoms.get_chemical_symbols() == [
            structure.species[[item.name for item in structure.species].index(name)].chemical_symbols[0]
            for name in structure.species_at_sites
        ]
