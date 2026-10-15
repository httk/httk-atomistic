"""LAMMPS data file reader and writer checks."""

import gzip
from fractions import Fraction
from pathlib import Path

import pytest
from httk.core import load as _load
from httk.core import save
from httk.core.loading import has_reader_for

from httk.atomistic import UnitcellStructureView
from httk.atomistic.integrations.lammps.io import read_lammps_data

ORTHO = """test cell

2 atoms
2 atom types

-1.0 3.0 xlo xhi
0.5 4.5 ylo yhi
0 5 zlo zhi

Masses

1 28.0855 # Si
2 15.999 # O

Atoms # atomic

2 2 1.5 2.5 2.5 1 0 0
1 1 -1.0 0.5 0.0

Velocities

1 0 0 0
2 0 0 0
"""
TILTED = """tilted

1 atoms
1 atom types

0 4.0 xlo xhi
0 3.0 ylo yhi
0 5.0 zlo zhi
1.0 0.5 -0.25 xy xz yz

Atom Type Labels

1 Fe1

Atoms

1 1 1.0 1.0 1.0
"""


def load(path: str, **kwargs: object) -> object:
    """Load with the explicit format (the reader claims no extension)."""
    if not path.endswith("POSCAR"):
        kwargs.setdefault("format", "lammps-data")
    return _load(path, **kwargs)


def _write(tmp_path: Path, text: str, name: str = "x.lmp") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _rows(structure: object) -> list[list[Fraction]]:
    return [[Fraction(x) for x in row] for row in UnitcellStructureView(structure).lattice_vectors]


def test_orthogonal_with_nonzero_origin_and_masses_comments(tmp_path: Path) -> None:
    structure = load(_write(tmp_path, ORTHO))
    assert _rows(structure) == [[4, 0, 0], [0, 4, 0], [0, 0, 5]]
    assert list(structure.species_at_sites) == ["Si", "O"]
    assert [item.mass for item in structure.species] == [(28.0855,), (15.999,)]
    assert [list(row) for row in structure.sites.reduced_coords.to_fractions()] == [
        [0, 0, 0],
        [Fraction(5, 8), Fraction(1, 2), Fraction(1, 2)],
    ]
    cart = UnitcellStructureView(structure).cartesian_site_positions
    assert [[Fraction(x) for x in row] for row in cart] == [[0, 0, 0], [Fraction(5, 2), 2, Fraction(5, 2)]]


def test_triclinic_tilts_and_atom_type_labels(tmp_path: Path) -> None:
    structure = load(_write(tmp_path, TILTED))
    assert _rows(structure) == [[4, 0, 0], [1, 3, 0], [Fraction(1, 2), Fraction(-1, 4), 5]]
    assert list(structure.species_at_sites) == ["Fe1"]
    assert structure.species[0].chemical_symbols == ("Fe",)


def test_header_comment_species_and_precedence(tmp_path: Path) -> None:
    text = (
        ORTHO.replace("test cell", "LAMMPS data file written by httk-workflow-lammps; types: Ge Sn")
        .replace(" # Si", "")
        .replace(" # O", "")
    )
    path = _write(tmp_path, text)
    assert list(load(path).species_at_sites) == ["Ge", "Sn"]
    assert list(load(path, species={1: "Si", 2: "O"}).species_at_sites) == ["Si", "O"]
    assert list(load(path, species=["Al", "N"]).species_at_sites) == ["Al", "N"]
    # explicit species= beats Masses comments
    assert list(load(_write(tmp_path, ORTHO, "y.lmp"), species=["Al", "N"]).species_at_sites) == ["Al", "N"]


def test_species_refused_when_unknown(tmp_path: Path) -> None:
    path = _write(tmp_path, ORTHO.replace(" # Si", "").replace(" # O", ""))
    with pytest.raises(ValueError, match="pass species"):
        load(path)
    with pytest.raises(ValueError, match="element symbol"):
        load(path, species={1: "Xx", 2: "O"})
    with pytest.raises(ValueError, match="sequence of 2"):
        load(path, species=["Si"])


def test_refusals(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="atom style"):
        load(_write(tmp_path, ORTHO.replace("# atomic", "# full")))
    with pytest.raises(ValueError, match="5 columns"):
        load(_write(tmp_path, ORTHO.replace("1 1 -1.0 0.5 0.0", "1 1 0 -1.0 0.5 0.0")))
    with pytest.raises(ValueError, match="5 columns"):
        load(_write(tmp_path, ORTHO.replace("# atomic", "")))  # 8 columns need the style comment
    with pytest.raises(ValueError, match="declares 3 atoms"):
        load(_write(tmp_path, ORTHO.replace("2 atoms", "3 atoms")))
    with pytest.raises(ValueError, match="no Atoms"):
        load(_write(tmp_path, "c\n\n1 atoms\n1 atom types\n0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n"))
    for units in ("lj", "si", "cgs", "electron", "micro", "nano"):
        with pytest.raises(ValueError, match="not supported"):
            read_lammps_data(_write(tmp_path, ORTHO), units=units)
    with pytest.raises(ValueError, match="units must be"):
        read_lammps_data(_write(tmp_path, ORTHO), units="bogus")
    assert read_lammps_data(_write(tmp_path, ORTHO), units="real")["format"] == "lammps-data"


def test_general_triclinic_header(tmp_path: Path) -> None:
    text = TILTED.replace(
        "0 4.0 xlo xhi\n0 3.0 ylo yhi\n0 5.0 zlo zhi\n1.0 0.5 -0.25 xy xz yz",
        "4 0 0 avec\n1 3 0 bvec\n0.5 -0.25 5 cvec\n1 1 1 abc origin",
    )
    structure = load(_write(tmp_path, text))
    assert _rows(structure) == [[4, 0, 0], [1, 3, 0], [Fraction(1, 2), Fraction(-1, 4), 5]]
    assert [Fraction(x) for x in UnitcellStructureView(structure).cartesian_site_positions[0]] == [0, 0, 0]


def test_exact_roundtrip_orthogonal_and_framed_triclinic(tmp_path: Path) -> None:
    structure = load(_write(tmp_path, ORTHO))
    out = str(tmp_path / "o.lmp")
    save(structure, out)
    again = load(out)
    assert _rows(again) == _rows(structure)
    assert list(again.species_at_sites) == ["Si", "O"]
    assert again.sites.reduced_coords.to_fractions() == structure.sites.reduced_coords.to_fractions()
    tri = load(_write(tmp_path, TILTED))
    out = str(tmp_path / "t.lmp")
    save(tri, out)
    assert "xy xz yz" in Path(out).read_text()
    again = load(out)
    assert _rows(again) == _rows(tri)
    assert again.sites.reduced_coords.to_fractions() == tri.sites.reduced_coords.to_fractions()


def test_unframed_cell_roundtrip_is_lossy_but_close(tmp_path: Path) -> None:
    # A hexagonal cell rotated 30 degrees about z is not LAMMPS-framed, so save rotates it in floats.
    text = (
        "hex\n1.0\n 2.598076211353316 1.5 0.0\n -2.598076211353316 1.5 0.0\n 0.0 0.0 5.0\n"
        "Si\n1\nDirect\n 0.25 0.5 0.75\n"
    )
    source = tmp_path / "POSCAR"
    source.write_text(text, encoding="utf-8")
    structure = load(str(source), precision=1e-6)
    out = str(tmp_path / "h.lmp")
    save(structure, out)
    again = load(out)
    mine, theirs = _rows(structure), _rows(again)
    gram = lambda rows: [sum(float(x) * float(y) for x, y in zip(u, v)) for u in rows for v in rows]
    assert all(abs(p - q) < 1e-12 for p, q in zip(gram(mine), gram(theirs)))
    assert all(
        abs(float(x) - float(y)) < 1e-12
        for u, v in zip(structure.sites.reduced_coords.to_fractions(), again.sites.reduced_coords.to_fractions())
        for x, y in zip(u, v)
    )


def test_left_handed_cell_refused(tmp_path: Path) -> None:
    text = "left\n1.0\n -2.5 0.0 0.0\n 0.0 3.0 0.0\n 0.0 0.0 4.0\nSi\n1\nDirect\n 0.25 0.5 0.75\n"
    source = tmp_path / "POSCAR"
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="left-handed"):
        save(load(str(source), precision=1e-6), str(tmp_path / "l.lmp"))


def test_raw_roundtrip_and_kwargs(tmp_path: Path) -> None:
    path = _write(tmp_path, ORTHO)
    payload = load(path, raw=True)
    assert payload["raw"] == ORTHO
    out = tmp_path / "copy.lmp"
    save(payload, str(out))
    assert out.read_bytes() == Path(path).read_bytes()
    with pytest.raises(ValueError, match="raw"):
        save(payload, str(out), masses={"Si": 1.0})


def test_writer_options_and_masses(tmp_path: Path) -> None:
    structure = load(_write(tmp_path, ORTHO))
    out = str(tmp_path / "m.lmp")
    save(structure, out, masses={"Si": 1.5, "O": 2.5})
    assert "1 1.5 # Si" in Path(out).read_text()
    with pytest.raises(ValueError, match="no mass"):
        save(structure, out, masses={"Si": 1.5})
    with pytest.raises(ValueError, match="atom_style"):
        save(structure, out, atom_style="full")


def test_format_spelling_and_compression(tmp_path: Path) -> None:
    path = _write(tmp_path, ORTHO, "data.si")
    assert list(load(path, format="lammps-data").species_at_sites) == ["Si", "O"]
    with pytest.raises(ValueError, match="Could not determine"):
        _load(path)
    assert not has_reader_for("x.lmp")  # the reader claims no extension; save("x.lmp") still works
    zipped = tmp_path / "z.lmp.gz"
    zipped.write_bytes(gzip.compress(ORTHO.encode()))
    assert list(load(str(zipped)).species_at_sites) == ["Si", "O"]


def test_ase_reads_our_file_and_we_read_ase(tmp_path: Path) -> None:
    ase_io = pytest.importorskip("ase.io")
    structure = load(_write(tmp_path, TILTED.replace("Atoms", "Atoms # atomic")))
    out = str(tmp_path / "a.lmp")
    save(structure, out)
    atoms = ase_io.read(out, format="lammps-data", atom_style="atomic")
    mine = UnitcellStructureView(structure)
    assert all(abs(float(x) - y) < 1e-8 for r, s in zip(mine.lattice_vectors, atoms.cell[:]) for x, y in zip(r, s))
    assert all(
        abs(float(x) - y) < 1e-8 for r, s in zip(mine.cartesian_site_positions, atoms.positions) for x, y in zip(r, s)
    )
    from ase import Atoms

    built = Atoms("Si2", positions=[[0, 0, 0], [1.3, 1.4, 1.5]], cell=[[4, 0, 0], [1, 3, 0], [0.5, 0.25, 5]], pbc=True)
    ase_out = str(tmp_path / "ase.lmp")
    ase_io.write(ase_out, built, format="lammps-data", atom_style="atomic", masses=True)
    ours = load(ase_out, species={1: "Si"})
    assert all(
        abs(float(x) - y) < 1e-8
        for r, s in zip(UnitcellStructureView(ours).lattice_vectors, built.cell[:])
        for x, y in zip(r, s)
    )
    assert all(
        abs(float(x) - y) < 1e-8
        for r, s in zip(UnitcellStructureView(ours).cartesian_site_positions, built.positions)
        for x, y in zip(r, s)
    )


def test_raw_of_newline_less_lines_roundtrips(tmp_path: Path) -> None:
    payload = read_lammps_data(ORTHO.splitlines())
    assert payload["raw"] == ORTHO
    out = str(tmp_path / "r.lmp")
    save(payload, out)
    assert _rows(load(out)) == _rows(load(_write(tmp_path, ORTHO)))
