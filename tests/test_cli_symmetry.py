"""Tests for the ``httk symmetry`` command-line interface."""

import subprocess
import sys
from pathlib import Path

import pytest
from httk.core import CLIContext, FracVector, load, save

from httk.atomistic import ASUStructure, Cell, Species, WyckoffSite
from httk.atomistic.cli import command
from httk.atomistic.symmetry.lift import canonicalize


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _rocksalt() -> ASUStructure:
    return ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        225,
        [WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("b", FracVector(()), "Cl")],
        _species("Na", "Cl"),
    )


def _key(structure: ASUStructure) -> tuple[object, ...]:
    metric = structure.cell.metric()
    return (
        structure.spacegroup.it_number,
        tuple(sorted((s.species, s.wyckoff, tuple(s.free_params.to_fractions())) for s in structure.wyckoff_sites)),
        tuple(metric._element((r, c)) for r in range(3) for c in range(3)),
    )


@pytest.fixture
def cif(tmp_path: Path) -> Path:
    path = tmp_path / "nacl.cif"
    save(_rocksalt(), str(path))
    return path


@pytest.fixture
def poscar(tmp_path: Path) -> Path:
    path = tmp_path / "POSCAR"
    save(_rocksalt(), str(path))
    return path


def _run(argv: list[str]) -> int:
    return command(argv, CLIContext(program="httk", cwd=Path.cwd()))


def test_info_on_declared_symmetry(cif: Path, capsys) -> None:
    assert _run(["info", str(cif)]) == 0
    out = capsys.readouterr().out
    assert "input: ASUStructure" in out
    assert "declared space group: IT 225" in out
    assert "F m -3 m" in out
    assert "a=5 b=5 c=5" in out
    assert "a  Na" in out


def test_info_on_unitcell_has_no_declared_symmetry(poscar: Path, capsys) -> None:
    assert _run(["info", str(poscar)]) == 0
    out = capsys.readouterr().out
    assert "input: UnitcellStructure" in out
    assert "none declared" in out


def test_info_recognize(poscar: Path, capsys) -> None:
    pytest.importorskip("spglib")
    assert _run(["info", str(poscar), "--recognize"]) == 0
    out = capsys.readouterr().out
    assert "recognized" in out
    assert "IT 225" in out


def test_canonicalize_default(cif: Path, capsys) -> None:
    pytest.importorskip("spglib")
    assert _run(["canonicalize", str(cif)]) == 0
    out = capsys.readouterr().out
    assert "canonical form (lift=False)" in out
    assert "IT 225" in out


def test_canonicalize_exact_roundtrip(cif: Path, tmp_path: Path, capsys) -> None:
    out_path = tmp_path / "canon.cif"
    assert _run(["canonicalize", str(cif), "--exact", "-o", str(out_path)]) == 0
    assert f"saved: {out_path}" in capsys.readouterr().out
    reloaded = load(str(out_path))
    assert isinstance(reloaded, ASUStructure)
    expected = canonicalize(_rocksalt()).asu
    assert _key(canonicalize(reloaded).asu) == _key(expected)


def test_rerepresent(cif: Path, capsys) -> None:
    assert _run(["rerepresent", str(cif), "--target", "166"]) == 0
    out = capsys.readouterr().out
    assert "re-represented in IT 166" in out
    assert "IT 166" in out


def test_rerepresent_save_poscar(cif: Path, tmp_path: Path, capsys) -> None:
    # A trigonal rerepresentation has an irrational cell that CIF cannot write exactly, but POSCAR
    # (float cell) saves it, proving the -o wiring for rerepresent.
    out_path = tmp_path / "trigonal.POSCAR"
    assert _run(["rerepresent", str(cif), "--target", "166", "-o", str(out_path)]) == 0
    assert f"saved: {out_path}" in capsys.readouterr().out
    assert out_path.is_file()


def test_representations_count_and_letters(cif: Path, capsys) -> None:
    assert _run(["representations", str(cif), "--target", "166"]) == 0
    out = capsys.readouterr().out
    assert "representations in IT 166: 2" in out
    # Both letter assignments of the deterministic pair are present.
    assert "a  Na" in out
    assert "b  Na" in out


def test_unrelated_target_errors(cif: Path, capsys) -> None:
    assert _run(["representations", str(cif), "--target", "191"]) == 2
    assert "unrelated" in capsys.readouterr().err


def test_exact_on_unitcell_errors(poscar: Path, capsys) -> None:
    assert _run(["canonicalize", str(poscar), "--exact"]) == 2
    assert "requires an input with declared symmetry" in capsys.readouterr().err


def test_no_subcommand_prints_help(capsys) -> None:
    assert _run([]) == 0
    assert "Inspect and canonicalize" in capsys.readouterr().out


def test_end_to_end_registration_smoke(cif: Path) -> None:
    # Prove the registry discovers and dispatches the command through the real `httk` console script.
    httk = Path(sys.executable).with_name("httk")
    if not httk.is_file():
        pytest.skip("httk console script is not installed in this environment")
    result = subprocess.run(
        [str(httk), "symmetry", "info", str(cif)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "declared space group: IT 225" in result.stdout
