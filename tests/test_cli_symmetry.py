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


def test_info_batches_and_continues_after_failure(cif: Path, tmp_path: Path, capsys) -> None:
    second = tmp_path / "second.cif"
    save(_rocksalt(), str(second))

    assert _run(["info", str(cif), str(tmp_path / "missing.cif"), str(second)]) == 1
    captured = capsys.readouterr()
    assert captured.out.count("input: ASUStructure") == 2
    assert f"==> {cif} <==" in captured.out
    assert f"==> {second} <==" in captured.out
    assert "missing.cif" in captured.err


def test_info_repairs_cif_and_logs_warning(capsys, caplog) -> None:
    fixture = Path(__file__).parent / "fixtures" / "malformed_auxiliary_loop.cif"
    with caplog.at_level("WARNING", logger="httk"):
        assert command(["info", str(fixture)], CLIContext("httk", Path.cwd())) == 0

    out = capsys.readouterr().out
    assert "input: ASUStructure" in out
    assert "sites: 1" in out
    assert any(
        record.name.startswith("httk")
        and record.levelno >= 30
        and "dropped malformed auxiliary loop" in record.getMessage()
        for record in caplog.records
    )


def test_info_falls_back_for_poscar_reader(tmp_path: Path, capsys) -> None:
    path = tmp_path / "POSCAR"
    path.write_text(
        "NaCl\n1.0\n5 0 0\n0 5 0\n0 0 5\nNa Cl\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n",
        encoding="utf-8",
    )

    assert command(["info", str(path)], CLIContext("httk", tmp_path)) == 0
    assert "input: UnitcellStructure" in capsys.readouterr().out


def test_info_does_not_retry_internal_repair_type_error(monkeypatch, capsys) -> None:
    from httk.atomistic import cli

    calls: list[tuple[str, dict[str, object]]] = []

    def failing_load(filename: str, **options: object) -> object:
        calls.append((filename, options))
        raise TypeError("boom repair failure")

    monkeypatch.setattr(cli, "load", failing_load)

    assert command(["info", "broken.cif"], CLIContext("httk", Path.cwd())) == 1
    assert calls == [("broken.cif", {"repair": True})]
    assert "boom repair failure" in capsys.readouterr().err


def test_info_recognize(poscar: Path, capsys) -> None:
    pytest.importorskip("spglib")
    assert _run(["info", "--recognize", str(poscar)]) == 0
    out = capsys.readouterr().out
    assert "recognized" in out
    assert "IT 225" in out


def test_canonicalize_default(cif: Path, capsys) -> None:
    pytest.importorskip("spglib")
    assert _run(["canonicalize", str(cif)]) == 0
    out = capsys.readouterr().out
    assert "canonical form (lift=False)" in out
    assert "IT 225" in out


def test_preserve_chirality_flag_threads_to_both_branches(cif: Path, monkeypatch, capsys) -> None:
    pytest.importorskip("spglib")  # the default (non-exact) branch recognizes with spglib
    from httk.atomistic import cli

    calls: dict[str, bool] = {}
    real_canonicalize, real_canonical_asu = cli.canonicalize, cli.canonical_asu

    def spy_canonicalize(*args, preserve_chirality=False, **kwargs):
        calls["exact"] = preserve_chirality
        return real_canonicalize(*args, preserve_chirality=preserve_chirality, **kwargs)

    def spy_canonical_asu(*args, preserve_chirality=False, **kwargs):
        calls["default"] = preserve_chirality
        return real_canonical_asu(*args, preserve_chirality=preserve_chirality, **kwargs)

    monkeypatch.setattr(cli, "canonicalize", spy_canonicalize)
    monkeypatch.setattr(cli, "canonical_asu", spy_canonical_asu)

    assert _run(["canonicalize", "--exact", "--preserve-chirality", str(cif)]) == 0  # exact -> canonicalize
    assert calls["exact"] is True
    assert _run(["canonicalize", "--preserve-chirality", str(cif)]) == 0  # default -> canonical_asu
    assert calls["default"] is True
    assert _run(["canonicalize", "--exact", str(cif)]) == 0  # flag absent -> default False
    assert calls["exact"] is False


def test_canonicalize_exact_roundtrip(cif: Path, tmp_path: Path, capsys) -> None:
    out_path = tmp_path / "canon.cif"
    assert _run(["canonicalize", "--exact", "-o", str(out_path), str(cif)]) == 0
    assert f"saved: {out_path}" in capsys.readouterr().out
    reloaded = load(str(out_path))
    assert isinstance(reloaded, ASUStructure)
    expected = canonicalize(_rocksalt()).asu
    assert _key(canonicalize(reloaded).asu) == _key(expected)


def test_canonicalize_batch_out_dir(tmp_path: Path, capsys) -> None:
    first = tmp_path / "first.cif"
    second = tmp_path / "second.cif"
    save(_rocksalt(), str(first))
    save(_rocksalt(), str(second))
    out_dir = tmp_path / "canonical"

    assert _run(["canonicalize", "--exact", "--out-dir", str(out_dir), str(first), str(second)]) == 0
    assert (out_dir / first.name).is_file()
    assert (out_dir / second.name).is_file()
    assert capsys.readouterr().out.count("saved:") == 2


def test_single_output_rejects_multiple_inputs(cif: Path, tmp_path: Path, capsys) -> None:
    assert _run(["canonicalize", "-o", str(tmp_path / "one.cif"), str(cif), str(cif)]) == 2
    assert "requires exactly one FILE" in capsys.readouterr().err


def test_rerepresent(cif: Path, capsys) -> None:
    assert _run(["rerepresent", "--target", "166", str(cif)]) == 0
    out = capsys.readouterr().out
    assert "re-represented in IT 166" in out
    assert "IT 166" in out


def test_rerepresent_save_poscar(cif: Path, tmp_path: Path, capsys) -> None:
    # A trigonal rerepresentation has an irrational cell that CIF cannot write exactly, but POSCAR
    # (float cell) saves it, proving the -o wiring for rerepresent.
    out_path = tmp_path / "trigonal.POSCAR"
    assert _run(["rerepresent", "--target", "166", "-o", str(out_path), str(cif)]) == 0
    assert f"saved: {out_path}" in capsys.readouterr().out
    assert out_path.is_file()


def test_representations_count_and_letters(cif: Path, capsys) -> None:
    assert _run(["representations", "--target", "166", str(cif)]) == 0
    out = capsys.readouterr().out
    assert "representations in IT 166: 2" in out
    # Both letter assignments of the deterministic pair are present.
    assert "a  Na" in out
    assert "b  Na" in out


def test_unrelated_target_errors(cif: Path, capsys) -> None:
    assert _run(["representations", "--target", "191", str(cif)]) == 1
    assert "unrelated" in capsys.readouterr().err


def test_exact_on_unitcell_errors(poscar: Path, capsys) -> None:
    assert _run(["canonicalize", "--exact", str(poscar)]) == 1
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


def test_module_reports_repair_warning_to_stderr() -> None:
    fixture = Path(__file__).parent / "fixtures" / "malformed_auxiliary_loop.cif"
    argv = [sys.executable, "-m", "httk.atomistic.cli", "info", str(fixture)]
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "dropped malformed auxiliary loop" in result.stderr
    assert "input: ASUStructure" in result.stdout

    merged = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert merged.returncode == 0, merged.stdout
    assert merged.stdout.index("dropped malformed auxiliary loop") < merged.stdout.index("input: ASUStructure")
