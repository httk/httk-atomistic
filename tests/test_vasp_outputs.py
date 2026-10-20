"""Tests for the lazy VASP directory composite."""

import bz2
from fractions import Fraction
from pathlib import Path

import httk.core
import pytest

from httk.atomistic import VASPTrajectory
from httk.atomistic._loading import _structure_from_poscar
from httk.atomistic.integrations.vasp.io import OutcarFile, VASPOutputs, XdatcarFile

POSCAR = """synthetic POSCAR
1.0
1 0 0
0 1 0
0 0 1
Si
1
Direct
0 0 0
"""
OSZICAR = "  1 F= -.10000 E0= -.20000 d E =-.00100\n"
OUTCAR = " vasp.5.2.12 synthetic\n General timing and accounting informations\n"


def test_outputs_lazy_resolution_and_suffixes(tmp_path: Path) -> None:
    (tmp_path / "POSCAR").write_text(POSCAR, encoding="utf-8")
    (tmp_path / "CONTCAR.bz2").write_bytes(bz2.compress(POSCAR.encode("utf-8")))
    (tmp_path / "OUTCAR.bz2").write_bytes(bz2.compress(OUTCAR.encode("utf-8")))
    (tmp_path / "OSZICAR").write_text(OSZICAR, encoding="utf-8")
    outputs = VASPOutputs(tmp_path)
    assert outputs.poscar is not None
    assert outputs.contcar is not None
    assert isinstance(outputs.outcar, OutcarFile)
    assert outputs.xdatcar is None
    assert outputs.oszicar is not None
    assert outputs.potcar is None
    loaded = httk.core.load(str(tmp_path / "OSZICAR"), raw=True)
    assert loaded["format"] == "vasp-oszicar"
    assert not outputs.closed


def test_outputs_owns_file_objects_and_close_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "OUTCAR").write_text(OUTCAR, encoding="utf-8")
    with VASPOutputs(tmp_path) as outputs:
        outcar = outputs.outcar
        assert outcar is not None
    assert outputs.closed
    assert outcar is not None and outcar.closed
    outputs.close()
    with pytest.raises(ValueError, match="closed VASP outputs"):
        _ = outputs.outcar


def test_outputs_without_precision_warns_and_omits_override(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "CONTCAR").write_text(POSCAR, encoding="utf-8")
    outputs = VASPOutputs(tmp_path)
    with caplog.at_level("WARNING"):
        payload = outputs.contcar
    assert payload is not None and payload["precision_override"] is None
    records = [r for r in caplog.records if getattr(r, "context", None) == "poscar"]
    assert records, "reading a CONTCAR without a precision should warn"


def test_outputs_precision_propagates_to_reads(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "POSCAR").write_text(POSCAR, encoding="utf-8")
    (tmp_path / "CONTCAR").write_text(POSCAR, encoding="utf-8")
    outputs = VASPOutputs(tmp_path, precision=5e-4)
    with caplog.at_level("WARNING"):
        assert outputs.poscar is not None and outputs.poscar["precision_override"] == 5e-4
        assert outputs.contcar is not None and outputs.contcar["precision_override"] == 5e-4
    assert not [r for r in caplog.records if getattr(r, "context", None) == "poscar"]


DIAMOND = """diamond
3.567
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
C
2
Direct
0.00 0.00 0.00
0.25 0.25 0.25
"""


def test_outputs_as_written_propagates_to_reads(tmp_path: Path) -> None:
    (tmp_path / "CONTCAR").write_text(DIAMOND, encoding="utf-8")
    exact = VASPOutputs(tmp_path, precision=5e-4, as_written=True)
    assert exact.as_written is True
    assert _structure_from_poscar(exact.contcar).cell.scale.fractions == Fraction(3567, 1000)
    snapped = VASPOutputs(tmp_path, precision=5e-4).contcar
    assert _structure_from_poscar(snapped).cell.scale.fractions == Fraction(346, 97)


@pytest.mark.parametrize("with_poscar", (True, False))
def test_as_written_trajectory_frames_match_header(tmp_path: Path, with_poscar: bool) -> None:
    if with_poscar:
        (tmp_path / "POSCAR").write_text(DIAMOND, encoding="utf-8")
    (tmp_path / "XDATCAR").write_text(DIAMOND.replace("Direct\n", "Direct configuration= 1\n"), encoding="utf-8")
    with VASPOutputs(tmp_path, precision=5e-4, as_written=True) as outputs:
        trajectory = VASPTrajectory(outputs)
        frame = trajectory.frame(0)
        assert frame.cell.basis == trajectory._header_structure().cell.basis
        assert frame.cell.basis.fractions[0][1] == Fraction(3567, 2000)
        assert frame.sites.reduced_coords.fractions[1] == (Fraction(1, 4),) * 3


def test_outputs_resolve_xdatcar(tmp_path: Path) -> None:
    xdatcar = POSCAR.replace("Si\n1\nDirect\n0 0 0\n", "Si\n1\nDirect configuration= 1\n0 0 0\n")
    (tmp_path / "XDATCAR").write_text(xdatcar, encoding="utf-8")
    with VASPOutputs(tmp_path) as outputs:
        assert isinstance(outputs.xdatcar, XdatcarFile)
        assert outputs.xdatcar.nframes == 1
