"""Tests for the lazy VASP directory composite."""

import bz2
from pathlib import Path

import httk.core
import pytest

from httk.atomistic.integrations.vasp.io import OutcarFile, VASPOutputs

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
