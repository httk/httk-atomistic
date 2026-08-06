"""Tests for the explicit lazy VASP structure integration."""

import bz2
from pathlib import Path

import httk.core
import pytest

from httk.atomistic import Species, UnitcellStructure, UnitcellStructureView, VASPStructure

pytest.importorskip("httk.io")

POSCAR = """Synthetic POSCAR
1.0
2.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 4.0
He
1
Direct
0.0 0.0 0.0
"""

SELECTIVE_POSCAR = """Selective POSCAR
1.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
He
1
Selective dynamics
Direct
0.0 0.0 0.0 T F T
"""


def _assert_geometry(view: UnitcellStructureView, basis: list[list[float]] | None = None) -> None:
    assert view.cell.basis.to_floats() == (basis or [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    assert view.sites.reduced_coords.to_floats() == [[0.0, 0.0, 0.0]]
    assert tuple(species.name for species in view.species) == ("He",)
    assert view.species_at_sites == ("He",)


@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_poscar_view_recovery_and_byte_exact_save(tmp_path: Path, line_ending: str) -> None:
    raw = POSCAR.replace("\n", line_ending).encode()
    source = tmp_path / "POSCAR"
    source.write_bytes(raw)

    backend = VASPStructure(source)
    view = UnitcellStructureView(backend)
    _assert_geometry(view)
    assert VASPStructure(view) is backend
    assert httk.core.unwrap(backend) is source

    for name, obj in (("backend.vasp", backend), ("view.vasp", view)):
        destination = tmp_path / name
        httk.core.save(obj, destination)
        assert destination.read_bytes() == raw


def test_compressed_poscar_preserves_decompressed_bytes(tmp_path: Path) -> None:
    raw = POSCAR.replace("\n", "\r\n").encode()
    source = tmp_path / "POSCAR.bz2"
    source.write_bytes(bz2.compress(raw))

    backend = VASPStructure(source)
    _assert_geometry(UnitcellStructureView(backend))
    destination = tmp_path / "decompressed.vasp"
    httk.core.save(backend, destination)
    assert destination.read_bytes() == raw


def test_poscar_is_lazy_and_preserves_extra_payload_fields(tmp_path: Path) -> None:
    invalid = POSCAR.replace("0.0 0.0 0.0", "not-a-number 0.0 0.0")
    source = tmp_path / "invalid.vasp"
    source.write_text(invalid, encoding="utf-8")

    backend = VASPStructure(source)
    with pytest.raises(ValueError, match="Invalid literal"):
        backend.resolve()

    selective_source = tmp_path / "selective.vasp"
    selective_source.write_text(SELECTIVE_POSCAR, encoding="utf-8")
    selective = VASPStructure(selective_source)
    assert selective.comment == "Selective POSCAR"
    assert selective.selective_dynamics == [[True, False, True]]
    assert selective.unwrap() is selective_source


def test_foreign_structure_uses_token_serializer_path(tmp_path: Path) -> None:
    structure = UnitcellStructure(
        [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        [[0, 0, 0]],
        [Species("He", ("He",), (1,))],
        ["He"],
    )
    backend = VASPStructure(structure)
    assert backend.payload.get("raw") is None

    destination = tmp_path / "foreign.vasp"
    httk.core.save(backend, destination)
    loaded = httk.core.load(str(destination))
    _assert_geometry(UnitcellStructureView(loaded), [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])


def test_missing_poscar_reader_names_httk_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "POSCAR"
    source.write_text(POSCAR, encoding="utf-8")
    monkeypatch.setattr(httk.core, "has_reader_for", lambda name: False)

    with pytest.raises(ImportError, match="httk-io"):
        VASPStructure(source)
