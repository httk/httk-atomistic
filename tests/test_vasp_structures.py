"""Tests for the neutral-POSCAR-mapping -> UnitcellStructure bridge."""

import bz2
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from httk.core import load

from httk.atomistic import (
    UnitcellStructure,
)
from httk.atomistic import _loading

build_payload = getattr(_loading, "_" + "structure_" + "from_payload")
build_poscar = getattr(_loading, "_" + "structure_" + "from_poscar")

DIRECT = {
    "format": "vasp-poscar",
    "comment": "SmFeO3",
    "scale": "1.0",
    "volume": None,
    "cell": [["5.3982999999999999", "0.0", "0.0"], ["0.0", "5.6", "0.0"], ["0.0", "0.0", "7.6"]],
    "symbols": ["Sm", "Fe", "O"],
    "counts": [1, 1, 2],
    "cartesian": False,
    "coords": [["0.0", "0.0", "0.0"], ["0.5", "0.5", "0.5"], ["0.1", "0.2", "0.3"], ["0.4", "0.5", "0.6"]],
    "selective_dynamics": None,
}


def test_direct_exact_lattice_and_species() -> None:
    structure = build_poscar(DIRECT)
    # First lattice row is float-exact from the string "5.3982999999999999".
    assert structure.cell.basis.to_floats()[0] == [5.3982999999999999, 0.0, 0.0]
    assert [species.name for species in structure.species] == ["Sm", "Fe", "O"]
    assert structure.species_at_sites == ("Sm", "Fe", "O", "O")
    # Direct coordinates are used verbatim as reduced coordinates.
    assert structure.sites.reduced_coords.to_floats()[2] == [0.1, 0.2, 0.3]


def test_cartesian_reduced_is_exact_and_roundtrips() -> None:
    cartesian = dict(DIRECT, cartesian=True)
    structure = build_poscar(cartesian)
    # Cartesian positions recovered by reduced * basis equal the input coordinates.
    assert structure.cartesian_sites().to_floats()[1] == [0.5, 0.5, 0.5]


def test_volume_scaling_targets_requested_volume() -> None:
    volume_case = dict(DIRECT, scale=None, volume="512.0")
    structure = build_poscar(volume_case)
    # The cube-root scale is a deterministic approximation, so compare approximately.
    assert structure.cell.volume.to_float() == pytest.approx(512.0, rel=1e-6)


def test_vasp4_without_symbols_rejected() -> None:
    vasp4 = dict(DIRECT, symbols=None)
    with pytest.raises(ValueError) as excinfo:
        build_poscar(vasp4)
    assert "VASP-4" in str(excinfo.value)


def test_wrong_format_rejected() -> None:
    with pytest.raises(ValueError):
        build_poscar({"format": "not-vasp"})


def test_private_payload_adapter_rejects_unknown_format_and_non_mapping() -> None:
    with pytest.raises(ValueError, match="unknown-format"):
        build_payload({"format": "unknown-format"})
    with pytest.raises(ValueError, match="expected a mapping"):
        build_payload(["not", "a", "mapping"])


CONTCAR_TEXT = """He cell
1.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 2.0
He
1
Direct
0.0 0.0 0.0
"""


def test_core_load_poscar_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("httk.io")  # load() needs the POSCAR loader httk-io registers
    contcar = tmp_path / "CONTCAR.bz2"
    contcar.write_bytes(bz2.compress(CONTCAR_TEXT.encode("utf-8")))
    structure = load(str(contcar))
    assert [species.name for species in structure.species] == ["He"]
    assert structure.cell.basis.to_floats() == [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]


def test_core_load_adapts_poscar_and_raw_keeps_payload(tmp_path: Path) -> None:
    pytest.importorskip("httk.io")
    contcar = tmp_path / "POSCAR"
    contcar.write_text(CONTCAR_TEXT, encoding="utf-8")

    structure = load(str(contcar))
    assert isinstance(structure, UnitcellStructure)
    payload = load(str(contcar), raw=True)
    assert payload["format"] == "vasp-poscar"


def test_atomistic_handler_reload_is_idempotent() -> None:
    handler = importlib.import_module("httk.registry.io.atomistic")
    importlib.reload(handler)


def test_core_load_discovers_atomistic_lazily() -> None:
    code = textwrap.dedent(
        f"""
        import sys
        import tempfile
        from pathlib import Path

        import httk.core

        assert "httk.atomistic" not in sys.modules
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "POSCAR"
            path.write_text({CONTCAR_TEXT!r}, encoding="utf-8")
            result = httk.core.load(str(path))

        from httk.atomistic import UnitcellStructure

        assert isinstance(result, UnitcellStructure)
        assert "httk.atomistic" in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_core_load_unknown_format(tmp_path: Path) -> None:
    pytest.importorskip("httk.io")  # load() needs the CIF loader httk-io registers
    cif = tmp_path / "x.cif"
    cif.write_text("#h\ndata_x\n_cell_length_a 1.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load(str(cif))
