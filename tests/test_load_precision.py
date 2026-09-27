"""``httk.core.load(path, precision=...)`` is accepted uniformly by the structure readers."""

from pathlib import Path

import pytest
from httk.core import load

from httk.atomistic import UnitcellStructureView

FIXTURES = Path(__file__).parent / "fixtures"

_CIF = """\
data_nacl
_cell_length_a 5.6402(3)
_cell_length_b 5.6402(3)
_cell_length_c 5.6402(3)
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0000 0.0000 0.0000
Cl1 Cl 0.5000 0.5000 0.5000
"""

_POSCAR = "NaCl\n1.0\n5.6402 0 0\n0 5.6402 0\n0 0 5.6402\nNa Cl\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n"


def _files(tmp_path: Path) -> list[Path]:
    (tmp_path / "nacl.cif").write_text(_CIF, encoding="utf-8")
    (tmp_path / "POSCAR").write_text(_POSCAR, encoding="utf-8")
    return [tmp_path / "nacl.cif", tmp_path / "POSCAR", FIXTURES / "magnetic_centered.mcif"]


def test_one_call_site_loads_every_structure_format_with_precision(tmp_path: Path) -> None:
    for path in _files(tmp_path):
        structure = UnitcellStructureView(load(str(path), precision=5e-4))
        assert len(structure.sites) == 2


@pytest.mark.parametrize("name", ["nacl.cif", "magnetic_centered.mcif"])
def test_precision_has_no_effect_on_exact_cif_formats(tmp_path: Path, name: str) -> None:
    path = next(path for path in _files(tmp_path) if path.name == name)

    with_precision = load(str(path), precision=5e-4)
    without = load(str(path))

    assert repr(UnitcellStructureView(with_precision)) == repr(UnitcellStructureView(without))
    assert load(str(path), raw=True, precision=5e-4) == load(str(path), raw=True)


@pytest.mark.parametrize("name", ["nacl.cif", "POSCAR", "magnetic_centered.mcif"])
@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), True, "5e-4"])
def test_invalid_precision_is_rejected_uniformly(tmp_path: Path, name: str, bad: object) -> None:
    path = next(path for path in _files(tmp_path) if path.name == name)

    with pytest.raises(ValueError, match="precision must be a finite number greater than zero"):
        load(str(path), precision=bad)
