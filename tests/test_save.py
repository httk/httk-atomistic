from fractions import Fraction

from httk.core import FracVector, load, save
from httk.io.cif.cif_reader import read_cif

from httk.atomistic import ASUStructure, Cell, Spacegroup, Species, UnitcellStructureView, WyckoffSite, same_crystal

P1 = """\
data_p1
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
_space_group_name_H-M_alt 'P 1'
loop_
_space_group_symop_operation_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si Si 1/3 0 0 1
"""


def test_cif_save_load_preserves_exact_p1_structure(tmp_path):
    source = tmp_path / "x.cif"
    source.write_text(P1, encoding="utf-8")
    original = load(source)
    destination = tmp_path / "y.cif"
    save(original, destination)
    recovered = load(destination)
    assert same_crystal(UnitcellStructureView(original), UnitcellStructureView(recovered))
    assert recovered.spacegroup.it_number == original.spacegroup.it_number
    assert recovered.wyckoff_sites[0].representative.to_fractions()[0] == Fraction(1, 3)


def test_cif_save_load_preserves_compression_suffixes(tmp_path):
    source = tmp_path / "x.cif"
    source.write_text(P1, encoding="utf-8")
    original = load(source)
    for suffix in (".gz", ".bz2"):
        destination = tmp_path / f"compressed.cif{suffix}"
        save(original, destination)
        recovered = load(destination)
        assert same_crystal(UnitcellStructureView(original), UnitcellStructureView(recovered))


def test_cif_save_load_preserves_a_declared_setting(tmp_path):
    setting = Spacegroup.for_setting("15:c1")
    lines = [
        "data_setting",
        "_cell_length_a 5",
        "_cell_length_b 6",
        "_cell_length_c 7",
        "_cell_angle_alpha 90",
        "_cell_angle_beta 90",
        "_cell_angle_gamma 90",
        f"_space_group_IT_number {setting.it_number}",
        "loop_",
        "_space_group_symop_operation_xyz",
        *(f"'{operation.wrapped().to_xyz()}'" for operation in setting.symmetry_operations),
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
        "Si1 Si 0.25 0.0 0.3333 1.0",
    ]
    source = tmp_path / "setting.cif"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    original = load(source)
    destination = tmp_path / "setting-out.cif"
    save(original, destination)
    recovered = load(destination)
    assert recovered.setting().setting == original.setting().setting == "15:c1"
    assert [site.wyckoff for site in recovered.wyckoff_sites] == [site.wyckoff for site in original.wyckoff_sites]
    assert same_crystal(UnitcellStructureView(original), UnitcellStructureView(recovered))


def test_cif_save_uses_standard_decimals_and_exact_companions(tmp_path):
    cell = Cell([[5, 0, 0], [0, 5, 0], [0, 0, 5]])
    original = ASUStructure(
        cell,
        221,
        [WyckoffSite("e", FracVector.create(["1/3"]), "Si")],
        [Species("Si", ("Si",), (1,))],
    )
    destination = tmp_path / "sg221.cif"
    save(original, destination)
    text = destination.read_text(encoding="utf-8")
    assert "_httk_atom_site_fract_x_exact" in text
    assert "'1/3'" in text
    _name, raw = read_cif(destination)[0][0]
    assert raw["atom_site_fract_x"] == ["0.3333333333333333"]

    recovered = load(destination)
    assert recovered.spacegroup.it_number == 221
    assert [site.wyckoff for site in recovered.wyckoff_sites] == ["e"]
    assert recovered.wyckoff_sites[0].free_params.to_fractions() == [Fraction(1, 3)]
    assert same_crystal(original, recovered)

    fixed = ASUStructure(cell, 221, [WyckoffSite("a", (), "Si")], [Species("Si", ("Si",), (1,))])
    decimal_destination = tmp_path / "decimal.cif"
    save(fixed, decimal_destination)
    assert "_httk_atom_site_fract_" not in decimal_destination.read_text(encoding="utf-8")
