from fractions import Fraction
from pathlib import Path

import pytest

from httk.core import FracVector, load, save

from httk.atomistic import ASUStructure, Cell, Spacegroup, Species, UnitcellStructureView, WyckoffSite, same_crystal
from httk.atomistic.io.cif.cif_reader import read_cif

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


def test_cif_save_load_preserves_disorder_without_writing_it_lossily(tmp_path):
    original = load(Path(__file__).with_name("fixtures") / "disorder" / "217.cif")
    destination = tmp_path / "disorder.cif"

    save(original, destination)
    recovered = load(destination)

    assert recovered.species == original.species
    assert [(site.wyckoff, site.species, site.free_params) for site in recovered.wyckoff_sites] == [
        (site.wyckoff, site.species, site.free_params) for site in original.wyckoff_sites
    ]


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
    setting = Spacegroup.from_setting("15:c1")
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
        [WyckoffSite("e", FracVector(["1/3"]), "Si")],
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


# A relaxed CONTCAR carries full-precision float noise (0.355 arrives as 0.3549999999999969);
# with a stated precision the CIF must render clean decimals and NOT fabricate _httk_*_exact
# rational companions from that noise (e.g. 563492063538/1587301587431 for what is just 0.355).
RELAXED_POSCAR = """CrVO4-like
1.0
5.5679999999999996 0.0000000000000000 0.0000000000000003
0.0000000000000013 8.2080000000000002 0.0000000000000005
0.0000000000000000 0.0000000000000000 5.9770000000000003
V O
1 1
Direct
0.0000000000000000 0.3549999999999969 0.2500000000000000
0.0000000000000000 0.7590000000000003 0.5260000000000034
"""


def test_relaxed_cif_snaps_float_noise_to_clean_decimals(tmp_path):
    source = tmp_path / "CONTCAR"
    source.write_text(RELAXED_POSCAR, encoding="utf-8")
    structure = load(str(source), precision=5e-4)
    destination = tmp_path / "relaxed.cif"
    save(structure, destination)
    text = destination.read_text(encoding="utf-8")

    # No exact companions and no huge-denominator rationals leaked from the float noise.
    assert "_exact" not in text
    assert "/" not in "".join(text.splitlines()[12:])  # no fraction tokens in the atom/loop body
    # Clean, snapped values rather than the raw float noise.
    assert "_cell_length_a 5.568" in text
    assert "0.3549999999999969" not in text
    # Coordinates are written only to the precision they resolve (~1e-4 -> 4 decimals), not padded
    # to 16 places, so the digit count does not claim machine precision the data lacks.
    assert "V V 0.0000 0.3550 0.2500" in text
    assert "0.3550000000000000" not in text
    # A CIF round-trip therefore recovers a realistic precision, not 1e-16 -- and stays there on a
    # second round-trip (the width follows the structure's precision, not just the first-write path).
    reloaded = load(destination)
    assert reloaded.coordinate_precision == Fraction(1, 10000)
    resaved = tmp_path / "relaxed2.cif"
    save(reloaded, resaved)
    assert load(resaved).coordinate_precision == Fraction(1, 10000)


def test_relaxed_cif_strict_mode_refuses_snapped_cell(tmp_path):
    # A relaxed structure has no exact CIF form: snapping to precision makes the cell
    # honestly approximate, so the exact-or-nothing opt-out must refuse rather than
    # writing float noise as if it were exact.
    source = tmp_path / "CONTCAR"
    source.write_text(RELAXED_POSCAR, encoding="utf-8")
    structure = load(str(source), precision=5e-4)
    with pytest.raises(ValueError, match="approximate=False"):
        save(structure, tmp_path / "strict.cif", format="cif", approximate=False)
