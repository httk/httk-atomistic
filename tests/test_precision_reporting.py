"""What the readers report about how precisely a file wrote its numbers.

This was computed and discarded for as long as the readers have existed — the value went
into a `'resolution'` key that nothing read after its one consumer was deleted. It is now
reported in a form a structure builder can use, so it is worth pinning what each reader
claims and in what units.
"""

import fractions
import io
from pathlib import Path

import pytest
from httk.core import decimal_precision

from httk.atomistic.integrations.vasp.io import read_poscar
from httk.atomistic.io.cif.cif_parser import parse_cif_float, single_asu_from_cif_file
from httk.atomistic.io.cif.cif_reader import read_cif
from httk.atomistic.io.cif.mcif_parser import cifblock_to_mag_asu

F = fractions.Fraction


def _cif(tmp_path: Path, *, cell: str, sites: str) -> Path:
    path = tmp_path / "test.cif"
    path.write_text(
        "data_test\n"
        f"{cell}"
        "loop_\n_space_group_symop_operation_xyz\n'x,y,z'\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        f"{sites}",
        encoding="utf-8",
    )
    return path


CUBIC_CELL = (
    "_cell_length_a 5.6402\n_cell_length_b 5.6402\n_cell_length_c 5.6402\n"
    "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
)


# --- what a single token claims ---


@pytest.mark.parametrize(
    ("token", "precision", "esd"),
    [
        ("0.3333", F(1, 10000), None),
        ("0.3333(7)", F(1, 10000), F(7, 10000)),
        ("5.6402(3)", F(1, 10000), F(3, 10000)),
        ("1.234(5)", F(1, 1000), F(1, 200)),
        ("10", None, None),
        ("1.2e-3", None, None),
        ("3(1)e-1", None, F(1, 10)),
        ("+4.2(3)e-1", None, F(3, 100)),
        ("1/3", None, None),
        ("?", None, None),
    ],
)
def test_a_token_reports_its_own_precision_and_uncertainty(token: str, precision: F | None, esd: F | None) -> None:
    """Both exact rationals. They measure different things and both are reported."""
    _value, meta = parse_cif_float(token, meta=True)
    assert meta["precision"] == precision
    assert meta["esd"] == esd


def test_precision_and_esd_are_exact_not_floats() -> None:
    """So neither picks up binary noise on the way to becoming a tolerance."""
    _value, meta = parse_cif_float("5.6402(3)", meta=True)
    assert isinstance(meta["precision"], F)
    assert isinstance(meta["esd"], F)


def test_signed_exponent_with_esd_has_consistent_value_and_metadata() -> None:
    value, meta = parse_cif_float("+4.2(3)e-1", meta=True)
    assert value == 0.42
    assert meta == {"precision": None, "esd": F(3, 100)}


@pytest.mark.parametrize(
    "literal",
    [
        "42",
        "5",
        "0",
        "1",
        "-3",
        "-42",
        "42.",
        "5.",
        "0.",
        "1.",
        "-3.",
        "-42.",
        "42.2",
        "5.3",
        "0.1",
        "1.3",
        "-3.0",
        "-42.6",
    ],
)
def test_cif_numbers_with_at_most_one_decimal_make_no_precision_claim(literal: str) -> None:
    _value, meta = parse_cif_float(literal, meta=True)
    assert meta == {"precision": None, "esd": None}


def test_two_or_more_cif_decimals_make_a_precision_claim() -> None:
    for literal, precision in (
        ("0.00", F(1, 100)),
        ("0.10", F(1, 100)),
        ("0.25", F(1, 100)),
        ("0.50", F(1, 100)),
        ("0.75", F(1, 100)),
        ("1.00", F(1, 100)),
    ):
        _value, meta = parse_cif_float(literal, meta=True)
        assert meta["precision"] == precision


# --- CIF ---


def test_cif_reports_coordinate_and_basis_precision(tmp_path: Path) -> None:
    data = single_asu_from_cif_file(str(_cif(tmp_path, cell=CUBIC_CELL, sites="Na1 Na 0.1234 0.1234 0.1234 \n")))
    assert data["coordinate_precision"] == F(1, 10000)
    assert data["basis_precision"] == F(1, 10000)


def test_conventional_half_coordinates_do_not_widen_precision(tmp_path: Path) -> None:
    sites = "Na1 Na 0.123456 0.123456 0.123456 \nCl1 Cl 0.5 0.5 0.5 \n"
    data = single_asu_from_cif_file(str(_cif(tmp_path, cell=CUBIC_CELL, sites=sites)))
    assert data["coordinate_precision"] == F(1, 1000000)


@pytest.mark.skipif(
    decimal_precision("0.") is not None, reason="requires unreleased core trailing-point precision semantics"
)
def test_cif_trailing_point_coordinates_do_not_widen_precision(tmp_path: Path) -> None:
    data = single_asu_from_cif_file(str(_cif(tmp_path, cell=CUBIC_CELL, sites="Na1 Na 0. 1. 0.125 \n")))
    assert data["coordinate_precision"] == F(1, 1000)


def test_a_stated_uncertainty_widens_the_precision(tmp_path: Path) -> None:
    """``5.6402(3)`` is good to 3e-4, not to the 1e-4 its four decimals alone suggest."""
    cell = (
        "_cell_length_a 5.6402(3)\n_cell_length_b 5.6402\n_cell_length_c 5.6402\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
    )
    data = single_asu_from_cif_file(str(_cif(tmp_path, cell=cell, sites="Na1 Na 0.1234 0.1234 0.1234 \n")))
    assert data["basis_precision"] == F(3, 10000)


def test_an_exact_fraction_makes_no_claim(tmp_path: Path) -> None:
    """A coordinate of 1/3 states a value; it must not drag the table down to 1e-1."""
    sites = "Na1 Na 1/3 2/3 0.1234 \n"
    data = single_asu_from_cif_file(str(_cif(tmp_path, cell=CUBIC_CELL, sites=sites)))
    assert data["coordinate_precision"] == F(1, 10000)


def test_the_basis_precision_ignores_the_angles(tmp_path: Path) -> None:
    """Angles are in degrees, and a right angle is symmetry rather than a measurement.

    Folding a 1-degree "precision" from a bare ``90`` into a length would be nonsense.
    """
    cell = (
        "_cell_length_a 5.6402\n_cell_length_b 5.6402\n_cell_length_c 5.6402\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
    )
    data = single_asu_from_cif_file(str(_cif(tmp_path, cell=cell, sites="Na1 Na 0.1234 0.1234 0.1234 \n")))
    assert data["basis_precision"] == F(1, 10000)


def test_the_cell_parameters_are_kept_as_written(tmp_path: Path) -> None:
    """So 5.6402 can become 56402/10000 rather than a binary float."""
    data = single_asu_from_cif_file(str(_cif(tmp_path, cell=CUBIC_CELL, sites="Na1 Na 0.1234 0.1234 0.1234 \n")))
    assert data["cell_parameters_exact"] == ("5.6402", "5.6402", "5.6402", "90", "90", "90")


def test_the_dead_resolution_key_is_gone(tmp_path: Path) -> None:
    data = single_asu_from_cif_file(str(_cif(tmp_path, cell=CUBIC_CELL, sites="Na1 Na 0.1234 0.1234 0.1234 \n")))
    assert "resolution" not in data


def test_cif_payload_keeps_only_exact_geometry_channels(tmp_path: Path) -> None:
    data = single_asu_from_cif_file(str(_cif(tmp_path, cell=CUBIC_CELL, sites="Na1 Na 0.1234 0.1234 0.1234 \n")))
    assert {"basis", "positions", "cell_parameters"}.isdisjoint(data)


def test_cif_keeps_occupancy_value_spelling_and_precision(tmp_path: Path) -> None:
    path = tmp_path / "occupancies.cif"
    path.write_text(
        "data_test\n"
        f"{CUBIC_CELL}"
        "loop_\n_space_group_symop_operation_xyz\n'x,y,z'\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n"
        "Na1 Na 0 0 0 1\n"
        "Na2 Na 0 0 0 1/3\n"
        "Na3 Na 0 0 0 0.3333\n"
        "Na4 Na 0 0 0 0.3333(7)\n"
        "Na5 Na 0 0 0 ?\n",
        encoding="utf-8",
    )

    data = single_asu_from_cif_file(str(path))
    assert data["occupancies"] == [1.0, 1.0 / 3.0, 0.3333, 0.3333, None]
    assert data["occupancies_exact"] == ["1", "1/3", "0.3333", "0.3333", None]
    assert data["occupancy_precisions"] == [None, None, F(1, 10000), F(7, 10000), None]

    _name, raw = read_cif(str(path))[0][0]
    assert raw["atom_site_occupancy"] == ["1", "1/3", "0.3333", "0.3333(7)", "?"]


def test_cif_occupancy_with_esd_preserves_exponent_and_leading_plus(tmp_path: Path) -> None:
    path = tmp_path / "exponent-occupancies.cif"
    path.write_text(
        "data_test\n"
        f"{CUBIC_CELL}"
        "loop_\n_space_group_symop_operation_xyz\n'x,y,z'\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n"
        "Na1 Na 0 0 0 3(1)e-1\n"
        "Na2 Na 0 0 0 +4.2(3)e-1\n",
        encoding="utf-8",
    )

    data = single_asu_from_cif_file(str(path))
    assert data["occupancies"] == [0.3, 0.42]
    assert data["occupancies_exact"] == ["3e-1", "+4.2e-1"]
    assert [F(token) for token in data["occupancies_exact"]] == [F(3, 10), F(21, 50)]
    assert data["occupancy_precisions"] == [F(1, 10), F(3, 100)]

    _name, raw = read_cif(str(path))[0][0]
    assert raw["atom_site_occupancy"] == ["3(1)e-1", "+4.2(3)e-1"]


def test_no_occupancy_column_keeps_all_occupancy_payload_keys_none(tmp_path: Path) -> None:
    data = single_asu_from_cif_file(str(_cif(tmp_path, cell=CUBIC_CELL, sites="Na1 Na 0.1234 0.1234 0.1234 \n")))
    assert data["occupancies"] is None
    assert data["occupancies_exact"] is None
    assert data["occupancy_precisions"] is None


def test_mcif_payload_threads_occupancy_fidelity_from_shared_atom_parser() -> None:
    data = cifblock_to_mag_asu(
        {
            "cell_length_a": "5.6402",
            "cell_length_b": "5.6402",
            "cell_length_c": "5.6402",
            "cell_angle_alpha": "90",
            "cell_angle_beta": "90",
            "cell_angle_gamma": "90",
            "atom_site_label": ["Na1"],
            "atom_site_type_symbol": ["Na"],
            "atom_site_fract_x": ["0"],
            "atom_site_fract_y": ["0"],
            "atom_site_fract_z": ["0"],
            "atom_site_occupancy": ["0.3333(7)"],
            "space_group_symop_magn_operation.xyz": ["x,y,z,+1"],
            "parent_space_group.name_h-m_alt": "P 1",
            "parent_space_group.it_number": "1",
        }
    )

    assert data["occupancies"] == [0.3333]
    assert data["occupancies_exact"] == ["0.3333"]
    assert data["occupancy_precisions"] == [F(7, 10000)]


# --- POSCAR ---


POSCAR = """NaCl
1.0
5.6400 0.0000 0.0000
0.0000 5.6400 0.0000
0.0000 0.0000 5.6400
Na Cl
1 1
Direct
0.00000 0.00000 0.00000
0.500 0.500 0.500
"""


def test_poscar_reports_its_token_precisions() -> None:
    """Raw, unconverted: the cell is still to be scaled and the coordinates may be Cartesian."""
    data = read_poscar(io.StringIO(POSCAR))
    assert data["cell_precision"] == F(1, 10000)
    assert data["scale_precision"] == F(1, 10)
    assert data["coordinate_precision"] == F(1, 1000)


def test_poscar_direct_special_fractions_do_not_widen_precision() -> None:
    special = POSCAR.replace("0.500 0.500 0.500", "0.5 0.5 1.0")
    assert read_poscar(io.StringIO(special))["coordinate_precision"] == F(1, 100000)


@pytest.mark.parametrize("literal", ["0.0", "+0.5", "-0.5", "1.0"])
def test_poscar_direct_special_fractions_alone_make_no_precision_claim(literal: str) -> None:
    special = POSCAR.replace("0.00000 0.00000 0.00000", f"{literal} {literal} {literal}").replace(
        "0.500 0.500 0.500", f"{literal} {literal} {literal}"
    )
    assert read_poscar(io.StringIO(special))["coordinate_precision"] is None


def test_poscar_cartesian_values_are_not_treated_as_special_fractions() -> None:
    cartesian = POSCAR.replace("Direct", "Cartesian").replace("0.500 0.500 0.500", "0.5 0.5 1.0")
    assert read_poscar(io.StringIO(cartesian))["coordinate_precision"] == F(1, 10)


def test_a_volume_scaled_poscar_reports_no_scale_precision() -> None:
    """The negative form states a target volume, so there is no scale token to read."""
    data = read_poscar(io.StringIO(POSCAR.replace("1.0\n", "-179.4\n", 1)))
    assert data["volume"] is not None
    assert data["scale_precision"] is None
