import logging
from pathlib import Path

import pytest
from httk.core import load
from httk.core._plugins import resolve_callable

from httk.atomistic.io.cif.cif_parser import read_cif_asus
from httk.atomistic.io.cif.cif_reader import _PROTECTED_LOOP_TAGS, read_cif

FIXTURE = Path(__file__).parent / "fixtures" / "malformed_auxiliary_loop.cif"
HINT = " (an auxiliary loop like this can be dropped by loading with repair=True, which applies documented repairs with warnings)"


def test_protected_set_includes_atom_declaration_tags():
    assert {
        "atom_site_wyckoff_label",
        "atom_site_symmetry_multiplicity",
        "atom_site_site_symmetry_multiplicity",
        "atom_site_site_symmetry_order",
    } <= _PROTECTED_LOOP_TAGS


def test_repair_drops_malformed_auxiliary_loop_and_stamps_payload(caplog):
    with pytest.raises(ValueError) as error:
        read_cif(FIXTURE)
    assert HINT in str(error.value)

    with caplog.at_level(logging.WARNING):
        reader = resolve_callable("httk.atomistic.io.cif:read_cif_asus")
        assert reader is read_cif_asus
        payload = reader(FIXTURE, repair=True)

    assert payload["repair"] is True
    assert payload["blocks"][0]["labels"] == ["Na1"]
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.name == "httk.atomistic.io.cif.cif_reader"
    assert record.context == "cif"
    assert "_audit_tag" in record.message
    assert "dropped" in record.message


def test_repair_reconstructs_missing_symmetry_from_hall(tmp_path, caplog):
    source = tmp_path / "missing-symmetry.cif"
    source.write_text(
        """data_test
_symmetry_space_group_name_Hall '-P 2yn'
_symmetry_Int_Tables_number 14
_cell_length_a 6
_cell_length_b 7
_cell_length_c 8
_cell_angle_alpha 90
_cell_angle_beta 100
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.12345 0.23456 0.34567
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CIF block has no symmetry operations"):
        load(str(source))

    with caplog.at_level(logging.WARNING, logger="httk.atomistic.io.cif.cif_parser"):
        structure = load(str(source), repair=True)

    assert structure.spacegroup.setting == "14:b2"
    assert len(caplog.records) == 1
    assert caplog.records[0].context == "cif"
    assert "generated 4" in caplog.records[0].getMessage()
    assert "Hall symbol '-P 2yn'" in caplog.records[0].getMessage()

    source.write_text(source.read_text(encoding="utf-8").replace("-P 2yn", "not-a-Hall-symbol"), encoding="utf-8")
    with pytest.raises(ValueError, match="CIF block has no symmetry operations"):
        load(str(source), repair=True)


def test_repair_does_not_drop_malformed_atom_site_loop(tmp_path):
    atom_site_fixture = tmp_path / "malformed_atom_site.cif"
    atom_site_fixture.write_text(FIXTURE.read_text(encoding="utf-8").replace("_audit_tag", "_atom_site_occupancy"))

    for repair in (False, True):
        with pytest.raises(ValueError) as error:
            read_cif(atom_site_fixture, repair=repair)
        assert HINT not in str(error.value)


def test_repair_does_not_drop_malformed_modulation_loop(tmp_path):
    modulation_fixture = tmp_path / "malformed_modulation.cif"
    modulation_fixture.write_text(FIXTURE.read_text(encoding="utf-8").replace("_audit_tag", "_cell_wave_vector_x"))

    for repair in (False, True):
        with pytest.raises(ValueError) as error:
            read_cif(modulation_fixture, repair=repair)
        assert HINT not in str(error.value)


def test_repair_does_not_drop_malformed_wyckoff_declaration_loop(tmp_path):
    wyckoff_fixture = tmp_path / "malformed_wyckoff.cif"
    wyckoff_fixture.write_text(FIXTURE.read_text(encoding="utf-8").replace("_audit_tag", "_atom_site_Wyckoff_label"))

    for repair in (False, True):
        with pytest.raises(ValueError) as error:
            read_cif(wyckoff_fixture, repair=repair)
        assert HINT not in str(error.value)


def test_structural_only_skips_auxiliary_loops_and_fields(tmp_path):
    source = tmp_path / "structural-only.cif"
    source.write_text(
        """data_test
_audit_scalar ignored
loop_
_audit_first
_audit_second
1 2
3 4 _cell_length_a 5
loop_
_atom_site_label
_atom_site_fract_x
C1 0
""",
        encoding="utf-8",
    )

    _, complete = read_cif(source)[0][0]
    _, structural = read_cif(source, structural_only=True)[0][0]

    assert complete["audit_scalar"] == "ignored"
    assert complete["audit_first"] == ["1", "3"]
    assert structural == {
        "cell_length_a": "5",
        "loop_0": ["atom_site_label", "atom_site_fract_x"],
        "atom_site_label": ["C1"],
        "atom_site_fract_x": ["0"],
    }


def test_structural_only_preserves_malformed_auxiliary_loop_policy():
    with pytest.raises(ValueError) as error:
        read_cif(FIXTURE, structural_only=True)
    assert HINT in str(error.value)

    _, block = read_cif(FIXTURE, structural_only=True, repair=True)[0][0]
    assert "audit_tag" not in block
    assert block["atom_site_label"] == ["Na1"]


def test_comment_before_new_loop_preserves_the_control_token():
    source = [
        "data_test\n",
        "loop_\n",
        "_audit_first\n",
        "_audit_second\n",
        "1 2\n",
        "# next loop\n",
        "loop_\n",
        "_atom_site_label\n",
        "C1\n",
        "# quoted reserved words remain values\n",
        "'loop_'\n",
    ]

    for structural_only in (False, True):
        _, block = read_cif(source, pragmatic=False, structural_only=structural_only)[0][0]
        assert block["atom_site_label"] == ["C1", "loop_"]
