"""Tests for building asymmetric-unit structures from CIF files.

A CIF is the natural source for an ASU: it lists one site per orbit and states the
operations that generate the rest, so no symmetry search is needed and spglib is not
involved. What has to be right is the *setting* — a file written in a non-standard setting
must be recognized as such rather than reinterpreted — and the fidelity of the numbers.

CIF text is generated from the vendored tables so that a fixture can be produced for any
setting; the coordinates and occupancies in it are written by hand.
"""

import fractions
from pathlib import Path

import pytest
from httk.core import load

from httk.atomistic import (
    Spacegroup,
    UnitcellStructureView,
    asu_structure_from_cif,
    asu_structures_from_cif,
    cif_setting,
    load_asu_structure,
    load_structure,
    same_crystal,
)

F = fractions.Fraction

pytest.importorskip("httk.io", reason="the CIF reader lives in httk-io")


def _write_cif(
    path: Path,
    setting: str,
    parameters: tuple[float, ...],
    sites: list[tuple[str, str, tuple[str, str, str], str]],
    *,
    name: str = "test",
    declare_number: bool = True,
) -> Path:
    """A CIF for one setting, with its complete symmetry-operation list."""
    spacegroup = Spacegroup.for_setting(setting)
    a, b, c, alpha, beta, gamma = parameters
    lines = [
        f"data_{name}",
        f"_cell_length_a {a}",
        f"_cell_length_b {b}",
        f"_cell_length_c {c}",
        f"_cell_angle_alpha {alpha}",
        f"_cell_angle_beta {beta}",
        f"_cell_angle_gamma {gamma}",
    ]
    if declare_number:
        lines.append(f"_space_group_IT_number {spacegroup.it_number}")
        lines.append(f"_space_group_name_H-M_alt '{spacegroup.hermann_mauguin}'")
    lines += ["loop_", "_space_group_symop_operation_xyz"]
    lines += [f"'{operation.wrapped().to_xyz()}'" for operation in spacegroup.symmetry_operations]
    lines += [
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    for label, symbol, (x, y, z), occupancy in sites:
        lines.append(f"{label} {symbol} {x} {y} {z} {occupancy}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _rocksalt_cif(tmp_path: Path) -> Path:
    return _write_cif(
        tmp_path / "nacl.cif",
        Spacegroup.standard(225).setting,
        (5.64, 5.64, 5.64, 90, 90, 90),
        [("Na1", "Na", ("0.0", "0.0", "0.0"), "1.0"), ("Cl1", "Cl", ("0.5", "0.5", "0.5"), "1.0")],
        name="NaCl",
    )


# --- reading ---


def test_cif_expands_to_the_full_cell(tmp_path: Path) -> None:
    asu = load_asu_structure(str(_rocksalt_cif(tmp_path)))
    assert asu.spacegroup.it_number == 225
    assert [(site.wyckoff, site.species) for site in asu.asu_sites] == [("a", "Na"), ("b", "Cl")]

    structure = UnitcellStructureView(asu)
    assert len(structure.sites) == 8
    assert sorted(structure.species_at_sites) == ["Cl"] * 4 + ["Na"] * 4
    # Exact, not approximate: expansion never leaves the rationals.
    assert set(tuple(row) for row in structure.sites.reduced_coords.to_fractions()) == {
        (F(0), F(0), F(0)),
        (F(0), F(1, 2), F(1, 2)),
        (F(1, 2), F(0), F(1, 2)),
        (F(1, 2), F(1, 2), F(0)),
        (F(0), F(0), F(1, 2)),
        (F(0), F(1, 2), F(0)),
        (F(1, 2), F(0), F(0)),
        (F(1, 2), F(1, 2), F(1, 2)),
    }


def test_load_structure_and_load_asu_structure_agree(tmp_path: Path) -> None:
    path = str(_rocksalt_cif(tmp_path))
    assert same_crystal(load_structure(path), UnitcellStructureView(load_asu_structure(path)))


def test_the_cell_is_exact_not_the_files_rounded_basis(tmp_path: Path) -> None:
    """Built from a, b, c and the angles, so a cubic cell keeps exact right angles.

    The mapping also carries a pre-multiplied floating-point basis whose off-diagonal
    entries are ~3e-16 rather than zero; using it would put that noise into every
    structure.
    """
    asu = load_asu_structure(str(_rocksalt_cif(tmp_path)))
    assert asu.cell.angles == (F(90), F(90), F(90))
    assert asu.cell.lengths[0] == asu.cell.lengths[1] == asu.cell.lengths[2]


# --- settings ---


def test_a_non_standard_setting_is_recognized_as_itself(tmp_path: Path) -> None:
    """Identified from the symmetry operations, so the file is not silently reinterpreted."""
    path = _write_cif(
        tmp_path / "sg15.cif",
        "15:c1",
        (5, 6, 7, 90, 90, 90),
        [("Si1", "Si", ("0.25", "0.0", "0.3333"), "1.0")],
        name="SG15c1",
    )
    asu = load_asu_structure(str(path))
    setting = asu.setting()
    assert setting is not None
    assert setting.setting == "15:c1"
    assert not asu.is_standard_setting
    assert asu.spacegroup.setting == "15:b1"


def test_the_setting_is_found_even_when_the_file_declares_nothing(tmp_path: Path) -> None:
    """With no symbol to narrow the search, the operations alone still identify it."""
    path = _write_cif(
        tmp_path / "bare.cif",
        "15:c1",
        (5, 6, 7, 90, 90, 90),
        [("Si1", "Si", ("0.25", "0.0", "0.3333"), "1.0")],
        declare_number=False,
    )
    assert cif_setting(load(str(path))["blocks"][0]).setting == "15:c1"


def test_an_unidentifiable_setting_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    """A transform cannot be derived; infinitely many are valid and they differ."""
    path = _rocksalt_cif(tmp_path)
    block = dict(load(str(path))["blocks"][0])
    # Drop most of the operations, so the set matches no tabulated group, and drop the
    # declaration too so the failure is about the operations rather than a contradiction.
    block["symops"] = block["symops"][:3]
    block["space_group_nbr"] = None
    block["space_group_name_hall"] = None
    with pytest.raises(ValueError, match="no tabulated space-group setting"):
        asu_structure_from_cif(block)

    # Ignoring the declaration does not rescue it: there is genuinely no such setting.
    with pytest.raises(ValueError, match="no tabulated space-group setting"):
        asu_structure_from_cif(block, trust_declared_symmetry=False)


def test_a_block_with_no_symmetry_operations_is_refused(tmp_path: Path) -> None:
    path = _rocksalt_cif(tmp_path)
    block = dict(load(str(path))["blocks"][0])
    block["symops"] = []
    with pytest.raises(ValueError, match="no symmetry operations"):
        asu_structure_from_cif(block)


# --- fidelity ---


def test_occupancies_survive_into_the_structure(tmp_path: Path) -> None:
    """They were parsed and then dropped before; a half-occupied site is not a full one."""
    path = _write_cif(
        tmp_path / "partial.cif",
        Spacegroup.standard(225).setting,
        (5.64, 5.64, 5.64, 90, 90, 90),
        [("Na1", "Na", ("0.0", "0.0", "0.0"), "0.5"), ("Cl1", "Cl", ("0.5", "0.5", "0.5"), "1.0")],
        name="Partial",
    )
    asu = load_asu_structure(str(path))
    concentrations = {species.name: species.concentration for species in asu.species}
    assert concentrations["Na1"] == (0.5,)
    assert concentrations["Cl"] == (1.0,)
    # A partially occupied site is named for its CIF label, since two sites of one element
    # can carry different occupancies.
    assert [site.species for site in asu.asu_sites] == ["Na1", "Cl"]


def test_coordinates_embed_as_the_decimal_the_file_wrote(tmp_path: Path) -> None:
    """``0.3333`` is 3333/10000, not the binary value of ``float("0.3333")``.

    The free parameter keeps what the file said; only the position's fixed components are
    replaced by their exact values.
    """
    path = _write_cif(
        tmp_path / "sg15.cif",
        "15:b1",
        (5, 6, 7, 90, 90, 90),
        [("Si1", "Si", ("0.0", "0.3333", "0.25"), "1.0")],
    )
    asu = load_asu_structure(str(path))
    assert asu.asu_sites[0].wyckoff == "e"
    assert asu.asu_sites[0].free_params.to_fractions() == [F(3333, 10000)]


def test_uncertainties_are_stripped_from_coordinates(tmp_path: Path) -> None:
    path = _write_cif(
        tmp_path / "esd.cif",
        "15:b1",
        (5, 6, 7, 90, 90, 90),
        [("Si1", "Si", ("0.0", "0.3333(7)", "0.25"), "1.0")],
    )
    assert load_asu_structure(str(path)).asu_sites[0].free_params.to_fractions() == [F(3333, 10000)]


def test_a_site_on_no_special_position_falls_back_to_the_general_one(tmp_path: Path) -> None:
    """A CIF site can always be placed, because the general position accepts any point.

    That is worth stating rather than assuming: it means reading a CIF never fails for want
    of a matching Wyckoff position, and a site that is not on any special position simply
    generates the full orbit of the general position — which is exactly what the file's own
    symmetry operations would generate from it.
    """
    path = _write_cif(
        tmp_path / "general.cif",
        Spacegroup.standard(225).setting,
        (5.64, 5.64, 5.64, 90, 90, 90),
        [("Na1", "Na", ("0.3", "0.11", "0.07"), "1.0")],
    )
    asu = asu_structure_from_cif(load(str(path))["blocks"][0], tolerance=0.0)
    general = Spacegroup.standard(225).wyckoff[-1]
    assert general.free_count == 3
    assert asu.asu_sites[0].wyckoff == general.letter
    assert len(UnitcellStructureView(asu).sites) == general.multiplicity


# --- payload handling ---


def test_asu_structures_from_cif_reports_why_a_file_yielded_nothing(tmp_path: Path) -> None:
    """An empty result must not read as "this file contained no structures"."""
    path = tmp_path / "incomplete.cif"
    path.write_text(
        "data_x\n_cell_length_a 5.0\nloop_\n_atom_site_label\n_atom_site_fract_x\n"
        "_atom_site_fract_y\n_atom_site_fract_z\nNa 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    payload = load(str(path))
    assert payload["blocks"] == []
    with pytest.raises(ValueError, match="no structure that could be interpreted"):
        asu_structures_from_cif(payload)
    with pytest.raises(ValueError, match="no unit cell"):
        asu_structures_from_cif(payload)


def test_a_multi_block_cif_yields_one_structure_per_block(tmp_path: Path) -> None:
    first = _rocksalt_cif(tmp_path).read_text(encoding="utf-8")
    second = _write_cif(
        tmp_path / "second.cif",
        "15:b1",
        (5, 6, 7, 90, 90, 90),
        [("Si1", "Si", ("0.0", "0.3333", "0.25"), "1.0")],
        name="second",
    ).read_text(encoding="utf-8")
    combined = tmp_path / "both.cif"
    combined.write_text(first + second, encoding="utf-8")

    structures = asu_structures_from_cif(load(str(combined)))
    assert [structure.spacegroup.it_number for structure in structures] == [225, 15]

    with pytest.raises(ValueError, match="holds 2 structures"):
        load_asu_structure(str(combined))


def test_a_non_cif_payload_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected a 'cif' mapping"):
        asu_structure_from_cif({"format": "vasp-poscar"})


# --- what the file declares is checked, not merely used as a hint ---


def _sg15_cif(tmp_path: Path, *, declaration: str) -> Path:
    """SG 15 operations, with whatever space-group declaration is passed."""
    spacegroup = Spacegroup.for_setting("15:b1")
    operations = "\n".join(f"'{op.wrapped().to_xyz()}'" for op in spacegroup.symmetry_operations)
    path = tmp_path / "declared.cif"
    path.write_text(
        "data_x\n_cell_length_a 5.0\n_cell_length_b 6.0\n_cell_length_c 7.0\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        + declaration
        + f"loop_\n_space_group_symop_operation_xyz\n{operations}\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Si1 Si 0.0 0.3333 0.25\n",
        encoding="utf-8",
    )
    return path


def test_a_conventionally_spelled_hall_symbol_is_recognized(tmp_path: Path) -> None:
    """CIFs write ``-C 2yc``; the tables key it ``-c_2yc``.

    Without normalizing, every correctly declared Hall symbol looks unknown — which used to
    be survivable only because the miss was silent, and would now be an error.
    """
    block = load(str(_sg15_cif(tmp_path, declaration="_space_group_name_Hall '-C 2yc'\n")))["blocks"][0]
    assert cif_setting(block).setting == "15:b1"


def test_a_hall_symbol_naming_no_setting_is_an_error(tmp_path: Path) -> None:
    block = load(str(_sg15_cif(tmp_path, declaration="_space_group_name_Hall 'Not A Symbol'\n")))["blocks"][0]
    with pytest.raises(ValueError, match="names no known space-group setting"):
        cif_setting(block)


def test_a_hall_symbol_naming_the_wrong_group_is_an_error(tmp_path: Path) -> None:
    """SG 14's Hall symbol on a file whose operations are SG 15's: the file contradicts itself."""
    block = load(str(_sg15_cif(tmp_path, declaration="_space_group_name_Hall '-P 2ybc'\n")))["blocks"][0]
    with pytest.raises(ValueError, match="contradicts itself"):
        cif_setting(block)


def test_a_wrong_it_number_is_an_error(tmp_path: Path) -> None:
    block = load(str(_sg15_cif(tmp_path, declaration="_space_group_IT_number 14\n")))["blocks"][0]
    with pytest.raises(ValueError, match="contradicts itself"):
        cif_setting(block)


@pytest.mark.parametrize(
    ("declaration", "message"),
    [("_space_group_IT_number 999\n", "outside the range"), ("_space_group_IT_number banana\n", "not a\nnumber")],
    ids=["out-of-range", "not-a-number"],
)
def test_an_unusable_it_number_is_an_error(tmp_path: Path, declaration: str, message: str) -> None:
    block = load(str(_sg15_cif(tmp_path, declaration=declaration)))["blocks"][0]
    with pytest.raises(ValueError, match=message.replace("\n", " ")):
        cif_setting(block)


@pytest.mark.parametrize(
    "declaration",
    [
        "_space_group_name_Hall 'Not A Symbol'\n",
        "_space_group_name_Hall '-P 2ybc'\n",
        "_space_group_IT_number 14\n",
        "_space_group_IT_number 999\n",
    ],
    ids=["unknown-hall", "wrong-hall", "wrong-number", "out-of-range-number"],
)
def test_the_declaration_can_be_ignored_on_request(tmp_path: Path, declaration: str) -> None:
    """The escape hatch: when the operations are the trustworthy half of the file."""
    block = load(str(_sg15_cif(tmp_path, declaration=declaration)))["blocks"][0]
    assert cif_setting(block, trust_declared_symmetry=False).setting == "15:b1"

    asu = asu_structure_from_cif(block, trust_declared_symmetry=False)
    assert asu.spacegroup.it_number == 15
    assert asu.asu_sites[0].wyckoff == "e"


def test_the_escape_hatch_reaches_through_load_asu_structure(tmp_path: Path) -> None:
    path = _sg15_cif(tmp_path, declaration="_space_group_IT_number 14\n")
    with pytest.raises(ValueError, match="contradicts itself"):
        load_asu_structure(str(path))
    assert load_asu_structure(str(path), trust_declared_symmetry=False).spacegroup.it_number == 15


def test_a_file_with_no_declaration_searches_every_setting(tmp_path: Path) -> None:
    block = load(str(_sg15_cif(tmp_path, declaration="")))["blocks"][0]
    assert cif_setting(block).setting == "15:b1"
