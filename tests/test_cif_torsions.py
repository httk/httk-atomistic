"""Torsion statements written to CIF as a ``_geom_torsion_`` loop by ``save`` and read back by ``load``.

Every read statement is checked against the geometry of the loaded structure, so a round trip that
keeps the statements also proves their signs; a left-handed cell is written as its mirror image.
"""

import hashlib
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from httk.core import load, save
from test_bond_network_torsions import FIXTURE_DOMAINS, ONE, T, _cell, _rows, _staggered
from test_cif_bond_network import (
    LOGGER,
    SOURCE,
    _bonded_rutile_cell,
    _dictionary_definitions,
    _quietly_loaded,
    _rutile,
    _saved,
)

from httk.atomistic import BondNetwork, BondSource, Cell, SiteTorsion, UnitcellStructureView
from httk.atomistic._writing import _cif_angle
from httk.atomistic.cif_structures import _CIF_TORSION_SOURCE
from httk.atomistic.io.cif.cif_parser import parse_cif_float
from httk.atomistic.models.structure.bonds import _signed_torsions
from httk.atomistic.models.structure.network import _TORSION_SYMBOLS, _mirrored_torsion

FIXTURES = Path(__file__).parent / "fixtures"
HAND = BondSource("Torsions chosen by hand", "manual")


def _human(structure: Any) -> Any:
    """The structure with hand-made statements: uncertainties, signed and unsigned symbols, ``none``."""
    rows = list(_rows(structure))
    rows[0] = replace(rows[0], uncertainty=0.15)
    angle = rows[1].angle or 0.0
    rows[1] = replace(rows[1], symbol="+sc" if angle > 0 else "-sc", uncertainty=0.4)
    if len(rows) > 2:
        rows[2] = replace(rows[2], symbol="none", angle=None)
    if len(rows) > 3:
        rows[3] = replace(rows[3], symbol="t")
    return structure._with_bonds(replace(structure.bonds, torsions=rows, torsions_source=HAND))


def _loaded(path: Path, caplog: pytest.LogCaptureFixture, cell: bool) -> Any:
    caplog.clear()  # only what reading logs
    loaded = _quietly_loaded(path, caplog)
    return UnitcellStructureView(loaded).unview() if cell else loaded


def _assert_geometric(structure: Any) -> None:
    rows = [row for row in _rows(structure) if row.angle is not None]
    for row, angle in zip(rows, _signed_torsions(structure, rows), strict=True):
        assert row.angle is not None and angle is not None
        assert abs((row.angle - angle + 180) % 360 - 180) < 1e-6, (row, angle)


def _sourced(structure: Any) -> Any:
    """The structure with a bond source, which is also the source of its torsions."""
    return structure._with_bonds(replace(structure.bonds, source=SOURCE))


ROUND_TRIPS = {
    "NaCl (centrosymmetric)": lambda: FIXTURE_DOMAINS["NaCl (centrosymmetric, special positions)"](),
    "diopside (centrosymmetric, C2/c)": lambda: FIXTURE_DOMAINS["diopside (C2/c)"](),
    "sarcosine (Sohncke)": lambda: FIXTURE_DOMAINS["sarcosine (Sohncke)"](),
    "diamond (non-standard origin choice)": lambda: FIXTURE_DOMAINS["diamond (non-standard origin choice)"](),
    "sarcosine unit cell": lambda: _cell(FIXTURE_DOMAINS["sarcosine (Sohncke)"]()),
    "unit cell without bonds": lambda: _sourced(_staggered(Cell([5, 5, 10, 90, 90, 120]))),
}


@pytest.mark.parametrize("name", ROUND_TRIPS)
def test_torsions_round_trip_through_cif(name: str, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    original = _human(ROUND_TRIPS[name]()) if name != "unit cell without bonds" else ROUND_TRIPS[name]()
    cell = not hasattr(original, "wyckoff_sites")
    path = _saved(original, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "loop_\n_geom_torsion_atom_site_label_1\n" in text and "_space_group_symop_id" in text
    loaded = _loaded(path, caplog, cell)
    assert loaded.bonds.pairs == original.bonds.pairs
    assert loaded.bonds.torsions == original.bonds.torsions
    assert loaded.bonds.torsions_source == original.bonds.torsions_source
    _assert_geometric(loaded)
    if name != "unit cell without bonds":
        assert "_httk_geom_torsion_symbol" in text and "_httk_torsions_source_software manual" in text.replace("'", "")
        assert re.search(r"\(150*\) ", text) and re.search(r"\(40*\) ", text) and re.search(r"\s[+-]?sc\s", text)
        assert len(_rows(original)) < 3 or re.search(r"\snone\s", text)
    else:  # no bond loop: a network known to have no bonds, with its torsions; no symbols, the bond source
        assert "_httk_bonds_count 0" in text and "_httk_geom_torsion_symbol" not in text
        assert "_httk_torsions_source_software manual" in text.replace("'", "")
        assert loaded.bonds.source == SOURCE and loaded.bonds.torsions_source is None
        # without any source the torsions read back with the default source of a torsion loop
        unsourced = original._with_bonds(replace(original.bonds, source=None))
        text = _saved(unsourced, tmp_path, "unsourced.cif").read_text(encoding="utf-8")
        assert "_httk_torsions_source" not in text
        assert _loaded(tmp_path / "unsourced.cif", caplog, cell).bonds.torsions_source == _CIF_TORSION_SOURCE


def test_symmetry_codes_name_every_atom_that_is_not_the_written_one(tmp_path: Path) -> None:
    domain = FIXTURE_DOMAINS["diopside (C2/c)"]()
    text = _saved(domain, tmp_path).read_text(encoding="utf-8")
    loop = text.split("_geom_torsion\n")[1].split("_httk_torsions_source")[0]
    codes = [code for line in loop.splitlines() for code in line.split()[4:8]]
    assert "." in codes and any(re.fullmatch(r"\d+_\d\d\d", code) for code in codes)
    assert "11_555" not in codes  # the identity at zero translation is always '.'


@pytest.mark.parametrize(
    "name", ["sarcosine mirrored (left-handed basis)", "sarcosine with a and b swapped (transform det -1)"]
)
def test_a_left_handed_cell_is_written_as_its_mirror_image(
    name: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    domain = FIXTURE_DOMAINS[name]()
    cell = domain.transform.determinant() < 0
    if cell:
        # a domain whose setting transform has determinant -1 has no registered setting: CIF writes its unit cell
        with pytest.raises(ValueError, match="unregistered setting transform"):
            save(domain, tmp_path / "refused.cif")
        domain = _cell(domain)
    original = _human(domain)
    assert original.cell.basis.det() < 0
    with pytest.raises(ValueError, match="cannot exactly represent"):
        save(original, tmp_path / "strict.cif", approximate=False)
    loaded = _loaded(_saved(original, tmp_path), caplog, cell)
    assert loaded.cell.basis.det() > 0
    # the reader's geometry check kept every statement: their angles are the loaded geometry's
    assert loaded.bonds.torsions == tuple(_mirrored_torsion(row) for row in _rows(original))
    assert {row.symbol for row in _rows(loaded)} >= {None, "none"}
    _assert_geometric(loaded)


def test_the_angle_token_keeps_value_and_uncertainty() -> None:
    cases = [(-62.3, 0.4, "-62.3(4)"), (-62.3, 0.15, "-62.30(15)"), (173.74, None, "173.74"), (0.0, 100.0, "0.0(1000)")]
    for angle, uncertainty, token in cases:
        assert _cif_angle(angle, uncertainty) == token
        value, meta = parse_cif_float(token, meta=True)
        assert value == angle and (None if meta["esd"] is None else float(meta["esd"])) == uncertainty
    assert _cif_angle(None, None) is None
    assert _cif_angle(1e-7, 2e-8) == "0.00000010(2)"
    # a value with more digits than its uncertainty keeps them all: the round trip is exact
    assert _cif_angle(-166.34257032127596, 0.08) == "-166.34257032127596(8000000000000)"


@pytest.mark.parametrize(
    ("name", "angles"),
    [
        ("7219102", [(-166.34, 0.08), (-6.45, 0.12), (173.74, 0.08)]),  # sarcosine: atoms as listed
        ("7219077", None),  # symmetry codes on atoms 1 and 4
        ("4134228", None),  # all four code columns and _geom_torsion_publ_flag
    ],
)
def test_a_cod_torsion_loop_is_read_with_its_signs(name: str, angles: Any, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        structure = load(str(FIXTURES / "cod" / f"{name}.cif"))
    assert "torsion" not in caplog.text
    rows = _rows(structure)
    assert structure.bonds.torsions_source == _CIF_TORSION_SOURCE and all(row.symbol is None for row in rows)
    if angles is not None:
        assert sorted((row.angle, row.uncertainty) for row in rows) == sorted(angles)
    for row, angle in zip(rows, _signed_torsions(structure, rows), strict=True):
        assert row.angle is not None and angle is not None
        assert abs(row.angle - angle) <= max(3 * (row.uncertainty or 0), 1), (row, angle)
        assert abs(row.angle) < 1 or abs(row.angle) > 179 or (row.angle > 0) == (angle > 0)


def test_rows_with_no_defined_geometry_are_dropped(caplog: pytest.LogCaptureFixture) -> None:
    # COD 7218683: Cu on an inversion centre makes O1-Cu1-O1' straight; those rows state 130(86) and the like
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        structure = load(str(FIXTURES / "cod" / "7218683.cif"))
    assert len(caplog.records) == 1 and "dropped 3 of the 29 rows of its torsion loop" in caplog.text
    assert structure.bonds.pairs and len(_rows(structure)) == 26


def _sarcosine_text(tmp_path: Path) -> tuple[Any, str, str]:
    original = FIXTURE_DOMAINS["sarcosine (Sohncke)"]()
    text = _saved(original, tmp_path).read_text(encoding="utf-8")
    (line,) = (line for line in text.splitlines() if line.startswith("N1 C12 C11 O11 "))
    return original, text, line


@pytest.mark.parametrize(
    ("edit", "dropped"),
    [
        (lambda line: line.replace("N1 ", "Xx ", 1), 1),  # a label that names no atom site
        (lambda line: line.replace(" . ", " 99_555 ", 1), 1),  # a code that names no operation
        (lambda line: line.replace(" . ", " 1_5x5 ", 1), 1),  # a code that is no code
        (lambda line: re.sub(r"\S+ $", "150.0 ", line), 1),  # an angle that disagrees with the geometry
        (lambda line: re.sub(r"(\S+) $", r"-\1 ", line), 1),  # the angle with the wrong sign
        (lambda line: line + "\n" + re.sub(r"\S+ $", "173.5 ", line), 2),  # a conflicting duplicate: both go
        (lambda line: line + "\n" + line, 0),  # a repeat collapses
        (lambda line: re.sub(r"\S+ $", "? ", line), 1),  # neither angle nor symbol
    ],
)
def test_a_bad_torsion_row_is_dropped_alone_with_one_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, edit: Any, dropped: int
) -> None:
    original, text, line = _sarcosine_text(tmp_path)
    path = tmp_path / "edited.cif"
    path.write_text(text.replace(line, edit(line)), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load(str(path))
    assert loaded.bonds.pairs == original.bonds.pairs and loaded.bonds.torsions_source == original.bonds.torsions_source
    kept = _rows(loaded)
    if dropped:
        assert len(caplog.records) == 1 and f"dropped {dropped} of the" in caplog.text
        assert "CIF block 'structure': dropped" in caplog.text and "the bonds are kept" in caplog.text
        assert len(kept) == len(_rows(original)) - 1 and set(kept) < set(_rows(original))
    else:
        assert not caplog.records and kept == _rows(original)


def test_a_torsion_loop_without_a_bond_network_is_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _original, text, _line = _sarcosine_text(tmp_path)
    head, _bonds, rest = re.split(
        r"(loop_\n_geom_bond_atom_site_label_1\n.*?)(?=loop_\n_geom_torsion)", text, flags=re.DOTALL
    )
    path = tmp_path / "unbonded.cif"
    path.write_text(head + rest, encoding="utf-8")
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        structure = load(str(path))
    assert structure.bonds is None
    assert "ignored its torsion loop, the block has no bond network" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_torsion_columns_of_different_lengths_are_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    original, text, _line = _sarcosine_text(tmp_path)
    path = tmp_path / "split.cif"
    path.write_text(text + "_httk_geom_torsion_symbol t\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load(str(path))
    assert "ignored its torsion loop, the torsion columns have different lengths" in caplog.text
    assert loaded.bonds.pairs == original.bonds.pairs and loaded.bonds.torsions is None


def test_a_malformed_torsion_loop_is_dropped_without_repair(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    original, text, line = _sarcosine_text(tmp_path)
    path = tmp_path / "malformed.cif"
    path.write_text(text.replace(line, line + "\nN1"), encoding="utf-8")  # one value too many
    with caplog.at_level(logging.WARNING):
        loaded = load(str(path))
    assert "dropped malformed auxiliary loop starting with _geom_torsion_atom_site_label_1" in caplog.text
    assert loaded.bonds.pairs == original.bonds.pairs and loaded.bonds.torsions is None


def test_a_translation_a_symmetry_code_cannot_hold_raises(tmp_path: Path) -> None:
    structure = _staggered(Cell([5, 5, 10, 90, 90, 120]))
    far = SiteTorsion((0, 1, 2, 3), (T(6, 0, 0), ONE, ONE, ONE), "t")
    with pytest.raises(ValueError, match=r"sites \(0, 1, 2, 3\): the lattice translation \(6, 0, 0\) of an atom"):
        save(structure._with_bonds(BondNetwork((), torsions=(far,))), tmp_path / "far.cif")


# sha256 of the CIF text these structures wrote before torsions were written to CIF
UNCHANGED = {
    "rutile with bonds, shapes and sources": (
        _rutile,
        "aa182a778b0aed93eed6b8cd98b48c1448407fb1d2c900973dcb4e8705809fa0",
    ),
    "rutile without a network": (
        lambda: _rutile()._with_bonds(None),
        "e747e11189a23d3f6f13ad22a164ef0ee9e56300bb85d6dc64b633f9bc2c208b",
    ),
    "rutile unit cell with bonds and shapes": (
        _bonded_rutile_cell,
        "d511b0f683a6cd49d3861154695105dafd426a6c9345fa1746e50f9eb9e65334",
    ),
}


@pytest.mark.parametrize("name", UNCHANGED)
def test_output_without_torsions_is_unchanged(name: str, tmp_path: Path) -> None:
    make, digest = UNCHANGED[name]
    assert hashlib.sha256(_saved(make(), tmp_path).read_bytes()).hexdigest() == digest


def test_the_dictionary_defines_the_torsion_items(tmp_path: Path) -> None:
    text = _saved(_human(FIXTURE_DOMAINS["sarcosine (Sohncke)"]()), tmp_path).read_text(encoding="utf-8")
    httk = {line.split()[0].lower() for line in text.splitlines() if line.startswith("_httk_")}
    assert "_httk_geom_torsion_symbol" in httk and "_httk_torsions_source_uri" in httk
    definitions = _dictionary_definitions()
    defined = set()
    for values in definitions.values():
        defined |= {str(name).lower() for name in values.get("_alias.definition_id", ())}
    assert httk <= defined
    symbol = definitions["geom_torsion.httk_symbol"]
    assert symbol["_name.category_id"] == ["geom_torsion"]
    assert set(symbol["_enumeration_set.state"]) == _TORSION_SYMBOLS
    assert "httk_torsions" in {str(value).lower() for value in definitions["httk_torsions"]["_name.object_id"]}
