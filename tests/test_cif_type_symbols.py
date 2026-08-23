"""CIF atom-type symbols, pseudo-sites, isotopes, and masses."""

from fractions import Fraction
from pathlib import Path

import pytest
from httk.core import load, save

from httk.atomistic.cif_structures import _CIF_CORE_TYPE_SYMBOLS, _decode_type_symbol

_COD_DEUTERIDE = Path(__file__).resolve().parents[2] / "DATA/COD/cif/1/00/88/1008801.cif"

_CELL = """\
data_symbols
_cell_length_a 5
_cell_length_b 5
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_IT_number 1
loop_
_space_group_symop_operation_xyz
'x,y,z'
"""


def _single_site(symbol: str, *, atom_type_loop: str = "", occupancy: str = "0.5") -> str:
    return (
        _CELL
        + atom_type_loop
        + "loop_\n"
        + "_atom_site_label\n"
        + "_atom_site_type_symbol\n"
        + "_atom_site_fract_x\n"
        + "_atom_site_fract_y\n"
        + "_atom_site_fract_z\n"
        + "_atom_site_occupancy\n"
        + f"site1 {symbol} 0 0 0 {occupancy}\n"
    )


@pytest.mark.parametrize("raw", sorted(_CIF_CORE_TYPE_SYMBOLS))
def test_every_cif_core_type_symbol_is_recognized(raw: str) -> None:
    decoded = _decode_type_symbol(raw, None)

    assert decoded.recognized
    assert decoded.chemical_symbol != "X"


@pytest.mark.parametrize(
    ("raw", "chemical_symbol", "label", "mass"),
    [
        ("D", "H", "D", 2.008),
        ("D0", "H", "D", 2.008),
        ("T", "H", "T", 3.0160),
        ("X", "X", None, None),
        ("Vac", "vacancy", None, 0.0),
        ("Va", "vacancy", None, 0.0),
        ("vacancy", "vacancy", None, 0.0),
    ],
)
def test_special_type_symbols_have_explicit_semantics(
    raw: str, chemical_symbol: str, label: str | None, mass: float | None
) -> None:
    decoded = _decode_type_symbol(raw, None)

    assert decoded.recognized
    assert decoded.chemical_symbol == chemical_symbol
    assert decoded.species_label == label
    assert decoded.mass == mass


@pytest.mark.parametrize("raw", ["M", "R", "LP", "Lp", "dummy", "FeNi"])
def test_unrecognized_type_symbols_warn_and_are_preserved_as_labels(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, raw: str
) -> None:
    path = tmp_path / "unknown.cif"
    path.write_text(_single_site(raw), encoding="utf-8")

    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        structure = load(path)

    species = structure.species[0]
    assert species.chemical_symbols == ("X", "vacancy")
    assert species.labels == (raw, None)
    assert species.name == "site1"
    assert [record.getMessage() for record in caplog.records] == [
        (f"unrecognized CIF atom-type symbol {raw!r}; represented as chemical symbol 'X' with species label {raw!r}")
    ]


def test_repair_normalizes_lowercase_atom_type_symbols(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "lowercase.cif"
    path.write_text(_single_site("c", occupancy="1"), encoding="utf-8")

    assert load(path).species[0].chemical_symbols == ("X",)
    caplog.clear()
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        species = load(path, repair=True).species[0]

    assert species.name == "C"
    assert species.chemical_symbols == ("C",)
    assert "normalized lowercase atom-type symbol 'c' to 'C'" in caplog.text


@pytest.mark.parametrize(
    ("raw", "symbol", "charge"),
    [("Fe4+", "Fe", 4), ("Fe+3", "Fe", 3), ("O-2", "O", -2), ("Na+1", "Na", 1), ("Cl-", "Cl", -1)],
)
def test_charge_spelling_variants_retain_the_element(tmp_path: Path, raw: str, symbol: str, charge: int) -> None:
    path = tmp_path / "charged.cif"
    path.write_text(_single_site(raw, occupancy="1"), encoding="utf-8")

    species = load(path).species[0]

    assert species.chemical_symbols == (symbol,)
    assert species.charges == (charge,)
    assert species.labels is None


def test_a_full_unknown_type_uses_the_cif_symbol_as_its_species_name(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "unknown-full.cif"
    path.write_text(_single_site("dummy", occupancy="1"), encoding="utf-8")

    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        species = load(path).species[0]

    assert species.name == "dummy"
    assert species.chemical_symbols == ("X",)
    assert species.labels == ("dummy",)
    assert "unrecognized CIF atom-type symbol 'dummy'" in caplog.text


def test_a_vacancy_type_loads_with_zero_mass(tmp_path: Path) -> None:
    path = tmp_path / "vacancy.cif"
    path.write_text(_single_site("Vac", occupancy="1"), encoding="utf-8")

    species = load(path).species[0]

    assert species.name == "Vac"
    assert species.chemical_symbols == ("vacancy",)
    assert species.mass == (0.0,)


@pytest.mark.parametrize(
    ("symbol_tag", "mass_tag"),
    [("_atom_type_symbol", "_atom_type_mass"), ("_atom_type.symbol", "_atom_type.atomic_mass")],
)
def test_atom_type_mass_overrides_the_default_isotope_mass(tmp_path: Path, symbol_tag: str, mass_tag: str) -> None:
    path = tmp_path / "mass.cif"
    atom_types = f"loop_\n{symbol_tag}\n{mass_tag}\nD0 2.0141\n"
    path.write_text(_single_site("D0", atom_type_loop=atom_types), encoding="utf-8")

    species = load(path).species[0]

    assert species.chemical_symbols == ("H", "vacancy")
    assert species.concentration == (Fraction(1, 2), Fraction(1, 2))
    assert species.mass == (2.0141, 0.0)
    assert species.labels == ("D", None)


def test_deuterium_disorder_and_stated_mass_survive_cif_roundtrip(tmp_path: Path) -> None:
    source_path = tmp_path / "source.cif"
    atom_types = "loop_\n_atom_type_symbol\n_atom_type_mass\nD 2.0141\n"
    source_path.write_text(_single_site("D", atom_type_loop=atom_types), encoding="utf-8")
    source = load(source_path)
    destination = tmp_path / "roundtrip.cif"

    save(source, destination)
    restored = load(destination)

    assert restored.species == source.species
    assert "_atom_type_mass" in destination.read_text(encoding="utf-8")


def test_deuterium_is_inferred_from_declared_atom_types(tmp_path: Path) -> None:
    path = tmp_path / "inferred-deuterium.cif"
    path.write_text(
        _CELL
        + "loop_\n_atom_type_symbol\nD\nH\n"
        + "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        + "_atom_site_occupancy\nD1 0 0 0 0.7\nH1 0 0 0 0.3\n",
        encoding="utf-8",
    )

    structure = load(path)
    by_name = {species.name: species for species in structure.species}

    assert by_name["D1"].chemical_symbols == ("H",)
    assert by_name["D1"].labels == ("D",)
    assert by_name["D1"].mass == (2.008,)
    assert by_name["H1"].chemical_symbols == ("H",)
    assert by_name["H1"].mass is None
    assert structure.assemblies is not None
    assert structure.assemblies[0].group_probabilities == (Fraction(7, 10), Fraction(3, 10))


@pytest.mark.skipif(not _COD_DEUTERIDE.exists(), reason="workspace-only real-data fixture not present")
def test_cod_1008801_preserves_deuterium_and_disorder(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        structure = load(_COD_DEUTERIDE)

    deuterium = [species for species in structure.species if species.name.startswith("D")]
    assert len(deuterium) == 3
    assert all(species.chemical_symbols == ("H", "vacancy") for species in deuterium)
    assert all(species.mass == (2.008, 0.0) for species in deuterium)
    assert all(species.labels == ("D", None) for species in deuterium)
    assert len(structure.sites) == 54
    assert not [record for record in caplog.records if "unrecognized CIF atom-type symbol" in record.getMessage()]
