"""Bond networks, shapes and their sources written to CIF by ``save`` and read back by ``load``.

``load(save(s)) == s`` does not hold for CIF (coordinate precision, a unit cell reading back as
a P1 ``ASUStructure``, ``Species.original_name`` from unique labels), so the round trips compare
component by component.
"""

import importlib.resources
import logging
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import pytest
from httk.core import load, save

from httk.atomistic import (
    ASUStructure,
    BondNetwork,
    BondSource,
    Cell,
    CellParams,
    CIFStructure,
    FundamentalDomainStructure,
    SiteBond,
    Spacegroup,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    distance_bonds,
)
from httk.atomistic.cif_structures import _CIF_BOND_SOURCE
from httk.atomistic.io.cif import CIFDataItem, CIFLoop, CIFSaveFrame, read_cif
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.shapes import _exchange_symbols

FIXTURES = Path(__file__).parent / "fixtures"
DICTIONARY = importlib.resources.files("httk.atomistic.io.cif") / "cif_httk.dic"
LOGGER = "httk.atomistic.cif_structures"
SOURCE = BondSource("Bonds chosen by hand", "manual", "1.0", "https://example.org/bonds")
SHAPES_SOURCE = BondSource("Shapes chosen by hand", "manual")


def _rutile() -> ASUStructure:
    plain = ASUStructure(
        [[F(4594, 1000), 0, 0], [0, F(4594, 1000), 0], [0, 0, F(2959, 1000)]],
        136,
        [WyckoffSite("a", (), "Ti"), WyckoffSite("f", (F(3048, 10000),), "O")],
        [Species("Ti", ("Ti",), (1,)), Species("O", ("O",), (1,))],
    )
    pairs = distance_bonds(plain, shapes=False).bonds
    assert pairs is not None
    return plain._with_bonds(BondNetwork(pairs.pairs, ("TPR6", "TPY3"), SOURCE, SHAPES_SOURCE))


def _domain_15_c1(tmp_path: Path) -> FundamentalDomainStructure:
    setting = Spacegroup.from_setting("15:c1")
    lines = [
        "data_setting",
        *(f"_cell_{name} {value}" for name, value in zip(("length_a", "length_b", "length_c"), (5, 6, 7), strict=True)),
        *(f"_cell_angle_{name} 90" for name in ("alpha", "beta", "gamma")),
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
        "Si1 Si 0.25 0.0 0.3333",
        "O1 O 0.1 0.2 0.3",
    ]
    source = tmp_path / "setting.cif"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    asu = load(source)
    domain = FundamentalDomainStructure(asu.cell, asu.spacegroup, asu.wyckoff_sites, asu.species, asu.transform)
    network = distance_bonds(domain, shapes=False).bonds
    assert network is not None
    return domain._with_bonds(BondNetwork(network.pairs, ("L2", "A2"), SOURCE, SHAPES_SOURCE))


def _saved(structure: Any, tmp_path: Path, name: str = "out.cif") -> Path:
    path = tmp_path / name
    save(structure, path)
    return path


def _quietly_loaded(path: Path, caplog: pytest.LogCaptureFixture) -> Any:
    """Load ``path`` and check that reading it warned about nothing, the distance check included."""
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load(path)
    assert not caplog.records
    return loaded


def _assert_same_domain(loaded: Any, original: Any) -> None:
    assert loaded.cell == original.cell
    assert loaded.spacegroup == original.spacegroup
    assert loaded.transform == original.transform
    assert loaded.setting().setting == original.setting().setting
    assert [(s.wyckoff, s.free_params, s.species) for s in loaded.wyckoff_sites] == [
        (s.wyckoff, s.free_params, s.species) for s in original.wyckoff_sites
    ]
    assert [(s.name, s.chemical_symbols, s.concentration) for s in loaded.species] == [
        (s.name, s.chemical_symbols, s.concentration) for s in original.species
    ]
    _assert_same_network(loaded.bonds, original.bonds)


def _assert_same_network(loaded: BondNetwork | None, original: BondNetwork | None) -> None:
    assert loaded is not None and original is not None
    assert loaded.pairs == original.pairs
    assert loaded.shapes == original.shapes
    assert loaded.source == original.source
    assert loaded.shapes_source == original.shapes_source


def test_an_asu_round_trips_its_bonds_shapes_and_sources(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    original = _rutile()
    path = _saved(original, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "_space_group_symop_id" in text and "Ti O . 3_555 1.98025" in text
    assert "_httk_shapes_source_software manual" in text.replace("'", "")
    _assert_same_domain(_quietly_loaded(path, caplog), original)
    _assert_same_network(CIFStructure.from_cif(path).resolve().bonds, original.bonds)  # the lossless route


def test_a_fundamental_domain_in_a_non_standard_setting_round_trips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    original = _domain_15_c1(tmp_path)
    assert original.setting().setting == "15:c1"
    _assert_same_domain(_quietly_loaded(_saved(original, tmp_path), caplog), original)


def test_a_unit_cell_round_trips_its_bonds_shapes_and_sources(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cell = UnitcellStructureView(_rutile()).unview()
    assert cell.bonds is not None
    original = cell._with_bonds(replace(cell.bonds, shapes=("TPR6", "OC6", "TPY3", "TP3", "TP3", "TP3")))
    path = _saved(original, tmp_path)
    assert "Ti1 O1 . 1_545 1.94865" in path.read_text(encoding="utf-8")
    loaded = UnitcellStructureView(_quietly_loaded(path, caplog)).unview()
    assert loaded.cell == original.cell
    assert loaded.sites == original.sites
    assert loaded.species_at_sites == original.species_at_sites
    _assert_same_network(loaded.bonds, original.bonds)


def test_a_known_empty_network_round_trips(tmp_path: Path) -> None:
    original = _rutile()._with_bonds(BondNetwork((), source=SOURCE))
    path = _saved(original, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "_httk_bonds_count 0" in text and "_geom_bond" not in text
    _assert_same_network(load(path).bonds, original.bonds)


def test_a_count_of_zero_alone_loads_as_an_empty_network(tmp_path: Path) -> None:
    text = _saved(_rutile()._with_bonds(None), tmp_path).read_text(encoding="utf-8")
    path = tmp_path / "empty.cif"
    path.write_text(text + "_httk_bonds_count 0\n", encoding="utf-8")
    network = load(path).bonds
    assert network == BondNetwork(()) and network is not None and network.source is None


def test_a_count_that_disagrees_with_the_loop_warns_and_the_loop_wins(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    original = _rutile()
    path = _saved(original, tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "_httk_bonds_count 7\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load(path)
    assert "_httk_bonds_count 7 disagrees with its 2 bond row(s)" in caplog.text
    _assert_same_network(loaded.bonds, original.bonds)


@pytest.mark.parametrize(
    ("row", "replacement", "message"),
    [
        ("Ti TPR6", "Xx TPR6", "label 'Xx' names no atom site"),
        ("Ti TPR6", "Ti XYZ6", "unknown shape symbol 'XYZ6'"),
        ("Ti TPR6", "Ti SAPR8", "SAPR8"),
        ("Ti TPR6", "Ti I7", "'I7' is for 7 bonds, but the site has 6"),
        ("O TPY3", "O I2", "unknown shape symbol 'I2'"),
        ("O TPY3", "Ti OC6", "names a site that already has a shape"),
    ],
)
def test_a_bad_shape_row_warns_and_keeps_the_bonds_without_shapes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, row: str, replacement: str, message: str
) -> None:
    original = _rutile()
    path = _saved(original, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert row in text
    path.write_text(text.replace(row, replacement), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        network = load(path).bonds
    assert "left every shape unassigned" in caplog.text and message in caplog.text
    assert network is not None and original.bonds is not None and network.pairs == original.bonds.pairs
    assert network.shapes is None and network.shapes_source is None and network.source == SOURCE


def test_shapes_without_bonds_warn_and_are_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    text = _saved(_rutile()._with_bonds(None), tmp_path).read_text(encoding="utf-8")
    path = tmp_path / "shapes.cif"
    path.write_text(text + "loop_\n_httk_shape_atom_site_label\n_httk_shape_symbol\nTi OC6\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert load(path).bonds is None
    assert "ignored its shape loop, the block lists no bonds" in caplog.text


def test_a_translation_a_symmetry_code_cannot_hold_raises(tmp_path: Path) -> None:
    far = SiteBond(0, 0, AffineOperation("x+6,y,z"))
    structure = UnitcellStructure(
        Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), [[0, 0, 0]], [Species("C", ("C",), (1,))], ["C"]
    )
    with pytest.raises(ValueError, match=r"bond 0-0: its lattice translation \(6, 0, 0\) is outside -5\.\.\+4"):
        save(structure._with_bonds(BondNetwork((far,))), tmp_path / "far.cif")


def test_a_label_named_species_on_several_sites_is_refused(tmp_path: Path) -> None:
    half = Species("Ox", ("O",), (F(1, 2),))
    structure = UnitcellStructure(
        Cell([[3, 0, 0], [0, 3, 0], [0, 0, 3]]), [[0, 0, 0], [F(1, 2), 0, 0]], [half], ["Ox", "Ox"]
    )
    bonded = structure._with_bonds(BondNetwork((SiteBond(0, 1, AffineOperation.identity()),)))
    with pytest.raises(ValueError, match="species 'Ox' is read back by its site label"):
        save(bonded, tmp_path / "half.cif")
    save(structure, tmp_path / "plain.cif")  # without a network the labels may repeat


def test_a_standard_bond_loop_keeps_the_cif_source_and_no_shapes() -> None:
    network = load(FIXTURES / "cif_bonds" / "ordered.cif").bonds
    assert network is not None and network.pairs
    assert network.source == _CIF_BOND_SOURCE and network.shapes is None and network.shapes_source is None


def test_output_without_a_network_is_unchanged(tmp_path: Path) -> None:
    text = _saved(_rutile()._with_bonds(None), tmp_path).read_text(encoding="utf-8")
    assert "_space_group_symop_id" not in text and "_httk_" not in text and "_geom_bond" not in text
    assert "loop_\n_space_group_symop_operation_xyz\n'-x+1/2,y+1/2,-z+1/2' \n" in text
    assert "Ti Ti 0.0000000000000000" in text


def _dictionary_definitions() -> dict[str, dict[str, Any]]:
    (dictionary,) = read_cif(str(DICTIONARY), lossless=True)
    definitions = {}
    for frame in dictionary.items:
        if isinstance(frame, CIFSaveFrame):
            values: dict[str, Any] = {}
            for item in frame.items:
                if isinstance(item, CIFDataItem):
                    values[item.name.lower()] = [item.value]
                elif isinstance(item, CIFLoop):
                    for index, name in enumerate(item.names):
                        values[name.lower()] = [row[index] for row in item.rows]
            definitions[frame.name.lower()] = values
    return definitions


def test_the_dictionary_enumerates_every_allowed_shape_symbol() -> None:
    definitions = _dictionary_definitions()
    symbol = definitions["httk_shape.symbol"]
    assert symbol["_definition.id"] == ["_httk_shape.symbol"]
    assert tuple(symbol["_enumeration_set.state"]) == _exchange_symbols()


def test_the_dictionary_defines_every_item_the_writer_emits(tmp_path: Path) -> None:
    cell = Cell(CellParams((F(10, 3), F(10, 3), F(10, 3), F(200, 3), F(200, 3), F(200, 3))))
    exact = UnitcellStructure(cell, [[F(1, 3), F(1, 3), F(1, 3)]], [Species("C", ("C",), (F(1, 3),))], ["C"])
    path = tmp_path / "exact.cif"
    save(exact, path, exact_companions=True)
    written = {
        line.split()[0].lower() for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("_")
    }
    for structure in (_rutile(), _rutile()._with_bonds(BondNetwork((), source=SOURCE))):
        text = _saved(structure, tmp_path).read_text(encoding="utf-8")
        written |= {line.split()[0].lower() for line in text.splitlines() if line.startswith("_")}
    httk = {name for name in written if name.startswith("_httk_")}
    assert len(httk) == 21
    defined = set()
    for values in _dictionary_definitions().values():
        defined |= {str(name).lower() for name in values.get("_alias.definition_id", ())}
    assert httk <= defined


def test_load_reads_the_exact_cell_companions(tmp_path: Path) -> None:
    structure = ASUStructure(
        [[F(10, 3), 0, 0], [0, F(10, 3), 0], [0, 0, 3]], 123, [WyckoffSite("a", (), "C")], [Species("C", ("C",), (1,))]
    )
    path = tmp_path / "cell.cif"
    save(structure, path, exact_companions=True)
    assert "_httk_cell_length_a_exact" in path.read_text(encoding="utf-8")
    assert load(path).cell.lengths[0] == F(10, 3)


@pytest.mark.parametrize(("token", "warns"), [("?", False), (".", False), ("x", True)])
def test_an_unknown_count_is_absent_and_a_malformed_one_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, token: str, warns: bool
) -> None:
    original = _rutile()
    path = _saved(original, tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + f"_httk_bonds_count {token}\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        loaded = load(path)
    assert ("_httk_bonds_count" in caplog.text) is warns
    _assert_same_network(loaded.bonds, original.bonds)


def test_an_irregular_shape_is_written_as_i_and_reads_back_with_its_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rutile = _rutile()
    assert rutile.bonds is not None
    original = rutile._with_bonds(replace(rutile.bonds, shapes=("I6", "I3")))
    path = _saved(original, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "Ti I \nO I \n" in text and "I6" not in text and "I3" not in text
    _assert_same_domain(_quietly_loaded(path, caplog), original)
    path.write_text(text.replace("Ti I \n", "Ti I6 \n"), encoding="utf-8")  # a matching count is accepted
    _assert_same_network(_quietly_loaded(path, caplog).bonds, original.bonds)
