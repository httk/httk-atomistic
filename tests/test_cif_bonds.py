"""Reading a CIF block's ``_geom_bond_`` loop into the bond network of the loaded structure.

The COD fixtures are unmodified entries; the ``cif_bonds`` fixtures are small synthetic P2_1/c
files written for these tests.
"""

import io
import logging
import math
from collections.abc import Sequence
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from httk.core import load
from test_protonet_molecular import MOLECULAR

from httk.atomistic import ASUStructure, SiteBond, canonical_protonetstructure, cif_structures
from httk.atomistic.io.cif import read_cif
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.xyz import operation_from_xyz

FIXTURES = Path(__file__).parent / "fixtures"
COD = FIXTURES / "cod"
BONDS = FIXTURES / "cif_bonds"
REPAIR = {"4002853", "4002910"}  # their occupancies load only with repair=True
LOGGER = "httk.atomistic.cif_structures"

# COD entry -> (loop rows, bond orbits, canonical protonetstructure label). The listed bond loop
# is the refinement program's choice: 7219077 and 4134228 list no bonds to H, so their labels
# differ from the covalent-cutoff labels of test_protonet_molecular.
NETWORK = {
    "7219077": (15, 13, "A9B9CD_mC160_15_2e8f_9f_f_f__1~2.20_2~21_4~5.6.7_20~3.7_8~10.5_9~10.6_3~3@3:C-H-N-S"),
    "4002910": (
        37,
        11,
        (
            "A8B2C2DE_mC28_12_4i_i_i_a_c__6~3.3@4.2.4@4_7~1@4.2_8~3@4@556.4@2_5SPY5~1.1@4@554.2@7_1SPY5_3SPY5"
            "_2I4_4TPY3:Se-Bi0.94-Bi0.755Mn0.245-Mn-In0.505Mn0.495"
        ),
    ),
    "7217843": (42, 42, MOLECULAR["7217843"][1]),
    "4134228": (
        13,
        13,
        "A17B9C3D_mP120_14_17e_9e_3e_e__25~24.28.30_26~27.28.29_27~22.24_18~19.20_19~21_20~22_21~23:H-C-N-O",
    ),
    "7218683": (25, 23, MOLECULAR["7218683"][1]),
    "4002853": (
        38,
        11,
        (
            "A8B2C2DE_mC28_12_4i_i_i_a_c__6~3.3@4.2.4@4_7~1@4.2_8~3@4@556.4@2_5SPY5~1.1@4@554.2@7_1SPY5_3SPY5"
            "_4I3:Se-Bi-Bi0.96Cd0.04-Bi0.12Cd0.826-Cd0.077Pb0.923"
        ),
    ),
    "4002923": (16, 1, "A6B2C_cF72_227_f_c_b__2~1@61:F0.833O0.167-Fe-O"),
}

# COD entry -> (loop rows, bond orbits). Their constrained (riding) X-H distances, stated to
# 0.0001 Å without an su, are reproduced only within the precision of the coordinates (up to
# 0.001 Å off), and their networks give the covalent-cutoff labels of test_protonet_molecular.
RIDING = {
    "7217003": (40, 40),
    "7219102": (12, 12),
    "4326344": (25, 25),
    "7217521": (18, 16),
    "4131686": (21, 21),
    "7219855": (24, 24),
}
NETWORK |= {entry: (rows, orbits, MOLECULAR[entry][1]) for entry, (rows, orbits) in RIDING.items()}


def _load(entry: str) -> ASUStructure:
    return load(str(COD / f"{entry}.cif"), repair=entry in REPAIR)


def _rows(entry: str) -> int:
    block = load(str(COD / f"{entry}.cif"), raw=True, repair=entry in REPAIR)["blocks"][0]
    return len(block["_httk_atomistic_bonds"]["atom_site_label_1"])


def _label(structure: ASUStructure) -> str:
    return str(canonical_protonetstructure(structure, timeout=None).label)


@pytest.mark.parametrize("entry", list(NETWORK))
def test_a_cod_bond_loop_becomes_the_network(entry: str, caplog: pytest.LogCaptureFixture) -> None:
    rows, orbits, _ = NETWORK[entry]
    with caplog.at_level("WARNING", logger=LOGGER):
        structure = _load(entry)
    assert not [record for record in caplog.records if "bond loop" in record.getMessage()]
    assert _rows(entry) == rows
    assert structure.bonds is not None and len(structure.bonds.pairs) == orbits


# COD entry -> the warning for the Wyckoff sites its listed loop leaves without a bond.
UNBONDED = {
    "7219077": "9 H sites have no bond in the listed loop",
    "4134228": "17 H sites have no bond in the listed loop",
    "4002923": "1 O site has no bond in the listed loop",
}


@pytest.mark.parametrize("entry", list(NETWORK))
def test_sites_a_bond_loop_leaves_unbonded_are_warned_about(entry: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger=LOGGER):
        _load(entry)
    warned = [record.getMessage() for record in caplog.records if "no bond in the listed loop" in record.getMessage()]
    assert warned == ([f"CIF block {entry!r}: {UNBONDED[entry]}"] if entry in UNBONDED else [])


def test_unbonded_sites_are_counted_per_element(caplog: pytest.LogCaptureFixture) -> None:
    structure = load(str(BONDS / "disordered.cif"))  # site 0 is Co1/Fe1, sites 1 and 2 are O

    def warned(*bonds: tuple[int, int]) -> list[str]:
        caplog.clear()
        ends = tuple(SimpleNamespace(site=site, partner=partner) for site, partner in bonds)
        stand_in = SimpleNamespace(
            wyckoff_sites=structure.wyckoff_sites, species=structure.species, bonds=SimpleNamespace(pairs=ends)
        )
        with caplog.at_level("WARNING", logger=LOGGER):
            cif_structures._warn_unbonded_sites("b", cast(Any, stand_in))
        return [record.getMessage() for record in caplog.records]

    assert warned((0, 1)) == ["CIF block 'b': 1 O site has no bond in the listed loop"]
    assert warned((1, 2)) == ["CIF block 'b': 1 Co/Fe site has no bond in the listed loop"]
    assert warned((0, 1), (0, 2)) == warned() == []


@pytest.mark.parametrize("occupancy", ["0", "0.0004", "0.5"])
def test_an_unbonded_site_of_low_occupancy_warns_and_loads(occupancy: str, caplog: pytest.LogCaptureFixture) -> None:
    text = (BONDS / "ordered.cif").read_text()
    row = "O2 O 0.12 0.1 0.75 1 d\n"
    with caplog.at_level("WARNING", logger=LOGGER):
        structure = load(io.StringIO(text.replace(row, f"{row}Li3 Li 0.4 0.4 0.1 {occupancy} d\n")), format="cif")
    assert structure.bonds == _bonds("ordered.cif")
    warned = [record.getMessage() for record in caplog.records if "no bond" in record.getMessage()]
    assert warned == ["CIF block 'ordered': 1 Li site has no bond in the listed loop"]


@pytest.mark.parametrize("name", ["ordered.cif", "disordered.cif", "empty_loop.cif", "no_loop.cif"])
def test_a_fully_bonded_or_empty_loop_gives_no_unbonded_warning(name: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger=LOGGER):
        load(str(BONDS / name))
    assert not [record for record in caplog.records if "no bond" in record.getMessage()]


@pytest.mark.parametrize("entry", list(NETWORK))  # each under 5 s
def test_a_cod_bond_loop_gives_a_protonetstructure_label(entry: str) -> None:
    assert _label(_load(entry)) == NETWORK[entry][2]


def _point(tokens: Sequence[str]) -> tuple[F, ...]:
    return tuple(F(Decimal(token.partition("(")[0])) for token in tokens)


def _apply(matrix: Any, vector: Any, point: tuple[F, ...]) -> tuple[F, ...]:
    return tuple(F(sum((matrix[r][k] * point[k] for k in range(3)), vector[r])) for r in range(3))


def _key(p: tuple[F, ...], q: tuple[F, ...]) -> tuple[F, ...]:
    """An unordered atom pair modulo lattice translations."""
    return min(tuple(x - math.floor(s) for x, s in zip((*a, *b), (*a, *a), strict=True)) for a, b in ((p, q), (q, p)))


def test_the_network_expands_to_the_pairs_the_file_states() -> None:
    """Every row of 7219077, recomputed from the file's text, is an expanded bond of the loaded structure."""
    ((_, block),), _ = read_cif(str(COD / "7219077.cif"))
    operations = [operation_from_xyz(xyz) for xyz in block["symmetry_equiv_pos_as_xyz"]]
    atoms = {
        label: _point(xyz)
        for label, *xyz in zip(*(block[f"atom_site_{name}"] for name in ("label", "fract_x", "fract_y", "fract_z")))
    }
    cell = [float(block[f"cell_{name}"].partition("(")[0]) for name in ("length_a", "length_b", "length_c")]
    beta = math.radians(float(block["cell_angle_beta"].partition("(")[0]))  # C2/c: alpha = gamma = 90
    stated = []
    for first, second, distance, code in zip(
        *(
            block[f"geom_bond_{name}"]
            for name in ("atom_site_label_1", "atom_site_label_2", "distance", "site_symmetry_2")
        ),
        strict=True,
    ):
        q = atoms[second]
        if code != ".":
            n, _, klm = code.partition("_")
            operation = operations[int(n) - 1]
            vector = [x + int(t) - 5 for x, t in zip(operation.vector.to_fractions(), klm, strict=True)]
            q = _apply(operation.matrix.to_fractions(), vector, q)
        p = atoms[first]
        a, b, c = (float(y - x) * length for x, y, length in zip(p, q, cell, strict=True))
        cartesian = (a + c * math.cos(beta), b, c * math.sin(beta))
        assert math.dist((0, 0, 0), cartesian) == pytest.approx(float(distance.partition("(")[0]), abs=1e-3)
        stated.append(_key(p, q))

    structure = _load("7219077")
    group, sites = structure.spacegroup, structure.wyckoff_sites
    elements = [(g.matrix.to_fractions(), g.vector.to_fractions()) for g in group.symmetry_operations]

    def reference(index: int) -> tuple[F, ...]:
        site = sites[index]
        return tuple(group.wyckoff_position(site.wyckoff).representative.coordinate(site.free_params).to_fractions())

    expanded = set()
    assert structure.bonds is not None
    for bond in structure.bonds.pairs:
        p = reference(bond.site)
        q = _apply(bond.operation.matrix.to_fractions(), bond.operation.vector.to_fractions(), reference(bond.partner))
        expanded |= {_key(_apply(*g, p), _apply(*g, q)) for g in elements}
    assert set(stated) <= expanded


def _bonds(name: str) -> Any:
    return load(str(BONDS / name)).bonds


def test_disordered_rows_named_in_the_loop_map_to_their_merged_site() -> None:
    structure = load(str(BONDS / "disordered.cif"))
    # Fe1 and Co1 share one site, so their bonds to O1 are one orbit; the dummy row comes first
    assert structure.bonds == _bonds("ordered.cif") and len(structure.bonds.pairs) == 2
    assert structure.species[0].name == "Co1/Fe1"


def test_a_non_identity_first_end_gives_the_same_bonds() -> None:
    assert _bonds("site_symmetry_1.cif") == _bonds("ordered.cif")


def test_symmetry_codes_use_the_operation_ids(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger=LOGGER):
        assert _bonds("symop_ids.cif") == _bonds("ordered.cif")
    assert caplog.records == []


def test_no_loop_and_an_empty_loop_differ() -> None:
    assert _bonds("no_loop.cif") is None
    assert _bonds("empty_loop.cif").pairs == ()


def test_the_coordinates_written_precision_is_part_of_the_tolerance(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Fe1-O1 is stated 0.002 Å off without an su: within the precision of its 4-decimal coordinates only."""
    assert _bonds("coordinate_precision.cif") == _bonds("ordered.cif")
    monkeypatch.setattr(cif_structures, "_row_coordinate_precisions", lambda data: [0.0] * len(data["labels"]))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _bonds("coordinate_precision.cif") is None
    (record,) = caplog.records
    assert "row 1 (Fe1 O1 ? . 1.0465): the atoms are 1.044" in record.getMessage()


@pytest.mark.parametrize(
    "name,reason",
    [
        ("unknown_code.cif", "row 2 (Fe1 O2 ? 4-554 1.2588): symmetry code '4-554' is not of the form n or n_klm"),
        ("distance_mismatch.cif", "row 2 (Fe1 O2 ? 4_554 1.634(5)): the atoms are 1.258831 Å apart, not 1.634(5)"),
    ],
)
def test_an_unreadable_loop_warns_and_gives_no_network(
    name: str, reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        structure = load(str(BONDS / name))
    (record,) = caplog.records
    assert record.getMessage().startswith(f"CIF block '{name.removesuffix('.cif')}': ignored its bond loop, {reason}")
    assert structure.bonds is None
    assert structure == load(str(BONDS / "no_loop.cif"))


def test_unknown_labels_and_dummy_atoms_are_unreadable(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    text = (BONDS / "disordered.cif").read_text()
    for wrong, reason in (("Xx9", "label 'Xx9' names no atom site"), ("Du1", "label 'Du1' names a dummy atom site")):
        path = tmp_path / f"{wrong}.cif"
        path.write_text(text.replace("Co1 O2 1.2588", f"Co1 {wrong} 1.2588"))
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            structure = load(str(path))
        assert structure.bonds is None
        (record,) = caplog.records
        assert record.getMessage().endswith(reason)


def _variant(tmp_path: Path, caplog: pytest.LogCaptureFixture, text: str, *, repair: bool = False) -> ASUStructure:
    """Load CIF text, collecting its warnings in ``caplog``."""
    path = tmp_path / "variant.cif"
    path.write_text(text)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        return load(str(path), repair=repair)


ORDERED = (BONDS / "ordered.cif").read_text()
HEAD = ORDERED.partition("loop_\n_geom_bond_atom_site_label_1")[0]
LOOP = "loop_\n_geom_bond_atom_site_label_1\n_geom_bond_atom_site_label_2\n_geom_bond_distance\n_geom_bond_site_symmetry_2\n"


@pytest.mark.parametrize("distance", ["nan", "Infinity"])
def test_a_distance_that_is_not_finite_is_unreadable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, distance: str
) -> None:
    assert _variant(tmp_path, caplog, HEAD + LOOP + f"Fe1 O1 {distance} .\n").bonds is None
    assert caplog.records[-1].getMessage().endswith(f"the distance {distance!r} is not a finite number")


def test_an_exponent_distance_with_an_su_is_read(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert _variant(tmp_path, caplog, HEAD + LOOP + "Fe1 O1 10.445(1)E-1 .\n").bonds is not None


def test_bond_columns_of_different_lengths_are_unreadable(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    loop = "loop_\n_geom_bond_atom_site_label_1\n_geom_bond_atom_site_label_2\nFe1 O1\nFe1 O2\n_geom_bond_distance 1.0445\n"
    assert _variant(tmp_path, caplog, HEAD + loop).bonds is None
    assert caplog.records[-1].getMessage().endswith("ignored its bond loop, the bond columns have different lengths")


def test_a_malformed_bond_loop_is_dropped_without_repair(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    structure = _variant(tmp_path, caplog, HEAD + LOOP + "Fe1 O1#c 1.0445 .\nFe1 O1\n")  # 6 values, 4 columns
    assert "dropped malformed auxiliary loop starting with _geom_bond_atom_site_label_1" in caplog.text
    with pytest.raises(ValueError, match="mismatched value counts"):
        read_cif(str(tmp_path / "variant.cif"))  # the general reader stays strict
    assert structure.bonds is None and structure == _variant(tmp_path, caplog, HEAD)


def test_codes_need_the_blocks_own_operations(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    hall = HEAD.replace("_space_group_IT_number 14", "_space_group_name_Hall '-P 2ybc'")
    without = (
        hall.partition("loop_\n_space_group_symop_operation_xyz")[0]
        + "loop_\n_atom_site_label"
        + (hall.partition("loop_\n_atom_site_label")[2])
    )
    # repair generates the operations from the Hall symbol: only codes naming none are readable
    assert _variant(tmp_path, caplog, without + LOOP + "Fe1 O1 1.0445 .\n", repair=True).bonds is not None
    assert _variant(tmp_path, caplog, without + LOOP + "Fe1 O2 1.2588 4_554\n", repair=True).bonds is None
    assert "cannot be resolved: the block lists no symmetry operations" in caplog.records[-1].getMessage()
    duplicated = (BONDS / "symop_ids.cif").read_text().replace("3 -x,-y,-z", "1 -x,-y,-z")
    assert _variant(tmp_path, caplog, duplicated).bonds is None
    assert "or ids that do not name them one to one" in caplog.records[-1].getMessage()


def test_labels_match_exactly_then_ignoring_case(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert _variant(tmp_path, caplog, HEAD + LOOP + "FE1 o1 1.0445 .\n").bonds.pairs == _bonds("ordered.cif").pairs[:1]
    twins = HEAD.replace("O2 O 0.12", "Ox2 O 0.4 0.4 0.4 1 d\nOX2 O 0.12")
    assert _variant(tmp_path, caplog, twins + LOOP + "Fe1 OX2 1.2588 4_554\n").bonds is not None  # exact wins
    assert _variant(tmp_path, caplog, twins + LOOP + "Fe1 ox2 1.2588 4_554\n").bonds is None
    assert caplog.records[-1].getMessage().endswith("label 'ox2' names several atom sites")


def test_a_loop_of_placeholders_gives_no_network_silently(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert _variant(tmp_path, caplog, HEAD + LOOP + "? ? ? ?\n").bonds is None
    assert caplog.records == []


def test_a_merged_row_written_at_another_orbit_member() -> None:
    """Co1 is written at -x,-y,-z of Fe1; its two bonds to O1 are Fe1's."""
    structure = load(str(BONDS / "merged_other_member.cif"))
    assert structure.bonds.pairs == (SiteBond(0, 1, AffineOperation.identity()),)


def test_a_snapped_row_written_as_a_translated_copy() -> None:
    """Fe1 written near (0, 1, 1/2) snaps onto 2c (0, 0, 1/2); its bond keeps the translation."""
    structure = load(str(BONDS / "snapped_translated.cif"))
    assert structure.wyckoff_sites[0].wyckoff == "c"
    assert structure.bonds.pairs == (SiteBond(0, 1, operation_from_xyz("x,y-1,z")),)
