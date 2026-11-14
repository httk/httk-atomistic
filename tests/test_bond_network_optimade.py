"""Serving bond networks as ``_httk_bonds`` and reading them back.

A client must be able to rebuild every bond of a crystal from the served floats alone: the served
positions, lattice and symmetry operations, and the bond rows that index those operations.
"""

import copy
import importlib.resources
import json
import logging
import math
from fractions import Fraction as F
from functools import cache
from pathlib import Path
from typing import Any

import httk.core
import pytest
from httk.core import FracVector
from httk.core.optimade import (
    IncompleteOptimadeResourceError,
    OptimadeDocument,
    OptimadeResource,
    OptimadeSchemaSnapshot,
)
from httk.core.storage import stored_property_projections
from test_bond_network_records import RECORDS, SHAPES_SOURCE, SOURCE, _rocksalt, _with

from httk.atomistic import (
    ASUStructure,
    BondNetwork,
    CIFStructure,
    FundamentalDomainStructure,
    NumericUnitcellStructure,
    OptimadeStructure,
    SettingTransform,
    Spacegroup,
    Species,
    StructureEntryProvider,
    UnitcellStructureView,
    WyckoffSite,
    distance_bonds,
)
from httk.atomistic.entries import bonds as bonds_module
from httk.atomistic.entries.bonds import (
    BOND_PROPERTY_KEYS,
    _bond_codes,
    _check_bonds_value,
    bond_definitions,
    bond_properties,
)
from httk.atomistic.entries.structures import StructureEntry
from httk.atomistic.models.structure.asu import _affine, _inv, _mul, _op
from httk.atomistic.models.structure.network import SiteBond
from httk.atomistic.models.structure.record import RecordStructure
from httk.atomistic.models.structure.semantics import StructureSymmetry
from httk.atomistic.storage.records import (
    ASUStructureRecord,
    FundamentalDomainStructureRecord,
    NumericUnitcellStructureRecord,
    UnitcellStructureRecord,
    _domain_structure_record_from_value,
    _unitcell_record_from_structure,
)
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.shapes import _exchange_symbols

FIXTURES = Path(__file__).parent / "fixtures"
BONDS_ID = "https://schemas.httk.org/defs/v0.1/properties/chemistry/bonds"
SI, O = Species("Si", ("Si",), (1,)), Species("O", ("O",), (1,))


@cache
def _schema() -> dict[str, Any]:
    text = importlib.resources.files("httk.registry.schemas.atomistic").joinpath("bonds.json").read_text()
    return json.loads(text)


def _validate(value: Any) -> None:
    """Validate a served value against ``bonds.json``; httk serves rows of six and never ``pair_rows``."""
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(_schema()).validate(value)
    if value is not None:
        assert all(len(row) == 6 for row in value["pairs"])
        assert value["shapes"] is None or set(value["shapes"]) == {"sites", "symbols"}


def _operations(row: dict[str, Any]) -> list[AffineOperation] | None:
    served = row["space_group_symmetry_operations_xyz"]
    return None if served is None else [AffineOperation(xyz) for xyz in served]


def _is_cell(row: dict[str, Any]) -> bool:
    return row["site_coordinate_span"] in ("unit_cell", "molecular_unit_cell")


def _row_operation(row: dict[str, Any], pair: list[int]) -> AffineOperation:
    """The operation of one served row: ``listed[op]`` followed by the translation ``t``."""
    g = AffineOperation((row["space_group_symmetry_operations_xyz"] or ["x,y,z"])[pair[2]])
    return AffineOperation(g.matrix, g.vector + FracVector(pair[3:6]))


def _load(name: str) -> Any:
    return httk.core.load(str(FIXTURES / name))


def _outside() -> ASUStructure:
    """A P-1 site whose reference atom lies outside [0, 1)."""
    sites = [
        WyckoffSite("i", (F(-1, 10), F(6, 5), F(-3, 10)), "Si"),
        WyckoffSite("i", (F(1, 10), F(13, 10), F(-1, 5)), "O"),
    ]
    return distance_bonds(ASUStructure([[5, 0, 0], [0, 6, 0], [-1, 0, 7]], 2, sites, [SI, O]))


_MONOCLINIC_SITES = (WyckoffSite("e", (F(1, 5), F(1, 4), F(3, 10)), "Si"), WyckoffSite("a", (), "O"))


def _monoclinic(transform: SettingTransform | None = None) -> ASUStructure:
    """Si on a general position and O on an inversion centre of P 1 21/c 1, in the cell of ``transform``."""
    basis: Any = [[5, 0, 0], [0, 5, 0], [-1, 0, 5]]
    if transform is None:
        return distance_bonds(ASUStructure(basis, 14, _MONOCLINIC_SITES, [SI, O]))
    return distance_bonds(ASUStructure(transform.basis_to_setting(basis), 14, _MONOCLINIC_SITES, [SI, O], transform))


def _monoclinic_setting() -> ASUStructure:
    """P 1 21/n 1, a tabulated non-standard setting of P 1 21/c 1, reached through a transform."""
    return _monoclinic(Spacegroup.from_hm_entry("P 1 21/n 1").transform_from_standard)


def _other_member() -> ASUStructure:
    """The monoclinic structure with Si served as its inversion image, another member of its orbit."""
    plain = _monoclinic()
    si, o = plain.wyckoff_sites
    image = tuple(-x for x in si.free_params.to_fractions())
    sites = [WyckoffSite("e", si.free_params, "Si", FracVector(image)), o]
    return ASUStructure(plain.cell, 14, sites, [SI, O], bonds=plain.bonds)


# a shift that moves reference atoms out of [0, 1), so the served positions are wrapped ones
_SHIFT = (F(7, 8), F(3, 4), F(1, 2))


def _supercell_monoclinic() -> ASUStructure:
    """The monoclinic structure in a cell doubled along a, shifted: two lattice cosets (M1)."""
    return _monoclinic(SettingTransform([[F(1, 2), 0, 0], [0, 1, 0], [0, 0, 1]], _SHIFT))


def _sheared_monoclinic() -> ASUStructure:
    """The monoclinic structure in a sheared, shifted, untabulated own cell (M2)."""
    return _monoclinic(SettingTransform([[1, 1, 0], [0, 1, 0], [0, 0, 1]], _SHIFT))


def _transformed_rocksalt(transform: SettingTransform) -> ASUStructure:
    plain = _rocksalt()
    basis = transform.basis_to_setting(plain.cell.basis)
    return distance_bonds(ASUStructure(basis, 225, plain.wyckoff_sites, plain.species, transform))


def _supercell_rocksalt() -> ASUStructure:
    """Rock salt in a cell doubled along a: an untabulated transform with two lattice cosets (M1)."""
    return _transformed_rocksalt(SettingTransform([[F(1, 2), 0, 0], [0, 1, 0], [0, 0, 1]], (0, 0, 0)))


SERVED = {
    "sites served as other orbit members (cod/4002923)": lambda: _load("cod/4002923.cif"),
    "sites served as other orbit members (cod/7219077)": lambda: _load("cod/7219077.cif"),
    "a site served as another orbit member": _other_member,
    "a reference atom outside [0, 1)": _outside,
    "inexact representatives (cif_bonds/snapped_translated)": lambda: _load("cif_bonds/snapped_translated.cif"),
    "a tabulated non-standard setting": _monoclinic_setting,
    "a supercell transform": _supercell_monoclinic,
    "an untabulated sheared transform": _sheared_monoclinic,
    "a unit cell": lambda: _with(UnitcellStructureView(_rocksalt()).unview(), ("OC6",) * 8, SOURCE, SHAPES_SOURCE),
}


def _row(structure: Any) -> dict[str, Any]:
    return next(iter(StructureEntryProvider({"x": structure}).records("structures")))


def _stored(record: Any) -> Any:
    """Save a record to an SQLite store and serve ``_httk_bonds`` from the fetched record."""
    pytest.importorskip("httk.store")
    pytest.importorskip("sqlalchemy")
    from httk.store import Backend, EntryIdScheme, SqlStore

    with Backend.sqlite() as database:
        store = SqlStore(database, entry_ids=EntryIdScheme("httk.test", "1"), entry_records=RECORDS)
        fetched = store.fetch(type(record), store.save(record))
        return stored_property_projections(type(record))["_httk_bonds"].response(fetched)


def _record(structure: Any) -> Any:
    if isinstance(structure, FundamentalDomainStructure):
        return _domain_structure_record_from_value(structure)
    return _unitcell_record_from_structure(structure)


# ---------------------------------------------------------------------------------------------
# the definition and the served value
# ---------------------------------------------------------------------------------------------
def test_the_definition_is_vendored_and_not_a_default_response_field() -> None:
    definition = bond_definitions()["_httk_bonds"]
    assert definition.definition_id == BONDS_ID
    assert definition.requirements["response-level"] == "should not"
    extended = StructureEntry.entry_type_definition()
    assert set(BOND_PROPERTY_KEYS) <= set(extended.properties)
    assert extended.properties["_httk_bonds"].requirements == definition.requirements
    assert _schema()["$id"] == BONDS_ID


@pytest.mark.parametrize("name", SERVED)
def test_the_served_value_validates_through_the_provider_and_the_store(name: str) -> None:
    structure = SERVED[name]()
    assert structure.bonds is not None
    row = _row(structure)
    value = row["_httk_bonds"]
    _validate(value)
    _check_bonds_value(value, row["nsites"], _operations(row), cell=_is_cell(row))
    assert value == bond_properties(structure)["_httk_bonds"]
    assert value["source"] == bond_properties(structure)["_httk_bonds"]["source"]
    stored = _stored(_record(structure))
    _validate(stored)
    assert stored == value
    if isinstance(structure, ASUStructure):  # the fundamental-domain record serves the same value
        domain = FundamentalDomainStructure(
            structure.cell,
            structure.spacegroup,
            structure.wyckoff_sites,
            structure.species,
            structure.transform,
            structure.coordinate_precision,
            bonds=structure.bonds,
        )
        assert isinstance(_record(domain), FundamentalDomainStructureRecord)
        from_domain = _stored(_record(domain))
        _validate(from_domain)
        assert from_domain == value
    pytest.importorskip("httk.store")
    from httk.store.validation import validate_record

    provider = StructureEntryProvider({"x": structure})
    keys = provider.property_keys("structures")
    validate_record(provider.entry_types()["structures"], {name: row[key] for name, key in keys.items()})


def test_the_unit_cell_fast_path_equals_the_structure_path() -> None:
    cell = SERVED["a unit cell"]()
    record = _unitcell_record_from_structure(cell)
    served = stored_property_projections(UnitcellStructureRecord)["_httk_bonds"].response(record)
    assert served == bond_properties(UnitcellStructureView(record))["_httk_bonds"]
    assert served["source"] == {"description": "hand-assigned", "software": "test", "version": "1", "uri": None}
    assert served["shapes_source"] == {"description": None, "software": "shapes", "version": None, "uri": None}


def test_the_definition_lays_out_rows_and_sparse_shapes() -> None:
    definition = _schema()
    assert set(definition["properties"]) == {"pairs", "shapes", "source", "shapes_source"}
    assert definition["x-optimade-definition"]["version"] == "0.2.0"
    assert "operation_xyz" not in json.dumps(definition["properties"])
    assert definition["properties"]["pairs"]["items"]["x-optimade-dimensions"]["sizes"] == [6]
    shapes = definition["properties"]["shapes"]
    assert shapes["required"] == ["sites", "symbols"] and set(shapes["properties"]) == {"sites", "symbols", "pair_rows"}
    assert shapes["properties"]["symbols"]["items"]["enum"] == list(_exchange_symbols())
    assert all(term in definition["title"] + definition["description"] for term in ("coordination polyhedra", "cages"))


ILLUSTRATION = {
    "pairs": [[1, 2, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0]],
    "shapes": {"sites": [1], "symbols": ["OC6"], "pair_rows": [[0]]},
    "source": None,
    "shapes_source": None,
}


def test_the_definition_accepts_the_illustration_and_rejects_unknown_symbols() -> None:
    pytest.importorskip("httk.store")
    from httk.store.validation import PropertyValidationError, validate_property

    definition = bond_definitions()["_httk_bonds"]
    validate_property(definition, ILLUSTRATION)
    validate_property(definition, {**ILLUSTRATION, "shapes": {**ILLUSTRATION["shapes"], "pair_rows": [None]}})
    validate_property(definition, {**ILLUSTRATION, "shapes": {"sites": [1], "symbols": ["SAPR8"]}})
    validate_property(definition, {**ILLUSTRATION, "shapes": {"sites": [1], "symbols": ["I"]}})
    for wrong in (
        {**ILLUSTRATION, "shapes": {"sites": [1], "symbols": ["I2"]}},
        {**ILLUSTRATION, "shapes": {"sites": [1], "symbols": ["I5"]}},  # exchange formats write irregular as I
        {**ILLUSTRATION, "shapes": {"sites": [1], "symbols": [None]}},
        {**ILLUSTRATION, "shapes": {"symbols": ["OC6"]}},
        {**ILLUSTRATION, "pairs": [[1, 2, 0, 0, 0, "0"]]},
        {**ILLUSTRATION, "pairs": [{"site": 1, "partner": 2, "operation_xyz": "x,y,z"}]},
        {**ILLUSTRATION, "source": "text"},
        {**ILLUSTRATION, "shapes_source": 7},
    ):
        with pytest.raises(PropertyValidationError):
            validate_property(definition, wrong)


def test_an_irregular_shape_is_served_as_i() -> None:
    cell = _with(UnitcellStructureView(_rocksalt()).unview(), ("I6",) * 8, SOURCE)
    value = _row(cell)["_httk_bonds"]
    assert value["shapes"] == {"sites": list(range(8)), "symbols": ["I"] * 8}
    _validate(value)
    stored = stored_property_projections(UnitcellStructureRecord)["_httk_bonds"].response(
        _unitcell_record_from_structure(cell)
    )
    assert stored == value
    domain = _with(_rocksalt(), ("I6", "OC6"), SOURCE)
    assert bond_properties(domain)["_httk_bonds"]["shapes"] == {"sites": [0, 1], "symbols": ["I", "OC6"]}


@pytest.mark.parametrize("name", SERVED)
def test_served_values_never_contain_pair_rows(name: str) -> None:
    value = bond_properties(SERVED[name]())["_httk_bonds"]
    assert value["shapes"] is None or "pair_rows" not in value["shapes"]
    assert "pair_rows" not in json.dumps(value)


def test_a_structure_without_symmetry_serves_operation_0() -> None:
    cell = SERVED["a unit cell"]()
    row = _row(cell)
    assert row["space_group_symmetry_operations_xyz"] is None
    assert row["_httk_bonds"]["pairs"] and all(pair[2] == 0 for pair in row["_httk_bonds"]["pairs"])
    assert row["_httk_bonds"]["shapes"] == {"sites": list(range(8)), "symbols": ["OC6"] * 8}


def test_a_unit_cell_with_stated_symmetry_names_its_identity() -> None:
    plain = SERVED["a unit cell"]()
    stated = ("-x,-y,-z", "x+1/2,y+1/2,z", "x,y,z", "-x+1/2,-y+1/2,-z")
    cell = type(plain)(
        plain.cell,
        plain.sites,
        plain.species,
        plain.species_at_sites,
        symmetry=StructureSymmetry(space_group_symmetry_operations_xyz=stated),
        bonds=plain.bonds,
    )
    row = _row(cell)
    assert row["space_group_symmetry_operations_xyz"] == list(stated)
    value = row["_httk_bonds"]
    assert all(pair[2] == 2 for pair in value["pairs"])
    assert [pair[3:] for pair in value["pairs"]] == [
        pair[3:] for pair in bond_properties(plain)["_httk_bonds"]["pairs"]
    ]
    _validate(value)
    _check_bonds_value(value, row["nsites"], _operations(row), cell=True)
    record = _unitcell_record_from_structure(cell)
    assert stored_property_projections(UnitcellStructureRecord)["_httk_bonds"].response(record) == value


def test_a_bond_operation_missing_from_the_served_list_raises_and_serves_null(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    structure = _monoclinic()
    positions = structure._representative_sites().reduced_coords.to_fractions()
    listed = [AffineOperation(xyz) for xyz in structure.space_group_symmetry_operations_xyz]
    assert len(_bond_codes(structure, positions, listed)) == len(structure.bonds.pairs)
    with pytest.raises(ValueError, match="not a served symmetry operation"):
        _bond_codes(structure, positions, listed[:1])
    with pytest.raises(ValueError, match="not a served symmetry operation"):
        _bond_codes(structure, positions, None)
    monkeypatch.setattr(ASUStructure, "space_group_symmetry_operations_xyz", property(lambda self: ("x,y,z",)))
    with caplog.at_level(logging.WARNING, logger="httk.atomistic.entries.bonds"):
        assert _row(structure)["_httk_bonds"] is None
    assert "is not a served symmetry operation" in caplog.text


def test_rows_without_a_servable_network_are_null(caplog: pytest.LogCaptureFixture) -> None:
    from httk.core.storage import project_storage_record

    from httk.atomistic.storage.cif_record import CIFStructureRecord

    cell = UnitcellStructureView(_rocksalt()).unview()
    assert _row(None)["_httk_bonds"] is None
    assert _row(cell._with_bonds(None))["_httk_bonds"] is None
    assert _row(NumericUnitcellStructure(cell))["_httk_bonds"] is None
    numeric = NumericUnitcellStructureRecord.from_structure(NumericUnitcellStructure(cell))
    assert stored_property_projections(NumericUnitcellStructureRecord)["_httk_bonds"].response(numeric) is None
    block = CIFStructure.from_cif(FIXTURES / "cif_bonds" / "ordered.cif")
    cif = CIFStructureRecord(**project_storage_record(CIFStructureRecord, block))  # type: ignore[arg-type]
    assert StructureEntryProvider({"cif": cif}).records("structures")[0]["_httk_bonds"] is None  # type: ignore[index]
    # a domain whose served position is no atom of its site (its stated precision removed) is null, with a warning
    inexact = SERVED["inexact representatives (cif_bonds/snapped_translated)"]()
    broken = copy.copy(inexact)
    object.__setattr__(broken, "_coordinate_precision", None)
    with caplog.at_level(logging.WARNING, logger="httk.atomistic.entries.bonds"):
        row = next(iter(StructureEntryProvider({"broken": broken}).records("structures")))
    assert row["_httk_bonds"] is None and row["nsites"] == inexact.nsites
    (warning,) = [record for record in caplog.records if record.name == "httk.atomistic.entries.bonds"]
    assert "broken" in warning.getMessage() and getattr(warning, "context", None) == "serve"


def test_a_failing_stored_domain_is_null_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(*_args: object) -> list[str]:
        raise ValueError("deliberate")

    monkeypatch.setattr(bonds_module, "_bond_codes", fail)
    record = _domain_structure_record_from_value(_rocksalt())
    with caplog.at_level(logging.WARNING, logger="httk.atomistic.entries.bonds"):
        assert stored_property_projections(ASUStructureRecord)["_httk_bonds"].response(record) is None
    assert any("deliberate" in item.getMessage() for item in caplog.records)


# ---------------------------------------------------------------------------------------------
# served symmetry operations (M1)
# ---------------------------------------------------------------------------------------------
def _generated(positions: list[list[float]], operations: list[str]) -> set[tuple[float, ...]]:
    points = set()
    for position in positions:
        exact = [F(x).limit_denominator(1000) for x in position]
        for operation in operations:
            points.add(tuple(x % 1 for x in AffineOperation(operation).apply(exact).to_fractions()))
    return points


@pytest.mark.parametrize("kind", [ASUStructure, FundamentalDomainStructure])
def test_served_operations_generate_the_whole_cell_of_a_supercell_transform(kind: type[Any]) -> None:
    asu = _supercell_rocksalt()
    structure = kind(asu.cell, asu.spacegroup, asu.wyckoff_sites, asu.species, asu.transform)
    assert structure.setting() is None and len(structure.transform.lattice_cosets()) == 2
    expanded = {tuple(x % 1 for x in row) for row in structure.expand_sites().reduced_coords.to_fractions()}
    assert len(expanded) == 16
    operations = list(structure.space_group_symmetry_operations_xyz)
    assert len(operations) == 2 * len(structure.spacegroup.symmetry_operations) == len(set(operations))
    assert _generated(structure.fractional_site_positions, operations) == expanded
    record_type = ASUStructureRecord if kind is ASUStructure else FundamentalDomainStructureRecord
    record = _domain_structure_record_from_value(structure)
    projection = stored_property_projections(record_type)["space_group_symmetry_operations_xyz"]
    assert projection.response(record) == operations


def test_served_operations_are_deduplicated_modulo_the_own_lattice() -> None:
    # F m -3 m in its primitive cell: the 192 conjugated operations are 48 modulo the smaller lattice
    structure = _transformed_rocksalt(SettingTransform([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], (0, 0, 0)))
    assert structure.setting() is None and len(structure.transform.lattice_cosets()) == 1
    operations = structure.space_group_symmetry_operations_xyz
    assert len(operations) == len(set(operations)) == 48
    record = _domain_structure_record_from_value(structure)
    presented = RecordStructure(record)
    assert presented.symmetry is not None
    assert presented.symmetry.space_group_symmetry_operations_xyz == operations
    assert presented.nsites == 2 and presented.bonds is not None


def test_tabulated_settings_serve_their_tabulated_operations() -> None:
    structure = _monoclinic_setting()
    assert structure.setting() is not None
    assert structure.space_group_symmetry_operations_xyz == tuple(
        operation.wrapped().to_xyz() for operation in structure.setting().symmetry_operations
    )


# ---------------------------------------------------------------------------------------------
# a client rebuilds every bond from the served floats
# ---------------------------------------------------------------------------------------------
def _float_map(xyz: str) -> tuple[list[list[float]], list[float]]:
    operation = AffineOperation(xyz)
    return operation.matrix.to_floats(), operation.vector.to_floats()


def _apply(op: tuple[list[list[float]], list[float]], point: list[float]) -> list[float]:
    W, w = op
    return [sum(W[r][c] * point[c] for c in range(3)) + w[r] for r in range(3)]


def _cartesian(basis: list[list[float]], f: list[float]) -> list[float]:
    return [sum(f[k] * basis[k][j] for k in range(3)) for j in range(3)]


def _wrapped_close(p: list[float], q: list[float]) -> bool:
    return all(min(abs(x - y) % 1, 1 - abs(x - y) % 1) < 0.02 for x, y in zip(p, q, strict=True))


def _inexact_sites(structure: Any) -> set[int]:
    """The sites whose served position is not an exact atom of their site (stated to limited precision)."""
    if not isinstance(structure, FundamentalDomainStructure):
        return set()
    served = structure._representative_sites().reduced_coords.to_fractions()
    return {
        number
        for number, (site, point) in enumerate(zip(structure.wyckoff_sites, served, strict=True))
        if point not in [candidate.to_fractions() for candidate in structure._representatives_for_site(site)]
    }


def _client_bonds(row: dict[str, Any], inexact: set[int]) -> list[tuple[list[float], float]]:
    """Expand the served rows with the served operations and lattice: one Cartesian vector per bond of the cell.

    Each row ``[site, partner, op, t1, t2, t3]`` bonds the served position of ``site`` to
    ``listed[op] + t`` applied to the served position of ``partner``, ``listed`` being the served
    ``space_group_symmetry_operations_xyz`` (``["x,y,z"]`` when it is null).

    :param row: The served row.
    :param inexact: The sites whose served position is not exact; their bonds get the error bound of the
        served precision, every other bond 1e-6 Angstrom.
    :return: Per bond, its vector and its tolerance.
    """
    basis, positions, value = row["lattice_vectors"], row["fractional_site_positions"], row["_httk_bonds"]
    symmetric = row["site_coordinate_span"] not in ("unit_cell", "molecular_unit_cell")
    operations = [_float_map(xyz) for xyz in (row["space_group_symmetry_operations_xyz"] if symmetric else ["x,y,z"])]
    precision = row["_httk_coordinate_precision"] or 0.0
    bonds: list[tuple[list[float], list[float], float]] = []
    listed = [_float_map(xyz) for xyz in (row["space_group_symmetry_operations_xyz"] or ["x,y,z"])]
    for site, partner, op, *t in value["pairs"]:
        S = (listed[op][0], [x + n for x, n in zip(listed[op][1], t, strict=True)])
        a, b = positions[site], _apply(S, positions[partner])
        tolerance = 1e-6
        if {site, partner} & inexact:
            columns = [_cartesian(basis, [S[0][m][k] for m in range(3)]) for k in range(3)]
            lengths = sum(math.hypot(*column) for column in columns) + sum(math.hypot(*row) for row in basis)
            tolerance = max(tolerance, 3 * precision * lengths)
        for g in operations:
            ga, gb = _apply(g, a), _apply(g, b)
            vector = [y - x for x, y in zip(ga, gb, strict=True)]
            atom = [x % 1 for x in ga]
            other = [(x + d) % 1 for x, d in zip(atom, vector, strict=True)]
            known = any(
                (_wrapped_close(atom, seen) and all(abs(x - y) < 0.02 for x, y in zip(vector, d, strict=True)))
                or (_wrapped_close(other, seen) and all(abs(x + y) < 0.02 for x, y in zip(vector, d, strict=True)))
                for seen, d, _tolerance in bonds
            )
            if not known:
                bonds.append((atom, vector, tolerance))
    return [(_cartesian(basis, vector), tolerance) for _, vector, tolerance in bonds]


def _exact_bonds(structure: Any) -> list[list[float]]:
    """One exact Cartesian vector per bond of the unit cell, from the exact unit-cell network."""
    cell = UnitcellStructureView(structure).unview()
    basis = cell.cell.basis.to_floats()
    rows = cell.sites.reduced_coords.to_fractions()
    vectors = []
    for bond in cell.bonds.pairs:
        n = bond.operation.vector.to_fractions()
        vectors.append(_cartesian(basis, [float(rows[bond.partner][k] + n[k] - rows[bond.site][k]) for k in range(3)]))
    return vectors


@pytest.mark.parametrize("name", SERVED)
def test_a_client_rebuilds_every_bond_from_the_served_floats(name: str) -> None:
    structure = SERVED[name]()
    inexact = _inexact_sites(structure)
    assert bool(inexact) is name.startswith("inexact")
    got = _client_bonds(_row(structure), inexact)
    want = _exact_bonds(structure)
    assert len(got) == len(want)
    for vector in want:
        errors = [min(math.dist(item, vector), math.dist([-x for x in item], vector)) for item, _ in got]
        best = min(range(len(got)), key=errors.__getitem__)
        assert errors[best] <= got[best][1], (name, errors[best], got[best][1])
        got.pop(best)


def _conjugated(structure: FundamentalDomainStructure) -> list[str]:
    """Each bond's operation merely written in the own setting, ``T o b o T^-1``, with no per-site correction."""
    assert structure.bonds is not None
    t = _op(structure.transform.operation)
    return [_affine(_mul(_mul(t, _op(bond.operation)), _inv(t))).to_xyz() for bond in structure.bonds.pairs]


@pytest.mark.parametrize(
    "name", ["a site served as another orbit member", "a supercell transform", "an untabulated sheared transform"]
)
def test_the_served_operations_are_corrected_per_site(name: str) -> None:
    structure = SERVED[name]()
    row = _row(structure)
    served = [_row_operation(row, pair).to_xyz() for pair in row["_httk_bonds"]["pairs"]]
    assert served != _conjugated(structure)
    if name != "a site served as another orbit member":
        assert len(structure.wyckoff_sites) == 2 and structure.wyckoff_sites[0].free_count == 3


# ---------------------------------------------------------------------------------------------
# serving through httk-serve, and reading back
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def client() -> Any:
    pytest.importorskip("httk.serve.optimade")
    pytest.importorskip("httk.store")
    pytest.importorskip("starlette")
    from httk.serve.optimade import adapter_from_stores, create_asgi_app
    from httk.store import Backend, EntryIdScheme, SqlStore
    from httk.store.backend.sql import StoredEntrySource
    from starlette.testclient import TestClient

    with Backend.sqlite() as database:
        store = SqlStore(database, entry_ids=EntryIdScheme("httk.test", "1"), entry_records=RECORDS)
        store.save(SERVED["a unit cell"]())
        store.save(_rocksalt())
        store.save(UnitcellStructureView(_rocksalt(F(560, 100))).unview()._with_bonds(None))
        adapter = adapter_from_stores([StoredEntrySource(store, StructureEntry, "bonds")])
        assert "_httk_bonds" not in adapter.schema.default_response_fields["structures"]
        with TestClient(create_asgi_app(adapter, baseurl="http://testserver"), base_url="http://testserver") as http:
            listed = http.get("/v1/structures", params={"response_fields": "site_coordinate_span,_httk_bonds"}).json()
            names = {
                "domain"
                if item["attributes"]["site_coordinate_span"] == "asymmetric_unit"
                else "plain"
                if item["attributes"]["_httk_bonds"] is None
                else "cell": item["id"]
                for item in listed["data"]
            }
            assert len(names) == 3
            yield http, names


def _resource(http: Any, entry_id: str) -> OptimadeResource:
    fields = (
        "lattice_vectors,fractional_site_positions,species,species_at_sites,site_coordinate_span,"
        "dimension_types,space_group_symmetry_operations_xyz,_httk_bonds"
    )
    info = http.get("/v1/info/structures")
    entry = http.get("/v1/structures", params={"response_fields": fields, "filter": f'id="{entry_id}"'})
    assert info.status_code == entry.status_code == 200
    schema = OptimadeSchemaSnapshot(
        "structures", OptimadeDocument.from_response(info.text, "http://testserver/v1/info")
    )
    return OptimadeResource(OptimadeDocument.from_response(entry.text, "http://testserver/v1/structures"), 0, schema)


def test_the_default_response_omits_bonds_and_response_fields_returns_them(client: Any) -> None:
    http, names = client
    default = http.get("/v1/structures").json()
    assert default["data"] and all("_httk_bonds" not in item["attributes"] for item in default["data"])
    asked = http.get("/v1/structures", params={"response_fields": "_httk_bonds"}).json()
    served = {item["id"]: item["attributes"]["_httk_bonds"] for item in asked["data"]}
    assert served[names["plain"]] is None
    for key in ("cell", "domain"):
        _validate(served[names[key]])
        assert served[names[key]]["pairs"]
    info = http.get("/v1/info/structures").json()
    assert info["data"]["properties"]["_httk_bonds"]["$id"] == BONDS_ID


def test_a_served_unit_cell_reads_back_with_its_network_and_sources(client: Any) -> None:
    http, names = client
    expected = SERVED["a unit cell"]().bonds
    read = OptimadeStructure(_resource(http, names["cell"]))
    assert read.bonds == expected
    assert (read.bonds.source, read.bonds.shapes_source) == (SOURCE, SHAPES_SOURCE)
    assert UnitcellStructureView(read).bonds == expected
    assert OptimadeStructure(_resource(http, names["domain"])).bonds is None  # read-back of a domain span is deferred
    assert OptimadeStructure(_resource(http, names["plain"])).bonds is None


SERVED_OPS = ("-x,-y,-z", "x,y,z")  # the identity is index 1
_EYE = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _bonds_resource(
    value: object,
    *,
    span: str = "unit_cell",
    name: str = "_httk_bonds",
    operations: tuple[str, ...] | None = SERVED_OPS,
) -> OptimadeResource:
    attributes = {
        "lattice_vectors": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        "fractional_site_positions": [[0, 0, 0], [0.5, 0.5, 0.5]],
        "species": [
            {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1]},
            {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1]},
        ],
        "species_at_sites": ["Na", "Cl"],
        "dimension_types": [1, 1, 1],
        "site_coordinate_span": span,
        "space_group_symmetry_operations_xyz": None if operations is None else list(operations),
        name: value,
    }
    properties: dict[str, object] = {key: {} for key in attributes if key != name}
    if name != "_httk_bonds":
        properties[name] = {"$id": BONDS_ID}
    info = OptimadeDocument.from_response(
        json.dumps({"meta": {"api_version": "1.3.0"}, "data": {"properties": properties}}), "https://example.test/info"
    )
    document = OptimadeDocument.from_response(
        json.dumps({"data": [{"id": "x", "type": "structures", "attributes": attributes}]}),
        "https://example.test/v1/structures",
    )
    return OptimadeResource(document, 0, OptimadeSchemaSnapshot("structures", info))


def _pairs(*pairs: tuple[int, ...]) -> dict[str, Any]:
    return {"pairs": [list(pair) for pair in pairs]}


# Na bonded to Cl at three translations (identity at index 1 of the served list), then a fourth
FAN = [(0, 1, 1, 0, 0, 0), (0, 1, 1, -1, 0, 0), (0, 1, 1, 0, -1, 0), (0, 1, 1, 0, 0, -1)]


def test_read_back_resolves_the_definition_id_and_validates() -> None:
    value = {**_pairs((1, 0, 1, 0, 0, 0), (0, 1, 1, -1, 0, 0)), "shapes": None, "source": {"software": "x"}}
    expected = BondNetwork(
        [SiteBond(0, 1, AffineOperation("x,y,z")), SiteBond(0, 1, AffineOperation("x-1,y,z"))], source={"software": "x"}
    )
    for name in ("_httk_bonds", "_bonds_elsewhere"):
        read = OptimadeStructure(_bonds_resource(value, name=name)).bonds
        assert read == expected and read.pairs == expected.pairs and read.source == expected.source
    assert OptimadeStructure(_bonds_resource(None)).bonds is None
    assert OptimadeStructure(_bonds_resource(value, span="fundamental_domain")).bonds is None
    # without served operations, every row names operation 0, the identity
    plain = {**_pairs((1, 0, 0, 0, 0, 0), (0, 1, 0, -1, 0, 0)), "source": {"software": "x"}}
    assert OptimadeStructure(_bonds_resource(plain, operations=None)).bonds == expected


def test_read_back_turns_i_into_its_count() -> None:
    read = OptimadeStructure(_bonds_resource({**_pairs(*FAN), "shapes": {"sites": [0, 1], "symbols": ["I", "I4"]}}))
    assert read.bonds is not None and read.bonds.shapes == ("I4", "I4")
    for wrong, message in (
        ({"sites": [0], "symbols": ["I5"]}, "is for 5 bonds"),
        ({"sites": [0], "symbols": ["I"], "pair_rows": [[0, 1]]}, "covers 2 bonded atoms"),
    ):
        with pytest.raises(IncompleteOptimadeResourceError, match=message):
            OptimadeStructure(_bonds_resource({**_pairs(*FAN), "shapes": wrong})).bonds  # noqa: B018


def test_read_back_decodes_the_full_shape_form() -> None:
    shapes = {"sites": [0, 1], "symbols": ["T4", "T4"]}
    expected = BondNetwork([SiteBond(i, j, AffineOperation(_EYE, t)) for i, j, _op, *t in FAN], ("T4", "T4"))
    for form in (shapes, {**shapes, "pair_rows": None}, {**shapes, "pair_rows": [None, [3, 2, 1, 0]]}):
        value = {**_pairs(*FAN), "shapes": form, "shapes_source": {"software": "s"}}
        read = OptimadeStructure(_bonds_resource(value)).bonds
        assert read == expected and read.shapes_source is not None and read.shapes_source.software == "s"
    partial = {**_pairs(*FAN), "shapes": {"sites": [0], "symbols": ["T4"]}}
    assert OptimadeStructure(_bonds_resource(partial)).bonds.shapes == ("T4", None)


@pytest.mark.parametrize(
    "shapes",
    [
        {"sites": [0], "symbols": ["L2"], "pair_rows": [[0, 1]]},  # a valid shape over some of the site's bonds
        {"sites": [0, 0], "symbols": ["L2", "L2"], "pair_rows": [[0, 1], [2, 3]]},  # two shapes on one site
    ],
)
def test_read_back_drops_shapes_it_cannot_represent(shapes: dict[str, Any], caplog: pytest.LogCaptureFixture) -> None:
    value = {**_pairs(*FAN), "shapes": shapes, "shapes_source": {"software": "s"}}
    with caplog.at_level(logging.WARNING, logger="httk.atomistic.models.structure.optimade"):
        read = OptimadeStructure(_bonds_resource(value)).bonds
    assert read == BondNetwork([SiteBond(i, j, AffineOperation(_EYE, t)) for i, j, _op, *t in FAN])
    assert read.shapes is None and read.shapes_source is None
    assert "the bonds are read without shapes" in caplog.text


MALFORMED = {
    "a non-identity operation": _pairs((0, 1, 0, 0, 0, 0)),
    "an operation out of range": _pairs((0, 1, 2, 0, 0, 0)),
    "a nonzero operation without served operations": (None, _pairs((0, 1, 1, 0, 0, 0))),
    "a fractional translation": _pairs((0, 1, 1, 0.5, 0, 0)),  # type: ignore[arg-type]
    "a short row": _pairs((0, 1, 1, 0, 0)),
    "an operation string": {"pairs": [{"site": 0, "partner": 1, "operation_xyz": "x,y,z"}]},
    "an index out of range": _pairs((0, 2, 1, 0, 0, 0)),
    "an unknown key": {**_pairs((0, 1, 1, 0, 0, 0)), "order": 1},
    "a bad symbol": {**_pairs(*FAN), "shapes": {"sites": [0], "symbols": ["XX4"]}},
    "a symbol with the wrong count": {**_pairs(*FAN), "shapes": {"sites": [0], "symbols": ["OC6"]}},
    "a per-site shape list": {**_pairs(*FAN), "shapes": ["T4", "T4"]},
    "unaligned shape lists": {**_pairs(*FAN), "shapes": {"sites": [0, 1], "symbols": ["T4"]}},
    "a null symbol": {**_pairs(*FAN), "shapes": {"sites": [0], "symbols": [None]}},
    "a shape on a site out of range": {**_pairs(*FAN), "shapes": {"sites": [2], "symbols": ["T4"]}},
    "an empty shape list": {**_pairs(*FAN), "shapes": {"sites": [], "symbols": []}},
    "an unknown shapes key": {**_pairs(*FAN), "shapes": {"sites": [0], "symbols": ["T4"], "order": [1]}},
    "a pair_rows row not touching its site": {
        **_pairs(*FAN, (1, 1, 1, 1, 0, 0)),
        "shapes": {"sites": [0], "symbols": ["T4"], "pair_rows": [[0, 1, 2, 4]]},
    },
    "a symbol whose number differs from the atoms covered": {
        **_pairs(*FAN),
        "shapes": {"sites": [0], "symbols": ["TP3"], "pair_rows": [[0, 1]]},
    },
    "a repeated site with overlapping rows": {
        **_pairs(*FAN),
        "shapes": {"sites": [0, 0], "symbols": ["A2", "TP3"], "pair_rows": [[0, 1], [1, 2, 3]]},
    },
    "missing pairs": {"shapes": None},
    "an unknown source key": {**_pairs((0, 1, 1, 0, 0, 0)), "source": {"author": "x"}},
    "a self bond": _pairs((0, 0, 1, 0, 0, 0)),
}


@pytest.mark.parametrize("label", MALFORMED)
def test_a_malformed_value_raises(label: str) -> None:
    case = MALFORMED[label]
    operations, value = case if isinstance(case, tuple) else (SERVED_OPS, case)
    with pytest.raises(IncompleteOptimadeResourceError, match="_httk_bonds"):
        OptimadeStructure(_bonds_resource(value, operations=operations)).bonds  # noqa: B018


IDENTITY_LISTED = [AffineOperation("-x,-y,-z"), AffineOperation("x,y,z")]

CHECKED = {
    "a row of five": ({"pairs": [[0, 1, 1, 0, 0]]}, "six integers"),
    "a site out of range": ({"pairs": [[0, 2, 1, 0, 0, 0]]}, "outside 0..1"),
    "an operation out of range": ({"pairs": [[0, 1, 2, 0, 0, 0]]}, "operation 2, outside 0..1"),
    "a non-identity operation in a unit cell": ({"pairs": [[0, 1, 0, 0, 0, 0]]}, "not the identity"),
    "unaligned lists": (
        {"pairs": [list(row) for row in FAN], "shapes": {"sites": [0], "symbols": ["T4"], "pair_rows": []}},
        "same length",
    ),
    "a pair_rows row not naming its site": (
        {"pairs": [list(row) for row in FAN], "shapes": {"sites": [1], "symbols": ["A2"], "pair_rows": [[0, 9]]}},
        r"rows \[9\] that do not name the site",
    ),
    "a repeated site with overlapping rows": (
        {
            "pairs": [list(row) for row in FAN],
            "shapes": {"sites": [0, 0], "symbols": ["A2", "A2"], "pair_rows": [[0, 1], [1, 2]]},
        },
        "not disjoint",
    ),
    "a repeated site with a null entry": (
        {"pairs": [list(row) for row in FAN], "shapes": {"sites": [0, 0], "symbols": ["T4", "A2"], "pair_rows": None}},
        "not disjoint",
    ),
    "a symbol outside the vocabulary": (
        {"pairs": [list(row) for row in FAN], "shapes": {"sites": [0], "symbols": ["I2"]}},
        "unknown symbol 'I2'",
    ),
    "a count differing from the atoms covered": (
        {"pairs": [list(row) for row in FAN], "shapes": {"sites": [1], "symbols": ["TP3"]}},
        "covers 4 bonded atoms",
    ),
}


@pytest.mark.parametrize("label", CHECKED)
def test_the_checker_enforces_each_rule(label: str) -> None:
    value, message = CHECKED[label]
    with pytest.raises(ValueError, match=message):
        _check_bonds_value(value, 2, IDENTITY_LISTED, cell=True)


def test_the_checker_counts_atoms_only_for_unit_cells_and_accepts_valid_values() -> None:
    fan = [list(row) for row in FAN]
    self_bond = {"pairs": [[0, 0, 1, 1, 0, 0]], "shapes": {"sites": [0], "symbols": ["L2"]}}
    _check_bonds_value(self_bond, 1, IDENTITY_LISTED, cell=True)  # (i, i, t) brings in two atoms at i
    with pytest.raises(ValueError, match="covers 2 bonded atoms"):
        _check_bonds_value({**self_bond, "shapes": {"sites": [0], "symbols": ["OC6"]}}, 1, IDENTITY_LISTED, cell=True)
    # a symmetric span: rows may name any listed operation, and the count is not checked
    _check_bonds_value(
        {"pairs": [[0, 1, 0, 0, 0, 0]], "shapes": {"sites": [0], "symbols": ["OC6"]}}, 2, IDENTITY_LISTED
    )
    with pytest.raises(ValueError, match="operation 1, outside 0..0"):
        _check_bonds_value({"pairs": fan}, 2, None)
    zeroed = [[i, j, 0, *t] for i, j, _op, *t in fan]
    _check_bonds_value(
        {"pairs": zeroed, "shapes": {"sites": [0, 0], "symbols": ["A2", "A2"], "pair_rows": [[0, 1], [2, 3]]}},
        2,
        None,
        cell=True,
    )
    _check_bonds_value(ILLUSTRATION, 3, None)
