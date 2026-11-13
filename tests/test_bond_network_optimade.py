"""Serving bond networks as ``_httk_bonds`` and reading them back.

A client must be able to rebuild every bond of a crystal from the served floats alone: the served
positions, lattice and symmetry operations, and the per-site corrected bond operations.
"""

import copy
import importlib.resources
import json
import logging
import math
import random
import re
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
from httk.atomistic.entries.bonds import BOND_PROPERTY_KEYS, bond_definitions, bond_properties
from httk.atomistic.entries.structures import StructureEntry
from httk.atomistic.models.structure.asu import _affine, _inv, _mul, _op
from httk.atomistic.models.structure.network import SiteBond
from httk.atomistic.models.structure.record import RecordStructure
from httk.atomistic.storage.records import (
    ASUStructureRecord,
    FundamentalDomainStructureRecord,
    NumericUnitcellStructureRecord,
    UnitcellStructureRecord,
    _domain_structure_record_from_value,
    _unitcell_record_from_structure,
)
from httk.atomistic.symmetry.affine_operation import AffineOperation

FIXTURES = Path(__file__).parent / "fixtures"
BONDS_ID = "https://schemas.httk.org/defs/v0.1/properties/chemistry/bonds"
SI, O = Species("Si", ("Si",), (1,)), Species("O", ("O",), (1,))


@cache
def _schema() -> dict[str, Any]:
    text = importlib.resources.files("httk.registry.schemas.atomistic").joinpath("bonds.json").read_text()
    return json.loads(text)


def _pattern() -> re.Pattern[str]:
    operation = _schema()["properties"]["pairs"]["items"]["properties"]["operation_xyz"]
    return re.compile(operation["x-undef-pattern"])


def _validate(value: Any) -> None:
    """Validate a served value against ``bonds.json``, its operation pattern included."""
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator(_schema()).validate(value)
    if value is not None:
        assert all(_pattern().fullmatch(pair["operation_xyz"]) for pair in value["pairs"])


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


def test_the_pattern_accepts_sheared_operations() -> None:
    structure = _sheared_monoclinic()
    assert structure.setting() is None
    value = bond_properties(structure)["_httk_bonds"]
    composed = [
        (AffineOperation(g) * AffineOperation(pair["operation_xyz"])).to_xyz()
        for g in structure.space_group_symmetry_operations_xyz
        for pair in value["pairs"]
    ]
    assert any(re.search(r"[xyz][-+]([0-9]+\*)?[xyz]", xyz) for xyz in composed)  # sheared terms such as x-2*y
    assert all(_pattern().fullmatch(xyz) for xyz in composed)
    examples = _schema()["properties"]["pairs"]["items"]["properties"]["operation_xyz"]["examples"]
    assert all(_pattern().fullmatch(xyz) for xyz in examples)
    for wrong in ("x,y", "x+1/0,y,z", "x,y,z+1/01", "x,,z", "x y,y,z"):
        assert not _pattern().fullmatch(wrong)


def test_the_pattern_accepts_every_to_xyz_spelling() -> None:
    rng = random.Random(7)

    def rational() -> F:
        return F(rng.randint(-12, 12), rng.choice((1, 1, 1, 2, 3, 4, 6, 7)))

    for _ in range(30_000):
        matrix = [[rational() if rng.random() < 0.6 else 0 for _ in range(3)] for _ in range(3)]
        xyz = AffineOperation(matrix, [rational() for _ in range(3)]).to_xyz()
        assert _pattern().fullmatch(xyz), xyz


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

    monkeypatch.setattr(bonds_module, "_served_operations", fail)
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
    """Expand the served pairs with the served operations and lattice: one Cartesian vector per bond of the cell.

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
    for pair in value["pairs"]:
        S = _float_map(pair["operation_xyz"])
        a, b = positions[pair["site"]], _apply(S, positions[pair["partner"]])
        tolerance = 1e-6
        if {pair["site"], pair["partner"]} & inexact:
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
    served = [pair["operation_xyz"] for pair in bond_properties(structure)["_httk_bonds"]["pairs"]]
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


def _bonds_resource(value: object, *, span: str = "unit_cell", name: str = "_httk_bonds") -> OptimadeResource:
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
        "space_group_symmetry_operations_xyz": ["x,y,z"],
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


def _pairs(*pairs: tuple[int, int, str]) -> dict[str, Any]:
    return {"pairs": [{"site": i, "partner": j, "operation_xyz": xyz} for i, j, xyz in pairs]}


def test_read_back_resolves_the_definition_id_and_validates() -> None:
    value = {**_pairs((1, 0, "x,y,z"), (0, 1, "x-1,y,z")), "shapes": None, "source": {"software": "x"}}
    expected = BondNetwork(
        [SiteBond(0, 1, AffineOperation("x,y,z")), SiteBond(0, 1, AffineOperation("x-1,y,z"))], source={"software": "x"}
    )
    for name in ("_httk_bonds", "_bonds_elsewhere"):
        read = OptimadeStructure(_bonds_resource(value, name=name)).bonds
        assert read == expected and read.pairs == expected.pairs and read.source == expected.source
    assert OptimadeStructure(_bonds_resource(None)).bonds is None
    assert OptimadeStructure(_bonds_resource(value, span="fundamental_domain")).bonds is None


MALFORMED = {
    "a non-translation": _pairs((0, 1, "-x,y,z")),
    "a fractional translation": _pairs((0, 1, "x+1/2,y,z")),
    "an index out of range": _pairs((0, 2, "x,y,z")),
    "an unknown key": {**_pairs((0, 1, "x,y,z")), "order": 1},
    "an unknown pair key": {"pairs": [{"site": 0, "partner": 1, "operation_xyz": "x,y,z", "order": 1}]},
    "a bad symbol": {**_pairs((0, 1, "x,y,z")), "shapes": ["XX1", None]},
    "a symbol with the wrong count": {**_pairs((0, 1, "x,y,z")), "shapes": ["L2", None]},
    "missing pairs": {"shapes": None},
    "a non-string operation": _pairs((0, 1, 7)),  # type: ignore[arg-type]
    "an unknown source key": {**_pairs((0, 1, "x,y,z")), "source": {"author": "x"}},
    "a self bond": _pairs((0, 0, "x,y,z")),
    "a division by zero": _pairs((0, 1, "x+1/0,y,z")),
}


@pytest.mark.parametrize("label", MALFORMED)
def test_a_malformed_value_raises(label: str) -> None:
    with pytest.raises(IncompleteOptimadeResourceError, match="_httk_bonds"):
        OptimadeStructure(_bonds_resource(MALFORMED[label])).bonds  # noqa: B018
