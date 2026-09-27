"""No symmetry stated is one canonical value: ``None``, never an empty object or invented P1."""

import json
import pathlib
from dataclasses import replace

import httk.core
from httk.core import load_entry_type_definition
from httk.core.optimade import OptimadeDocument, OptimadeResource, OptimadeSchemaSnapshot, served_entry
from httk.core.storage import content_id

from httk.atomistic import (
    OptimadeStructure,
    RecordStructure,
    StructureEntryProvider,
    StructureSymmetry,
    UnitcellStructure,
    UnitcellStructureView,
)
from httk.atomistic.entries.precision import precision_definitions
from httk.atomistic.storage.records import SymmetryRecord, _unitcell_record_from_structure

_STRUCTURES_ID = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"
_SILICON_POSCAR = """Si
1.0
0.0 2.5 2.5
2.5 0.0 2.5
2.5 2.5 0.0
Si
2
Direct
0 0 0
0.25 0.25 0.25
"""
_SILICON_ATTRIBUTES: dict[str, object] = {
    "lattice_vectors": [[0.0, 2.5, 2.5], [2.5, 0.0, 2.5], [2.5, 2.5, 0.0]],
    "fractional_site_positions": [[0, 0, 0], [0.25, 0.25, 0.25]],
    "species": [{"name": "Si", "chemical_symbols": ["Si"], "concentration": [1.0]}],
    "species_at_sites": ["Si", "Si"],
    "dimension_types": [1, 1, 1],
    "nperiodic_dimensions": 3,
}
# 2026-09-27: the POSCAR-derived pin; an OPTIMADE twin stating no symmetry must equal it.
_SILICON_CONTENT_ID = "c19e62e8e7eaa3dffcd7b06004863fc5f290f96e707ddd33676fa7c5de0cea7f"


def _definition_ids() -> dict[str, str]:
    schema = load_entry_type_definition(_STRUCTURES_ID)
    ids = {name: value.definition_id for name, value in schema.properties.items() if name != "id"}
    ids.update({name: value.definition_id for name, value in precision_definitions().items()})
    return ids


def _resource(attributes: dict[str, object]) -> OptimadeResource:
    ids = _definition_ids()
    properties = {name: {"$id": ids[name]} for name in attributes}
    info = OptimadeDocument.from_response(
        json.dumps({"data": {"properties": properties}}), "https://example.test/info/structures"
    )
    document = OptimadeDocument.from_response(
        json.dumps({"data": [{"id": "si", "type": "structures", "attributes": attributes}]}),
        "https://example.test/v1/structures",
    )
    return OptimadeResource(document, 0, OptimadeSchemaSnapshot("structures", info))


def _poscar_silicon(tmp_path: pathlib.Path) -> UnitcellStructure:
    path = tmp_path / "POSCAR"
    path.write_text(_SILICON_POSCAR)
    structure = httk.core.load(str(path))
    assert isinstance(structure, UnitcellStructure)
    return structure


def _silicon(symmetry: StructureSymmetry | None) -> UnitcellStructure:
    return UnitcellStructure(
        [[0, "5/2", "5/2"], ["5/2", 0, "5/2"], ["5/2", "5/2", 0]],
        [[0, 0, 0], ["1/4", "1/4", "1/4"]],
        species_at_sites=["Si", "Si"],
        symmetry=symmetry,
    )


def _served(structure: object) -> dict:
    return next(iter(StructureEntryProvider({"si": structure}).records("structures")))


def test_optimade_structure_stating_nothing_has_no_symmetry_and_matches_its_poscar_twin(
    tmp_path: pathlib.Path,
) -> None:
    backend = OptimadeStructure(_resource(_SILICON_ATTRIBUTES))
    assert backend.symmetry is None
    view = UnitcellStructureView(backend)
    assert view.symmetry is None
    assert _unitcell_record_from_structure(view).symmetry is None

    poscar = _poscar_silicon(tmp_path)
    assert poscar.symmetry is None
    assert content_id(poscar) == _SILICON_CONTENT_ID
    assert content_id(view) == _SILICON_CONTENT_ID


def test_an_explicit_all_empty_symmetry_collapses_to_none() -> None:
    structure = _silicon(StructureSymmetry())
    assert structure.symmetry is None
    assert _unitcell_record_from_structure(structure).symmetry is None
    assert content_id(structure) == content_id(_silicon(None))


def test_partial_metadata_is_a_claim_and_keeps_its_record() -> None:
    structure = _silicon(StructureSymmetry(space_group_symbol_hall="-F 4vw 2vw 3"))
    record = _unitcell_record_from_structure(structure).symmetry
    assert record is not None and record.space_group_symbol_hall == "-F 4vw 2vw 3"
    assert content_id(structure) != content_id(_silicon(None))


def test_a_stated_p1_is_distinct_from_nothing_stated() -> None:
    stated = _silicon(StructureSymmetry(space_group_symmetry_operations_xyz=("x,y,z",)))
    record = _unitcell_record_from_structure(stated).symmetry
    assert record is not None and record.space_group_symmetry_operations_xyz == ("x,y,z",)
    assert content_id(stated) != content_id(_silicon(None))


def test_serving_invents_no_operations_and_keeps_stated_ones() -> None:
    assert _served(_silicon(None))["space_group_symmetry_operations_xyz"] is None
    remote = UnitcellStructureView(OptimadeStructure(_resource(_SILICON_ATTRIBUTES)))
    assert _served(remote)["space_group_symmetry_operations_xyz"] is None
    stated = _silicon(StructureSymmetry(space_group_symmetry_operations_xyz=("x,y,z",)))
    assert _served(stated)["space_group_symmetry_operations_xyz"] == ["x,y,z"]


def test_stored_serving_invents_no_operations() -> None:
    from httk.atomistic.storage.stored_properties import _unitcell_symmetry_value

    record = _unitcell_record_from_structure(_silicon(None))
    assert _unitcell_symmetry_value(record, "space_group_symmetry_operations_xyz") is None
    stated = _unitcell_record_from_structure(
        _silicon(StructureSymmetry(space_group_symmetry_operations_xyz=("x,y,z",)))
    )
    assert _unitcell_symmetry_value(stated, "space_group_symmetry_operations_xyz") == ["x,y,z"]


def test_serving_a_structure_stating_nothing_round_trips_its_content_id(tmp_path: pathlib.Path) -> None:
    source = _poscar_silicon(tmp_path)
    ids = _definition_ids()
    served = {name: value for name, value in _served(source).items() if name in ids}
    assert served["space_group_symmetry_operations_xyz"] is None
    backend = OptimadeStructure(_resource(served))
    assert backend.symmetry is None
    assert content_id(UnitcellStructureView(backend)) == content_id(source) == _SILICON_CONTENT_ID


def test_a_legacy_record_with_an_empty_symmetry_record_presents_no_symmetry() -> None:
    plain = _silicon(None)
    legacy = replace(_unitcell_record_from_structure(plain), symmetry=SymmetryRecord())
    backend = RecordStructure(legacy)
    assert backend.symmetry is None
    assert content_id(UnitcellStructureView(backend)) == content_id(plain)


def test_a_nonperiodic_optimade_structure_stating_nothing_has_no_symmetry() -> None:
    attributes = {
        "lattice_vectors": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        "fractional_site_positions": [[0, 0, 0]],
        "species": [{"name": "C", "chemical_symbols": ["C"], "concentration": [1.0]}],
        "species_at_sites": ["C"],
        "dimension_types": [0, 0, 0],
        "nperiodic_dimensions": 0,
    }
    assert OptimadeStructure(_resource(attributes)).symmetry is None


def test_served_entry_reads_served_silicon_back_to_its_poscar_twin(tmp_path: pathlib.Path) -> None:
    source = _poscar_silicon(tmp_path)
    ids = _definition_ids()
    attributes = {name: value for name, value in _served(source).items() if name in ids}
    entry = served_entry("structures", attributes)
    assert isinstance(entry, UnitcellStructureView)
    assert entry.symmetry is None
    assert content_id(entry) == content_id(source) == _SILICON_CONTENT_ID
