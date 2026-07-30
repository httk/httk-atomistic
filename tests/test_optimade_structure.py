import json
from fractions import Fraction

import pytest
from httk.core import (
    IncompleteOptimadeResourceError,
    OptimadeDocument,
    OptimadeResource,
    OptimadeSchemaSnapshot,
    load_entry_type_schema,
)

from httk.atomistic import OptimadeStructure, StructureBackend, UnitcellStructureView, precision_definitions

_STRUCTURES_ID = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"


def _definition_ids() -> dict[str, str]:
    schema = load_entry_type_schema(_STRUCTURES_ID)
    return {
        name: schema.properties[name].definition_id
        for name in (
            "lattice_vectors",
            "cartesian_site_positions",
            "fractional_site_positions",
            "species",
            "species_at_sites",
            "dimension_types",
        )
    }


def _resource(*, ids: dict[str, object] | None = None, attributes: dict[str, object] | None = None) -> OptimadeResource:
    definition_ids = _definition_ids()
    names = {
        "lattice_vectors": "remote_lattice",
        "cartesian_site_positions": "remote_cartesian",
        "species": "remote_species",
        "species_at_sites": "remote_site_species",
        "dimension_types": "remote_dimensions",
    }
    info_properties = {names[name]: {"$id": definition_ids[name]} for name in names}
    if ids is not None:
        info_properties = {name: {"$id": definition_id} for name, definition_id in ids.items()}
    values = {
        names["lattice_vectors"]: [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
        names["cartesian_site_positions"]: [[0, 0, 0], [1, 1.5, 2]],
        names["species"]: [
            {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1]},
            {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1]},
        ],
        names["species_at_sites"]: ["Na", "Cl"],
        names["dimension_types"]: [1, 1, 0],
    }
    if attributes is not None:
        values = attributes
    info = OptimadeDocument.create(json.dumps({"data": {"properties": info_properties}}), "https://example.test/info")
    document = OptimadeDocument.create(
        json.dumps({"data": [{"id": "example-1", "type": "structures", "attributes": values}]}),
        "https://example.test/v1/structures",
    )
    return OptimadeResource(document, 0, OptimadeSchemaSnapshot("structures", info))


def test_optimade_structure_uses_definition_ids_not_transport_labels() -> None:
    resource = _resource()
    backend = OptimadeStructure(resource)
    view = UnitcellStructureView(backend)

    assert backend.id == "example-1"
    assert backend.type == "structures"
    assert view.cell.periodicity == (True, True, False)
    assert view.cell.basis.to_fractions_approx() == [
        [Fraction(2), Fraction(0), Fraction(0)],
        [Fraction(0), Fraction(3), Fraction(0)],
        [Fraction(0), Fraction(0), Fraction(4)],
    ]
    assert view.sites.reduced_coords.to_fractions() == [
        [Fraction(0), Fraction(0), Fraction(0)],
        [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
    ]
    assert tuple(species.name for species in view.species) == ("Na", "Cl")
    assert view.species_at_sites == ("Na", "Cl")


@pytest.mark.parametrize("bad_id", [None, "", 42, "relative/property", " https://schemas.example/id"])
def test_missing_or_invalid_definition_id_is_not_recognized(bad_id: object) -> None:
    ids = {"remote_lattice": bad_id}
    backend = OptimadeStructure(_resource(ids=ids))

    with pytest.raises(IncompleteOptimadeResourceError, match="lattice_vectors"):
        _ = backend.cell


def test_partial_resource_and_unitcell_view_stay_lazy() -> None:
    definition_ids = _definition_ids()
    resource = _resource(
        ids={"remote_species": definition_ids["species"]},
        attributes={"remote_species": [{"name": "Na", "chemical_symbols": ["Na"], "concentration": [1]}]},
    )
    backend = OptimadeStructure(resource)
    view = UnitcellStructureView(backend)

    assert backend.unwrap() is resource
    assert view._backend is backend
    assert view.species[0].name == "Na"
    with pytest.raises(IncompleteOptimadeResourceError, match="lattice_vectors"):
        _ = view.cell
    with pytest.raises(IncompleteOptimadeResourceError, match="species_at_sites"):
        _ = view.species_at_sites
    with pytest.raises(IncompleteOptimadeResourceError, match="cartesian_site_positions"):
        _ = view.sites


def test_required_attribute_missing_raises_only_for_affected_component() -> None:
    resource = _resource(attributes={"remote_dimensions": [1, 1, 1]})
    backend = OptimadeStructure(resource)

    with pytest.raises(IncompleteOptimadeResourceError, match="lattice_vectors"):
        _ = backend.cell


def test_decimal_token_precision_falls_back_without_registered_precision_properties() -> None:
    schema = OptimadeSchemaSnapshot(
        "structures",
        OptimadeDocument.create(
            json.dumps(
                {
                    "data": {
                        "properties": {
                            "remote_lattice": {"$id": _definition_ids()["lattice_vectors"]},
                            "remote_cartesian": {"$id": _definition_ids()["cartesian_site_positions"]},
                            "remote_species": {"$id": _definition_ids()["species"]},
                            "remote_site_species": {"$id": _definition_ids()["species_at_sites"]},
                            "remote_dimensions": {"$id": _definition_ids()["dimension_types"]},
                        }
                    }
                }
            ),
            "https://example.test/info/structures",
        ),
    )
    document = OptimadeDocument.create(
        """{"data":[{"id":"precision","type":"structures","attributes":{
        "remote_lattice":[[2.000,0.000,0.000],[0.000,3.000,0.000],[0.000,0.000,4.000]],
        "remote_cartesian":[[1.500,0.000,0.000]],
        "remote_species":[{"name":"Na","chemical_symbols":["Na"],"concentration":[1]}],
        "remote_site_species":["Na"],"remote_dimensions":[1,1,1]}}]}""",
        "https://example.test/v1/structures",
    )
    view = UnitcellStructureView(OptimadeStructure(OptimadeResource(document, 0, schema)))

    assert view.basis_precision == Fraction(1, 1000)
    assert view.coordinate_precision == Fraction(1, 2000)


def test_renamed_fractional_positions_are_exact_and_preferred_over_cartesian() -> None:
    definition_ids = _definition_ids()
    resource = _resource(
        ids={
            "coordinates_under_an_unrelated_name": definition_ids["fractional_site_positions"],
            "also_cartesian": definition_ids["cartesian_site_positions"],
        },
        attributes={
            "coordinates_under_an_unrelated_name": [[0.25, 0.125, 0.0625]],
            "also_cartesian": [[99, 99, 99]],
        },
    )
    backend = OptimadeStructure(resource)

    assert backend.sites.reduced_coords.to_fractions() == [[Fraction(1, 4), Fraction(1, 8), Fraction(1, 16)]]


@pytest.mark.parametrize("fractional_value", [pytest.param("missing", id="absent"), pytest.param(None, id="null")])
def test_cartesian_positions_are_fallback_when_fractional_is_absent_or_null(fractional_value: object) -> None:
    definition_ids = _definition_ids()
    ids: dict[str, object] = {
        "remote_fractional": definition_ids["fractional_site_positions"],
        "remote_cartesian": definition_ids["cartesian_site_positions"],
        "remote_lattice": definition_ids["lattice_vectors"],
        "remote_dimensions": definition_ids["dimension_types"],
    }
    attributes: dict[str, object] = {
        "remote_cartesian": [[1, 1.5, 2]],
        "remote_lattice": [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
        "remote_dimensions": [1, 1, 1],
    }
    if fractional_value is None:
        attributes["remote_fractional"] = None
    backend = OptimadeStructure(_resource(ids=ids, attributes=attributes))

    assert backend.sites.reduced_coords.to_fractions() == [[Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]]


def test_fractional_failure_is_lazy_local_and_does_not_fall_back_to_cartesian() -> None:
    definition_ids = _definition_ids()
    resource = _resource(
        ids={
            "remote_fractional": definition_ids["fractional_site_positions"],
            "remote_cartesian": definition_ids["cartesian_site_positions"],
            "remote_species": definition_ids["species"],
        },
        attributes={
            "remote_fractional": [["invalid", 0, 0]],
            "remote_cartesian": [[0, 0, 0]],
            "remote_species": [{"name": "Na", "chemical_symbols": ["Na"], "concentration": [1]}],
        },
    )
    view = UnitcellStructureView(OptimadeStructure(resource))

    assert view.species[0].name == "Na"
    with pytest.raises(IncompleteOptimadeResourceError, match="fractional_site_positions"):
        _ = view.sites


@pytest.mark.parametrize(
    ("explicit_precision", "expected"),
    [(False, Fraction(1, 10_000)), (True, Fraction(1, 100))],
)
def test_fractional_decimal_precision_uses_selected_representation_unless_explicit(
    explicit_precision: bool, expected: Fraction
) -> None:
    definition_ids = _definition_ids()
    properties = {
        "renamed_fractional": {"$id": definition_ids["fractional_site_positions"]},
        "renamed_cartesian": {"$id": definition_ids["cartesian_site_positions"]},
    }
    precision_attribute = ""
    if explicit_precision:
        properties["renamed_precision"] = {"$id": precision_definitions()["_httk_coordinate_precision"].definition_id}
        precision_attribute = ',"renamed_precision":0.01'
    schema = OptimadeSchemaSnapshot(
        "structures",
        OptimadeDocument.create(
            json.dumps({"data": {"properties": properties}}),
            "https://example.test/info/structures",
        ),
    )
    document = OptimadeDocument.create(
        """{"data":[{"id":"precision","type":"structures","attributes":{
        "renamed_fractional":[[0.1250,0.0000,0.0000]],
        "renamed_cartesian":[[9.0,9.0,9.0]]"""
        + precision_attribute
        + "}}]}",
        "https://example.test/v1/structures",
    )
    backend = OptimadeStructure(OptimadeResource(document, 0, schema))

    assert backend.sites.precision == expected


def test_source_and_backend_are_retained_on_view_round_trip() -> None:
    resource = _resource()
    backend = StructureBackend.create(resource)
    assert isinstance(backend, OptimadeStructure)
    view = UnitcellStructureView(backend)

    assert UnitcellStructureView(view) is view
    assert view._backend is backend
    assert view.unwrap() is resource


def test_portable_structure_profile_is_semantic_and_exact() -> None:
    definition_ids = _definition_ids()
    schema = load_entry_type_schema(_STRUCTURES_ID)
    portable = {
        "immutable_id": "source_immutable",
        "last_modified": "source_modified",
        "elements": "source_elements",
        "nelements": "source_nelements",
        "elements_ratios": "source_ratios",
        "chemical_formula_descriptive": "source_descriptive",
        "chemical_formula_reduced": "source_reduced",
        "chemical_formula_anonymous": "source_anonymous",
        "nperiodic_dimensions": "source_periodicity",
        "nsites": "source_nsites",
        "structure_features": "source_features",
    }
    properties = {f"remote_{name}": {"$id": definition_ids[name]} for name in definition_ids}
    properties.update({remote: {"$id": schema.properties[name].definition_id} for name, remote in portable.items()})
    resource = _resource(
        ids={name: value["$id"] for name, value in properties.items()},
        attributes={
            "source_immutable": "immutable",
            "source_modified": "2025-01-02T03:04:05+00:00",
            "source_elements": ["Cl", "Na"],
            "source_nelements": 2,
            "source_ratios": [0.500, 0.500],
            "source_descriptive": "NaCl",
            "source_reduced": "ClNa",
            "source_anonymous": "AB",
            "source_periodicity": 3,
            "source_nsites": 2,
            "source_features": [],
        },
    )
    backend = OptimadeStructure(resource)

    assert backend.immutable_id == "immutable"
    assert backend.last_modified.isoformat() == "2025-01-02T03:04:05+00:00"
    assert backend.elements == ("Cl", "Na")
    assert backend.nelements == 2
    assert backend.elements_ratios == (Fraction(1, 2), Fraction(1, 2))
    assert backend.chemical_formula_descriptive == "NaCl"
    assert backend.chemical_formula_reduced == "ClNa"
    assert backend.chemical_formula_anonymous == "AB"
    assert backend.nperiodic_dimensions == 3
    assert backend.nsites == 2
    assert backend.structure_features == ()
