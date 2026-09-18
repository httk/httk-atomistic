import datetime
import json
import logging
import pathlib
from fractions import Fraction

import pytest
from httk.core import load_entry_type_definition
from httk.core.optimade import (
    IncompleteOptimadeResourceError,
    OptimadeDocument,
    OptimadeResource,
    OptimadeSchemaSnapshot,
)

from httk.atomistic import (
    CartesianSiteMoments,
    OptimadeStructure,
    StructureBackend,
    StructureEntryProvider,
    UnitcellStructureView,
)
from httk.atomistic.entries.precision import precision_definitions
from httk.atomistic.models.structure import semantics as structure_semantics

_STRUCTURES_ID = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"


def _definition_ids() -> dict[str, str]:
    schema = load_entry_type_definition(_STRUCTURES_ID)
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
    info = OptimadeDocument.from_response(
        json.dumps({"data": {"properties": info_properties}}), "https://example.test/info"
    )
    document = OptimadeDocument.from_response(
        json.dumps({"data": [{"id": "example-1", "type": "structures", "attributes": values}]}),
        "https://example.test/v1/structures",
    )
    return OptimadeResource(document, 0, OptimadeSchemaSnapshot("structures", info))


def _semantic_resource(attributes: dict[str, object]) -> OptimadeResource:
    """Build a resource whose deliberately renamed fields cover *attributes*."""

    schema = load_entry_type_definition(_STRUCTURES_ID)
    properties = {
        f"transport_{index}": {"$id": schema.properties[name].definition_id} for index, name in enumerate(attributes)
    }
    renamed = {transport: attributes[name] for transport, name in zip(properties, attributes)}
    info = OptimadeDocument.from_response(
        json.dumps({"data": {"properties": properties}}), "https://example.test/info/structures"
    )
    document = OptimadeDocument.from_response(
        json.dumps({"data": [{"id": "semantic", "type": "structures", "attributes": renamed}]}),
        "https://example.test/v1/structures",
    )
    return OptimadeResource(document, 0, OptimadeSchemaSnapshot("structures", info))


def _moment_resource(moment_rows: object) -> OptimadeResource:
    schema = load_entry_type_definition(_STRUCTURES_ID)
    names = {
        "species_at_sites": "remote_species_at_sites",
        "nsites": "remote_nsites",
        "structure_features": "remote_structure_features",
    }
    properties = {remote: {"$id": schema.properties[name].definition_id} for name, remote in names.items()}
    info = OptimadeDocument.from_response(
        json.dumps({"data": {"properties": properties}}), "https://example.test/info/structures"
    )
    attributes = {
        names["species_at_sites"]: ["Na", "Cl"],
        names["nsites"]: 2,
        names["structure_features"]: ["_httk_magnetism"],
        "_httk_site_moments": moment_rows,
    }
    document = OptimadeDocument.from_response(
        json.dumps({"data": [{"id": "magnetic", "type": "structures", "attributes": attributes}]}),
        "https://example.test/v1/structures",
    )
    return OptimadeResource(document, 0, OptimadeSchemaSnapshot("structures", info))


def _complete_attributes() -> dict[str, object]:
    return {
        "elements": ["Cl", "Na"],
        "nelements": 2,
        "elements_ratios": [0.5, 0.5],
        "chemical_formula_descriptive": "NaCl",
        "chemical_formula_reduced": "ClNa",
        "chemical_formula_hill": "ClNa",
        "chemical_formula_anonymous": "AB",
        "dimension_types": [1, 1, 1],
        "nperiodic_dimensions": 3,
        "lattice_vectors": [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
        "space_group_symmetry_operations_xyz": ["x,y,z"],
        "space_group_symbol_hall": "P 1",
        "space_group_symbol_hermann_mauguin": "P 1",
        "space_group_symbol_hermann_mauguin_extended": "P 1",
        "space_group_it_number": 1,
        "cartesian_site_positions": [[0, 0, 0], [1, 1.5, 2]],
        "fractional_site_positions": [[0, 0, 0], [0.5, 0.5, 0.5]],
        "site_coordinate_span": "unit_cell",
        "site_coordinate_span_description": None,
        "nsites": 2,
        "species_at_sites": ["Na", "Cl"],
        "species": [
            {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1]},
            {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1]},
        ],
        "assemblies": None,
        "wyckoff_positions": ["a", "a"],
        "structure_features": [],
        "optimization_type": "experimental",
    }


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
        OptimadeDocument.from_response(
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
    document = OptimadeDocument.from_response(
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
        OptimadeDocument.from_response(
            json.dumps({"data": {"properties": properties}}),
            "https://example.test/info/structures",
        ),
    )
    document = OptimadeDocument.from_response(
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
    backend = StructureBackend._select_backend(resource)
    assert isinstance(backend, OptimadeStructure)
    view = UnitcellStructureView(backend)

    assert UnitcellStructureView(view) is view
    assert view._backend is backend
    assert view.unwrap() is resource


def test_portable_structure_profile_is_semantic_and_exact() -> None:
    definition_ids = _definition_ids()
    schema = load_entry_type_definition(_STRUCTURES_ID)
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


def test_complete_v13_profile_is_decoded_and_cross_checked_by_definition_iri() -> None:
    backend = OptimadeStructure(_semantic_resource(_complete_attributes()))

    assert backend.elements == ("Cl", "Na")
    assert backend.nelements == 2
    assert backend.elements_ratios == (Fraction(1, 2), Fraction(1, 2))
    assert backend.chemical_formula_descriptive == "NaCl"
    assert backend.chemical_formula_reduced == "ClNa"
    assert backend.chemical_formula_hill == "ClNa"
    assert backend.chemical_formula_anonymous == "AB"
    assert backend.dimension_types == (1, 1, 1)
    assert backend.nperiodic_dimensions == 3
    assert backend.lattice_vectors == (
        (Fraction(2), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(3), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(4)),
    )
    assert backend.space_group_symmetry_operations_xyz == ("x,y,z",)
    assert backend.space_group_symbol_hall == "P 1"
    assert backend.space_group_symbol_hermann_mauguin == "P 1"
    assert backend.space_group_symbol_hermann_mauguin_extended == "P 1"
    assert backend.space_group_it_number == 1
    assert backend.cartesian_site_positions == (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(3, 2), Fraction(2)),
    )
    assert backend.fractional_site_positions == (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)),
    )
    assert backend.site_coordinate_span == "unit_cell"
    assert backend.molecular is False
    assert backend.site_coordinate_span_description is None
    assert backend.nsites == 2
    assert backend.species_at_sites == ("Na", "Cl")
    assert tuple(species.name for species in backend.species) == ("Na", "Cl")
    assert backend.assemblies is None
    assert backend.wyckoff_positions == ("a", "a")
    assert backend.structure_features == ()
    assert backend.optimization_type == "experimental"


def test_dual_coordinate_arrays_are_cross_checked_lazily() -> None:
    attributes = _complete_attributes()
    attributes["cartesian_site_positions"] = [[0, 0, 0], [99, 99, 99]]
    backend = OptimadeStructure(_semantic_resource(attributes))

    assert backend.unwrap().id == "semantic"
    assert backend.species[0].name == "Na"
    with pytest.raises(IncompleteOptimadeResourceError, match="fractional_site_positions.*cartesian_site_positions"):
        _ = backend.fractional_site_positions


@pytest.mark.parametrize("span", ["molecular_entities", "other"])
def test_non_unit_cell_spans_remain_raw_but_refuse_native_projection(span: str) -> None:
    attributes = _complete_attributes()
    attributes["site_coordinate_span"] = span
    attributes["site_coordinate_span_description"] = "Two source entities" if span == "other" else None
    backend = OptimadeStructure(_semantic_resource(attributes))

    assert backend.site_coordinate_span == span
    assert backend.fractional_site_positions is not None
    assert backend.species_at_sites == ("Na", "Cl")
    assert tuple(value.name for value in backend.species) == ("Na", "Cl")
    assert backend.raw["attributes"] is backend.unwrap().unwrap()["attributes"]
    for component in ("cell", "sites"):
        with pytest.raises(IncompleteOptimadeResourceError, match=rf"{component}.*{span}"):
            _ = getattr(backend, component)
    with pytest.raises(IncompleteOptimadeResourceError, match=rf"{span}.*unit-cell"):
        UnitcellStructureView(backend)


def test_assemblies_preserve_exact_probabilities_and_present_vs_null() -> None:
    attributes = _complete_attributes()
    attributes["assemblies"] = [
        {"sites_in_groups": [[0], [1]], "group_probabilities": [0.3, 0.7]},
    ]
    attributes["structure_features"] = ["assemblies"]
    backend = OptimadeStructure(_semantic_resource(attributes))

    assert backend.assemblies is not None
    assert backend.assemblies[0].group_probabilities == (Fraction(3, 10), Fraction(7, 10))
    assert OptimadeStructure(_semantic_resource(_complete_attributes())).assemblies is None


@pytest.mark.parametrize(
    ("mutation", "property_name", "message"),
    [
        ({"nsites": 3}, "nsites", "disagrees"),
        ({"elements": ["Na", "Cl"]}, "elements", "alphabetical"),
        ({"chemical_formula_reduced": "ClNa2"}, "chemical_formula_reduced", "site composition"),
        ({"structure_features": ["disorder"]}, "structure_features", "disorder"),
        ({"space_group_it_number": 231}, "space_group_it_number", r"\[1, 230\]"),
    ],
)
def test_invalid_supplied_semantics_raise_only_when_accessed(
    mutation: dict[str, object], property_name: str, message: str
) -> None:
    attributes = _complete_attributes()
    attributes.update(mutation)
    backend = OptimadeStructure(_semantic_resource(attributes))

    assert backend.id == "semantic"
    with pytest.raises(IncompleteOptimadeResourceError, match=message):
        _ = getattr(backend, property_name)


def test_reduced_span_requires_symmetry_but_remains_lazy() -> None:
    attributes = _complete_attributes()
    attributes["site_coordinate_span"] = "fundamental_domain"
    attributes["space_group_symmetry_operations_xyz"] = None
    backend = OptimadeStructure(_semantic_resource(attributes))

    assert backend.id == "semantic"
    with pytest.raises(IncompleteOptimadeResourceError, match="requires space-group symmetry"):
        _ = backend.site_coordinate_span


def test_molecular_unit_cell_assertion_survives_semantic_projection() -> None:
    attributes = _complete_attributes()
    attributes["site_coordinate_span"] = "molecular_unit_cell"
    backend = OptimadeStructure(_semantic_resource(attributes))

    assert backend.molecular is True
    assert backend.cell.periodicity == (True, True, True)
    view = UnitcellStructureView(backend)
    assert view.site_coordinate_span == "molecular_unit_cell"
    assert view.space_group_it_number == 1
    assert view.wyckoff_positions == ("a", "a")


def test_source_composition_uses_precision_interval_and_provider_named_projection() -> None:
    attributes = _complete_attributes()
    attributes.update(
        {
            "elements_ratios": [0.3333, 0.6666],
            "chemical_formula_reduced": "ClNa2",
            "chemical_formula_hill": "ClNa2",
            "chemical_formula_anonymous": "A2B",
            "structure_features": ["implicit_atoms"],
        }
    )
    backend = OptimadeStructure(_semantic_resource(attributes))
    assert backend.composition.normalized
    assert backend.composition.normalization_status == "within_precision"
    record = next(iter(StructureEntryProvider({"remote": backend}).records("structures")))
    assert record["elements"] == ["Cl", "Na"]
    assert record["nelements"] == 2
    assert record["elements_ratios"] == [0.3333, 0.6666]
    assert record["chemical_formula_reduced"] == "ClNa2"
    assert record["chemical_formula_anonymous"] == "A2B"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("chemical_formula_reduced", "Cl99Na", "elements_ratios"),
        ("chemical_formula_hill", "Cl99Na", "elements_ratios"),
        ("chemical_formula_anonymous", "Z999", "symbol or coefficient order"),
        ("chemical_formula_descriptive", "rock salt sample", "invalid formula text"),
    ],
)
def test_source_formulas_are_validated_without_complete_sites(name: str, value: str, message: str) -> None:
    attributes = _complete_attributes()
    attributes["structure_features"] = ["implicit_atoms"]
    attributes[name] = value
    backend = OptimadeStructure(_semantic_resource(attributes))
    with pytest.raises(IncompleteOptimadeResourceError, match=message):
        _ = getattr(backend, name)


def test_source_symmetry_is_mutually_cross_checked() -> None:
    attributes = _complete_attributes()
    attributes["space_group_it_number"] = 2
    backend = OptimadeStructure(_semantic_resource(attributes))
    with pytest.raises(IncompleteOptimadeResourceError, match="mutually inconsistent|disagree"):
        _ = backend.space_group_it_number

    attributes = _complete_attributes()
    attributes["space_group_symmetry_operations_xyz"] = ["nonsense,a,b"]
    backend = OptimadeStructure(_semantic_resource(attributes))
    with pytest.raises(IncompleteOptimadeResourceError, match="three-coordinate strings"):
        _ = backend.space_group_symmetry_operations_xyz


def test_source_symmetry_settings_are_resolved_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    resolver = structure_semantics.declared_spacegroup_settings

    def counted(**kwargs: object) -> tuple[dict[str, object], ...]:
        nonlocal calls
        calls += 1
        return resolver(**kwargs)

    monkeypatch.setattr(structure_semantics, "declared_spacegroup_settings", counted)
    backend = OptimadeStructure(_semantic_resource(_complete_attributes()))

    assert backend.space_group_it_number == 1
    assert backend.space_group_symbol_hall == "P 1"
    assert backend.space_group_symmetry_operations_xyz == ("x,y,z",)
    assert backend.wyckoff_positions == ("a", "a")
    assert calls == 1


def test_span_description_is_only_valid_for_other() -> None:
    attributes = _complete_attributes()
    attributes["site_coordinate_span_description"] = "not applicable"
    backend = OptimadeStructure(_semantic_resource(attributes))
    with pytest.raises(IncompleteOptimadeResourceError, match="only valid"):
        _ = backend.site_coordinate_span_description


def test_official_descriptive_formula_group_abbreviation_is_accepted() -> None:
    attributes = _complete_attributes()
    attributes["chemical_formula_descriptive"] = "(CH3)3N+ - [CH2]2-OH = Me3N+ - CH2 - CH2OH"
    backend = OptimadeStructure(_semantic_resource(attributes))
    assert backend.chemical_formula_descriptive.endswith("CH2OH")


def test_remote_symmetry_uses_dimension_types_and_allows_arbitrary_settings() -> None:
    attributes = _complete_attributes()
    attributes.pop("nperiodic_dimensions")
    attributes["dimension_types"] = [0, 0, 0]
    backend = OptimadeStructure(_semantic_resource(attributes))
    with pytest.raises(IncompleteOptimadeResourceError, match="must be null"):
        _ = backend.space_group_it_number

    attributes = _complete_attributes()
    attributes.pop("space_group_symbol_hall")
    attributes.pop("space_group_symbol_hermann_mauguin")
    attributes.pop("space_group_symbol_hermann_mauguin_extended")
    attributes["space_group_it_number"] = 3
    attributes["space_group_symmetry_operations_xyz"] = ["x,y,z", "-x+2/7,y,-z"]
    backend = OptimadeStructure(_semantic_resource(attributes))
    assert backend.space_group_symmetry_operations_xyz == ("x,y,z", "-x+2/7,y,-z")

    attributes["space_group_symmetry_operations_xyz"] = ["x,y,z", "x+1/7,y,z"]
    backend = OptimadeStructure(_semantic_resource(attributes))
    with pytest.raises(IncompleteOptimadeResourceError, match="valid group|closed"):
        _ = backend.space_group_symmetry_operations_xyz

    attributes["space_group_symmetry_operations_xyz"] = ["x,y,z", "-x,y+1/2,-z"]
    backend = OptimadeStructure(_semantic_resource(attributes))
    with pytest.raises(IncompleteOptimadeResourceError, match="disagree"):
        _ = backend.space_group_symmetry_operations_xyz


def test_nullable_geometry_does_not_force_native_precision_projection() -> None:
    attributes = _complete_attributes()
    attributes["dimension_types"] = None
    attributes["nperiodic_dimensions"] = None
    backend = OptimadeStructure(_semantic_resource(attributes))
    record = next(iter(StructureEntryProvider({"remote": backend}).records("structures")))
    assert record["dimension_types"] is None
    assert record["_httk_coordinate_precision"] == pytest.approx(0.1)


def test_optimade_site_moments_and_magnetism_feature() -> None:
    backend = OptimadeStructure(_moment_resource([[1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]]))

    assert backend.site_moments == CartesianSiteMoments([[1.5, 0, 0], [-1.5, 0, 0]])
    assert backend.structure_features == ("_httk_magnetism",)


def test_optimade_site_moments_reject_malformed_shape() -> None:
    backend = OptimadeStructure(_moment_resource([[1.0, 0.0], [-1.0, 0.0]]))

    with pytest.raises(IncompleteOptimadeResourceError, match="_httk_site_moments"):
        _ = backend.site_moments


def test_unitcell_view_reads_metadata_through_the_decoding_backend() -> None:
    """A remote store row exposes raw JSON attributes by name; the view must not trust them."""
    import datetime
    from collections.abc import Mapping

    schema = load_entry_type_definition(_STRUCTURES_ID)
    names = (
        "lattice_vectors",
        "cartesian_site_positions",
        "species",
        "species_at_sites",
        "dimension_types",
        "last_modified",
    )
    properties = {name: {"$id": schema.properties[name].definition_id} for name in names}
    attributes = {
        "lattice_vectors": [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
        "cartesian_site_positions": [[0, 0, 0], [1, 1.5, 2]],
        "species": [
            {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1]},
            {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1]},
        ],
        "species_at_sites": ["Na", "Cl"],
        "dimension_types": [1, 1, 1],
        "last_modified": "2023-11-16T07:57:59Z",
    }

    class RawAttributeResource(OptimadeResource):
        def __getattr__(self, name: str) -> object:
            if name.startswith("__"):
                raise AttributeError(name)
            raw = self.unwrap().get("attributes")
            if isinstance(raw, Mapping) and name in raw:
                return raw[name]
            raise AttributeError(name)

    info = OptimadeDocument.from_response(json.dumps({"data": {"properties": properties}}), "https://example.test/info")
    document = OptimadeDocument.from_response(
        json.dumps({"data": [{"id": "raw-1", "type": "structures", "attributes": attributes}]}),
        "https://example.test/v1/structures",
    )
    resource = RawAttributeResource(document, 0, OptimadeSchemaSnapshot("structures", info))
    assert resource.last_modified == "2023-11-16T07:57:59Z"

    view = UnitcellStructureView(resource)
    assert view.last_modified == datetime.datetime(2023, 11, 16, 7, 57, 59, tzinfo=datetime.UTC)
    assert view.immutable_id is None
    assert view.formula == "ClNa"


_INFO_FIXTURES = pathlib.Path(__file__).parent / "data" / "optimade_info"


def _fixture_info(filename: str, source_url: str) -> OptimadeDocument:
    """Load a captured info document verbatim (fixture files are read-only)."""

    return OptimadeDocument.from_response((_INFO_FIXTURES / filename).read_text(), source_url)


def _fixture_info_with_version(filename: str, source_url: str, api_version: str | None) -> OptimadeDocument:
    """Load a captured info document with ``meta.api_version`` replaced or deleted.

    A string replaces the declared version; ``None`` deletes it (both simulate a
    differently versioned service). Only the in-memory copy is edited.
    """

    document = json.loads((_INFO_FIXTURES / filename).read_text())
    meta = document.setdefault("meta", {})
    if api_version is None:
        meta.pop("api_version", None)
    else:
        meta["api_version"] = api_version
    return OptimadeDocument.from_response(json.dumps(document), source_url)


def _resource_with_info(
    info: OptimadeDocument, attributes: dict[str, object], *, entry_type: str = "structures"
) -> OptimadeResource:
    """Wrap *attributes* in an entry document against schema snapshot *info*."""

    document = OptimadeDocument.from_response(
        json.dumps({"data": [{"id": "fixture-1", "type": "structures", "attributes": attributes}]}),
        "https://example.test/v1/structures",
    )
    return OptimadeResource(document, 0, OptimadeSchemaSnapshot(entry_type, info))


def _fixture_resource(filename: str, attributes: dict[str, object], *, source_url: str) -> OptimadeResource:
    """Build a resource against a captured real ``/info/structures`` document."""

    return _resource_with_info(_fixture_info(filename, source_url), attributes)


def _nacl_standard_attributes() -> dict[str, object]:
    """A valid rock-salt structure keyed by unprefixed standard property names."""

    return {
        "lattice_vectors": [[2, 0, 0], [0, 3, 0], [0, 0, 4]],
        "cartesian_site_positions": [[0, 0, 0], [1, 1.5, 2]],
        "species": [
            {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1]},
            {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1]},
        ],
        "species_at_sites": ["Na", "Cl"],
        "dimension_types": [1, 1, 1],
        "nsites": 2,
        "elements": ["Cl", "Na"],
        "nelements": 2,
        "chemical_formula_reduced": "ClNa",
        "last_modified": "2023-11-16T07:57:59Z",
        "immutable_id": "alx-1",
    }


def test_alexandria_schema_resolves_through_standard_name_rule() -> None:
    """A 1.1.0 provider that publishes no $id is presented as an httk structure."""

    resource = _fixture_resource(
        "alexandria_pbe_info_structures.json",
        _nacl_standard_attributes(),
        source_url="https://alexandria.example/v1/info/structures",
    )
    view = UnitcellStructureView(resource)

    assert str(view.formula) == "ClNa"
    assert view.species_at_sites == ("Na", "Cl")
    assert len(view.sites.reduced_coords.to_fractions()) == 2
    assert view.cell.volume == 24
    assert view.last_modified == datetime.datetime(2023, 11, 16, 7, 57, 59, tzinfo=datetime.UTC)
    assert view.last_modified.tzinfo is not None
    assert view.immutable_id == "alx-1"


def test_materials_project_space_group_resolves_at_declared_version() -> None:
    """A 1.2-introduced standard name resolves against a 1.2.0 provider."""

    attributes = _nacl_standard_attributes()
    attributes["space_group_it_number"] = 225
    resource = _fixture_resource(
        "materials_project_info_structures.json",
        attributes,
        source_url="https://materialsproject.example/v1/info/structures",
    )
    backend = OptimadeStructure(resource)

    assert backend.space_group_it_number == 225
    assert UnitcellStructureView(resource).space_group_it_number == 225


def test_standard_name_version_gate_hides_later_property() -> None:
    """The same 1.2 name stays unidentified when the service declares only 1.1.0."""

    attributes = _nacl_standard_attributes()
    attributes["space_group_it_number"] = 225
    info = _fixture_info_with_version(
        "materials_project_info_structures.json",
        "https://materialsproject.example/v1/info/structures",
        "1.1.0",
    )
    resource = _resource_with_info(info, attributes)
    backend = OptimadeStructure(resource)

    assert backend.space_group_it_number is None
    assert tuple(species.name for species in backend.species) == ("Na", "Cl")
    view = UnitcellStructureView(resource)
    assert str(view.formula) == "ClNa"
    assert view.space_group_it_number is None


def test_declared_definition_id_wins_over_unprefixed_name() -> None:
    """A declared $id under a renamed label beats an unprefixed same-name property."""

    schema = load_entry_type_definition(_STRUCTURES_ID)
    species_id = schema.properties["species"].definition_id
    properties = {
        "renamed_species": {"$id": species_id},
        "species": {"x-optimade-type": "list"},
    }
    info = OptimadeDocument.from_response(
        json.dumps({"data": {"properties": properties}, "meta": {"api_version": "1.1.0"}}),
        "https://declared.example/v1/info/structures",
    )
    attributes = {
        "renamed_species": [{"name": "Na", "chemical_symbols": ["Na"], "concentration": [1]}],
        "species": [{"name": "Wrong", "chemical_symbols": ["Cl"], "concentration": [1]}],
    }
    document = OptimadeDocument.from_response(
        json.dumps({"data": [{"id": "declared", "type": "structures", "attributes": attributes}]}),
        "https://example.test/v1/structures",
    )
    backend = OptimadeStructure(OptimadeResource(document, 0, OptimadeSchemaSnapshot("structures", info)))

    assert tuple(species.name for species in backend.species) == ("Na",)
    assert backend._remote_names_by_definition_id[species_id] == "renamed_species"


def test_provider_prefixed_name_is_never_inferred() -> None:
    """A provider-prefixed spelling of a standard name carries no standard identity."""

    info = OptimadeDocument.from_response(
        json.dumps(
            {"data": {"properties": {"_exmpl_species": {"x-optimade-type": "list"}}}, "meta": {"api_version": "1.1.0"}}
        ),
        "https://prefixed.example/v1/info/structures",
    )
    document = OptimadeDocument.from_response(
        json.dumps({"data": [{"id": "prefixed", "type": "structures", "attributes": {"_exmpl_species": []}}]}),
        "https://example.test/v1/structures",
    )
    backend = OptimadeStructure(OptimadeResource(document, 0, OptimadeSchemaSnapshot("structures", info)))

    with pytest.raises(IncompleteOptimadeResourceError, match="species"):
        _ = backend.species


def test_missing_api_version_disables_inference() -> None:
    """An info document that declares no version cannot complete standard names."""

    info = _fixture_info_with_version(
        "alexandria_pbe_info_structures.json",
        "https://alexandria.example/v1/info/structures",
        None,
    )
    backend = OptimadeStructure(_resource_with_info(info, _nacl_standard_attributes()))

    with pytest.raises(IncompleteOptimadeResourceError) as error:
        _ = backend.species
    message = str(error.value)
    assert "declares no OPTIMADE specification version" in message
    assert "'species'" in message


def test_identification_failure_message_names_declared_version_and_cause() -> None:
    """The required-property failure names the declared version and the actual cause."""

    # No declared version: inference cannot run, and the message says so.
    no_version = OptimadeStructure(
        _resource_with_info(
            _fixture_info_with_version(
                "alexandria_pbe_info_structures.json",
                "https://alexandria.example/v1/info/structures",
                None,
            ),
            _nacl_standard_attributes(),
        )
    )
    no_version_message = no_version._identification_failure("species")
    assert "declares no OPTIMADE specification version" in no_version_message
    assert "carries no standard identity" in no_version_message

    # A property not advertised at all, with a usable declared version present.
    prefixed_info = OptimadeDocument.from_response(
        json.dumps(
            {"data": {"properties": {"_exmpl_species": {"x-optimade-type": "list"}}}, "meta": {"api_version": "1.1.0"}}
        ),
        "https://prefixed.example/v1/info/structures",
    )
    not_advertised = OptimadeStructure(_resource_with_info(prefixed_info, {"_exmpl_species": []}))
    with pytest.raises(IncompleteOptimadeResourceError) as not_advertised_error:
        _ = not_advertised.species
    not_advertised_message = str(not_advertised_error.value)
    assert "'1.1.0'" in not_advertised_message
    assert "advertises no property named 'species'" in not_advertised_message

    # A 1.2 name at a service that declares only 1.1.0: the version gate fires.
    version_gated = OptimadeStructure(
        _resource_with_info(
            _fixture_info_with_version(
                "materials_project_info_structures.json",
                "https://materialsproject.example/v1/info/structures",
                "1.1.0",
            ),
            _nacl_standard_attributes(),
        )
    )
    gated_message = version_gated._identification_failure("space_group_it_number")
    assert "'1.1.0'" in gated_message
    assert "'1.2'" in gated_message
    assert "later than the declared version" in gated_message

    # A version-gated name that is NOT later than declared must not falsely
    # claim a version gate: an in-table 1.0 name at declared 1.2.0.
    not_later = OptimadeStructure(
        _fixture_resource(
            "materials_project_info_structures.json",
            _nacl_standard_attributes(),
            source_url="https://materialsproject.example/v1/info/structures",
        )
    )
    not_later_message = not_later._identification_failure("elements")
    assert "later than the declared version" not in not_later_message

    # An unusable (non-major-1) declared version disables inference.
    unusable = OptimadeStructure(
        _resource_with_info(
            _fixture_info_with_version(
                "alexandria_pbe_info_structures.json",
                "https://alexandria.example/v1/info/structures",
                "2.0.0",
            ),
            _nacl_standard_attributes(),
        )
    )
    unusable_message = unusable._identification_failure("species")
    assert "'2.0.0'" in unusable_message
    assert "which is not a usable major-1 version" in unusable_message

    # A non-standard entry type carries no standard-name meaning at all.
    vendor_info = OptimadeDocument.from_response(
        json.dumps(
            {"data": {"properties": {"lattice_vectors": {"x-optimade-type": "list"}}}, "meta": {"api_version": "1.2.0"}}
        ),
        "https://vendor.example/v1/info/structures-vendor",
    )
    vendor = OptimadeStructure(
        _resource_with_info(vendor_info, {"lattice_vectors": [[1, 0, 0]]}, entry_type="structures-vendor")
    )
    vendor_message = vendor._identification_failure("lattice_vectors")
    assert "'structures-vendor'" in vendor_message
    assert "not a standard OPTIMADE entry type" in vendor_message


def _materials_project_attributes() -> dict[str, object]:
    """Reproduce the Materials Project (api 1.2.0) pattern for ``mp-1018810``.

    ``elements_ratios`` is ordered by ``species_at_sites`` (Na first) rather
    than by the alphabetical ``elements`` list, so it disagrees with the
    projected site composition while every formula stays correct.

    :return: Structure attributes with misordered ratios but correct formulas.
    """
    return {
        "elements": ["Cl", "Na"],
        "nelements": 2,
        "elements_ratios": [0.6666666666666666, 0.3333333333333333],
        "chemical_formula_descriptive": "Na4Cl2",
        "chemical_formula_reduced": "ClNa2",
        "chemical_formula_hill": "Cl2Na4",
        "chemical_formula_anonymous": "A2B",
        "nsites": 6,
        "lattice_vectors": [[6, 0, 0], [0, 6, 0], [0, 0, 6]],
        "cartesian_site_positions": [[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3], [3, 3, 0], [0, 3, 3]],
        "species_at_sites": ["Na", "Na", "Na", "Na", "Cl", "Cl"],
        "species": [
            {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1]},
            {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1]},
        ],
        "structure_features": [],
        "dimension_types": [1, 1, 1],
        "nperiodic_dimensions": 3,
        "site_coordinate_span": "unit_cell",
    }


def test_misordered_elements_ratios_do_not_invalidate_correct_formulas() -> None:
    backend = OptimadeStructure(_semantic_resource(_materials_project_attributes()))
    assert backend.chemical_formula_reduced == "ClNa2"
    assert backend.chemical_formula_anonymous == "A2B"
    assert backend.chemical_formula_hill == "Cl2Na4"
    assert backend.chemical_formula_descriptive == "Na4Cl2"
    assert UnitcellStructureView(backend).formula == "ClNa2"
    with pytest.raises(IncompleteOptimadeResourceError, match="disagrees with the supplied site composition"):
        _ = backend.elements_ratios


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("chemical_formula_reduced", "ClNa"),
        ("chemical_formula_anonymous", "AB"),
        ("chemical_formula_hill", "ClNa"),
    ],
)
def test_formula_inconsistent_with_sites_still_raises(name: str, value: str) -> None:
    attributes = _materials_project_attributes()
    attributes[name] = value
    backend = OptimadeStructure(_semantic_resource(attributes))
    with pytest.raises(IncompleteOptimadeResourceError, match="disagrees with the supplied site composition"):
        _ = getattr(backend, name)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("chemical_formula_reduced", "Cl99Na"),
        ("chemical_formula_hill", "Cl99Na"),
    ],
)
def test_source_formulas_use_ratios_when_sites_are_absent(name: str, value: str) -> None:
    attributes = {
        "elements": ["Cl", "Na"],
        "nelements": 2,
        "elements_ratios": [0.5, 0.5],
        "structure_features": [],
        name: value,
    }
    backend = OptimadeStructure(_semantic_resource(attributes))
    assert backend._composition_from_sites is None
    with pytest.raises(IncompleteOptimadeResourceError, match="disagrees with 'elements_ratios'"):
        _ = getattr(backend, name)


def test_offsetless_last_modified_is_reported_once_and_decoded_as_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A Materials Project style offset-less timestamp is unknown, not naive UTC."""
    import httk.core.optimade.entries as optimade_entries

    optimade_entries._naive_timestamp_origins_warned.clear()
    attributes = _nacl_standard_attributes()
    attributes["last_modified"] = "2023-02-11T01:06:23.403000"
    resource = _fixture_resource(
        "alexandria_pbe_info_structures.json",
        attributes,
        source_url="https://alexandria.example/v1/info/structures",
    )
    backend = OptimadeStructure(resource)

    with caplog.at_level(logging.WARNING, logger="httk.core.optimade.entries"):
        assert backend.last_modified is None
        view = UnitcellStructureView(resource)
        assert view.last_modified is None
        assert str(view.formula) == "ClNa"
        assert view.species_at_sites == ("Na", "Cl")
        assert len(view.sites.reduced_coords.to_fractions()) == 2
        assert backend.immutable_id == "alx-1"

    optimade_warnings = [
        record
        for record in caplog.records
        if getattr(record, "context", None) == "optimade" and "example.test" in record.getMessage()
    ]
    assert len(optimade_warnings) == 1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("chemical_formula_reduced", "ClK2"),
        ("chemical_formula_hill", "Cl2K4"),
    ],
)
def test_named_formula_naming_foreign_element_raises_against_elements_with_sites(name: str, value: str) -> None:
    """The element-set check does not depend on ratio ordering, so it stays unconditional."""
    attributes = _materials_project_attributes()
    attributes[name] = value
    backend = OptimadeStructure(_semantic_resource(attributes))
    with pytest.raises(IncompleteOptimadeResourceError, match="disagrees with 'elements'"):
        _ = getattr(backend, name)


def _incomplete_projection_attributes(anonymous: str) -> dict[str, object]:
    """Rock salt plus a pure-unknown (``X``) site, so the projection is incomplete.

    The unknown site leaves ``chemical_formula_anonymous`` undefined on the
    projection while the known amounts stay 1:1, exercising the anonymous
    ratio-based fallback when sites are present.

    :param anonymous: The declared anonymous formula under test.
    :return: Structure attributes whose projected anonymous formula is ``None``.
    """
    return {
        "elements": ["Cl", "Na"],
        "nelements": 2,
        "elements_ratios": [0.5, 0.5],
        "chemical_formula_anonymous": anonymous,
        "nsites": 3,
        "lattice_vectors": [[6, 0, 0], [0, 6, 0], [0, 0, 6]],
        "cartesian_site_positions": [[0, 0, 0], [3, 0, 0], [0, 3, 0]],
        "species_at_sites": ["Na", "Cl", "Vac"],
        "species": [
            {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1]},
            {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1]},
            {"name": "Vac", "chemical_symbols": ["X"], "concentration": [1]},
        ],
        "structure_features": [],
        "dimension_types": [1, 1, 1],
        "site_coordinate_span": "unit_cell",
    }


def test_anonymous_formula_falls_back_to_ratios_when_projection_has_no_anonymous_formula() -> None:
    consistent = OptimadeStructure(_semantic_resource(_incomplete_projection_attributes("AB")))
    assert consistent._composition_from_sites is not None
    assert consistent._composition_from_sites.chemical_formula_anonymous is None
    assert consistent.chemical_formula_anonymous == "AB"

    inconsistent = OptimadeStructure(_semantic_resource(_incomplete_projection_attributes("A3B")))
    with pytest.raises(IncompleteOptimadeResourceError, match="disagrees with 'elements_ratios'"):
        _ = inconsistent.chemical_formula_anonymous
