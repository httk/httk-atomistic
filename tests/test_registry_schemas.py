"""Tests for httk-atomistic's IRI schema registrations."""

import httk.atomistic  # noqa: F401  (import triggers the atomistic entry-family/record registrations)
from httk.core import load_entry_type_definition, load_property_definition
from httk.core.register import (
    known_entry_families,
    known_entry_records,
    known_entry_type_definitions,
    known_property_definitions,
    resolve_entry_record,
)

from httk.atomistic.entries.definitions import load_httk_definitions
from httk.atomistic.storage.records import PrototemplateRecord, StructuretypeRecord

STRUCTURES_ID = "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"
HTTK_PROPERTY_IDS = {
    "_httk_setting_it_nc": "https://schemas.httk.org/defs/v0.1/properties/spacegroups/setting_it_nc",
    "_httk_hall_entry": "https://schemas.httk.org/defs/v0.1/properties/spacegroups/hall_entry",
    "_httk_is_reference_setting": "https://schemas.httk.org/defs/v0.1/properties/spacegroups/is_reference_setting",
    "_httk_crystal_system": "https://schemas.httk.org/defs/v0.1/properties/pointgroups/crystal_system",
    "_httk_centring_type": "https://schemas.httk.org/defs/v0.1/properties/spacegroups/centring_type",
    "_httk_setting_transform": "https://schemas.httk.org/defs/v0.1/properties/symmetry/affine_transformation",
    "_httk_coordinate_precision": "https://schemas.httk.org/defs/v0.1/properties/core/fractional_coordinate_precision",
    "_httk_basis_precision": "https://schemas.httk.org/defs/v0.1/properties/core/length_precision",
    "_httk_species_charges": "https://schemas.httk.org/defs/v0.1/properties/chemistry/species_charges",
    "_httk_species_labels": "https://schemas.httk.org/defs/v0.1/properties/chemistry/species_labels",
    "_httk_species_spins": "https://schemas.httk.org/defs/v0.1/properties/chemistry/species_spins",
    "_httk_charge": "https://schemas.httk.org/defs/v0.1/properties/chemistry/structure_charge",
    "_httk_site_moments": "https://schemas.httk.org/defs/v0.1/properties/magnetism/site_moments",
}


def test_atomistic_property_schemas_are_registered_and_loadable() -> None:
    assert set(HTTK_PROPERTY_IDS.values()) <= set(known_property_definitions())
    for definition_id in HTTK_PROPERTY_IDS.values():
        assert load_property_definition(definition_id).definition_id == definition_id
    assert STRUCTURES_ID in known_entry_type_definitions()
    assert load_entry_type_definition(STRUCTURES_ID).definition_id == STRUCTURES_ID


def test_taxonomy_entry_families_and_records_are_registered() -> None:
    families = set(known_entry_families())
    assert {"prototemplates", "structuretypes"} <= families
    records = set(known_entry_records())
    assert {"atomistic-prototemplate", "atomistic-structuretype"} <= records
    assert resolve_entry_record("atomistic-prototemplate") is PrototemplateRecord
    assert resolve_entry_record("atomistic-structuretype") is StructuretypeRecord


def test_httk_definitions_keep_served_names_and_published_ids() -> None:
    file_names = {
        served_name: definition_id.rsplit("/", 1)[-1] for served_name, definition_id in HTTK_PROPERTY_IDS.items()
    }
    definitions = load_httk_definitions(file_names)

    for served_name, definition in definitions.items():
        assert definition.name == served_name
        assert definition.definition_id == HTTK_PROPERTY_IDS[served_name]
