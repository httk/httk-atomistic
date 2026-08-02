"""Tests for httk-atomistic's IRI schema registrations."""

from httk.core import load_entry_type_schema, load_property_definition
from httk.core.register import known_entry_type_schemas, known_property_definitions

from httk.atomistic.httk_definitions import load_httk_definitions

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
}


def test_atomistic_property_schemas_are_registered_and_loadable() -> None:
    assert set(HTTK_PROPERTY_IDS.values()) <= set(known_property_definitions())
    for definition_id in HTTK_PROPERTY_IDS.values():
        assert load_property_definition(definition_id).definition_id == definition_id
    assert STRUCTURES_ID in known_entry_type_schemas()
    assert load_entry_type_schema(STRUCTURES_ID).definition_id == STRUCTURES_ID


def test_httk_definitions_keep_served_names_and_published_ids() -> None:
    file_names = {
        served_name: definition_id.rsplit("/", 1)[-1]
        for served_name, definition_id in HTTK_PROPERTY_IDS.items()
    }
    definitions = load_httk_definitions(file_names)

    for served_name, definition in definitions.items():
        assert definition.name == served_name
        assert definition.definition_id == HTTK_PROPERTY_IDS[served_name]
