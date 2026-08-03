"""Register httk-atomistic's vendored entry-type and property schemas."""

from httk.core.register import register_entry_type_definition, register_property_definition

register_entry_type_definition(
    definition_id="https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures",
    resource="httk.registry.schemas.atomistic:structures.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/symmetry/affine_transformation",
    resource="httk.registry.schemas.atomistic:affine_transformation.json",
)
register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/spacegroups/centring_type",
    resource="httk.registry.schemas.atomistic:centring_type.json",
)
register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/pointgroups/crystal_system",
    resource="httk.registry.schemas.atomistic:crystal_system.json",
)
register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/core/fractional_coordinate_precision",
    resource="httk.registry.schemas.atomistic:fractional_coordinate_precision.json",
)
register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/spacegroups/hall_entry",
    resource="httk.registry.schemas.atomistic:hall_entry.json",
)
register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/spacegroups/is_reference_setting",
    resource="httk.registry.schemas.atomistic:is_reference_setting.json",
)
register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/core/length_precision",
    resource="httk.registry.schemas.atomistic:length_precision.json",
)
register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/spacegroups/setting_it_nc",
    resource="httk.registry.schemas.atomistic:setting_it_nc.json",
)
