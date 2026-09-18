"""Register entry providers and storage backings implemented by :mod:`httk.atomistic`."""

from httk.core import register_entry_provider
from httk.core.register import (
    register_entry_family,
    register_entry_record,
    register_optimade_entry_binding,
)

register_entry_provider(
    name="atomistic-structures",
    factory="httk.atomistic.entries.structures:StructureEntryProvider",
)
register_entry_provider(
    name="atomistic-trajectories",
    factory="httk.atomistic.entries.trajectories:TrajectoryEntryProvider",
)

# Earliest OPTIMADE specification version in which each standard structures
# property name is a property of this entry type (determined from the
# specification text at release tags v1.0.1, v1.1.0, v1.2.0, v1.3.0). Gates
# standard-name inference for services that publish no property-definition $id.
_STRUCTURES_INTRODUCED_IN = {
    "assemblies": "1.0",
    "cartesian_site_positions": "1.0",
    "chemical_formula_anonymous": "1.0",
    "chemical_formula_descriptive": "1.0",
    "chemical_formula_hill": "1.0",
    "chemical_formula_reduced": "1.0",
    "dimension_types": "1.0",
    "elements": "1.0",
    "elements_ratios": "1.0",
    "fractional_site_positions": "1.3",
    "id": "1.0",
    "immutable_id": "1.0",
    "last_modified": "1.0",
    "lattice_vectors": "1.0",
    "nelements": "1.0",
    "nperiodic_dimensions": "1.0",
    "nsites": "1.0",
    "optimization_type": "1.3",
    "site_coordinate_span": "1.3",
    "site_coordinate_span_description": "1.3",
    "space_group_it_number": "1.2",
    "space_group_symbol_hall": "1.2",
    "space_group_symbol_hermann_mauguin": "1.2",
    "space_group_symbol_hermann_mauguin_extended": "1.2",
    "space_group_symmetry_operations_xyz": "1.2",
    "species": "1.0",
    "species_at_sites": "1.0",
    "structure_features": "1.0",
    "type": "1.0",
    "wyckoff_positions": "1.3",
}

register_optimade_entry_binding(
    name="atomistic-structure",
    definition_id="https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures",
    backend="httk.atomistic.models.structure.optimade:OptimadeStructure",
    view="httk.atomistic.models.structure.unitcell_view:UnitcellStructureView",
    query_fields=None,
    standard_property_versions=_STRUCTURES_INTRODUCED_IN,
)

register_entry_family(
    name="structures",
    family="httk.atomistic.entries.structures:StructureEntry",
    definition_id="https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures",
)
register_entry_family(
    name="trajectories",
    family="httk.atomistic.entries.trajectories:TrajectoryEntry",
    definition_id="https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/trajectories",
)

register_entry_record(
    name="atomistic-unitcell-structure",
    family="structures",
    record="httk.atomistic.storage.records:UnitcellStructureRecord",
)

register_entry_record(
    name="atomistic-fundamental-domain-structure",
    family="structures",
    record="httk.atomistic.storage.records:FundamentalDomainStructureRecord",
)

register_entry_record(
    name="atomistic-asu-structure",
    family="structures",
    record="httk.atomistic.storage.records:ASUStructureRecord",
)
register_entry_record(
    name="atomistic-trajectory",
    record="httk.atomistic.storage.records:TrajectoryRecord",
    family="trajectories",
    definition_id="https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/trajectories",
)

register_entry_family(
    name="protostructures",
    family="httk.atomistic.entries.prototypes:ProtostructureEntry",
)
register_entry_family(
    name="prototypes",
    family="httk.atomistic.entries.prototypes:PrototypeEntry",
)

register_entry_record(
    name="atomistic-protostructure",
    family="protostructures",
    record="httk.atomistic.storage.records:ProtostructureRecord",
)
register_entry_record(
    name="atomistic-prototype",
    family="prototypes",
    record="httk.atomistic.storage.records:PrototypeRecord",
)

register_entry_family(
    name="bare_prototypes",
    family="httk.atomistic.entries.prototypes:BarePrototypeEntry",
)
register_entry_record(
    name="atomistic-bare-prototype",
    family="bare_prototypes",
    record="httk.atomistic.storage.records:BarePrototypeRecord",
)

register_entry_family(
    name="bare_protostructures",
    family="httk.atomistic.entries.prototypes:BareProtostructureEntry",
)
register_entry_record(
    name="atomistic-bare-protostructure",
    family="bare_protostructures",
    record="httk.atomistic.storage.records:BareProtostructureRecord",
)
