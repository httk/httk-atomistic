"""Register entry providers and records implemented by :mod:`httk.atomistic`."""

from httk.core import register_entry_provider, register_entry_record, register_optimade_entry_binding

register_entry_provider(
    name="atomistic-structures",
    factory="httk.atomistic.structure_entries:StructureEntryProvider",
)

register_optimade_entry_binding(
    name="atomistic-structure",
    definition_id="https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures",
    backend="httk.atomistic.optimade_structure:OptimadeStructure",
    view="httk.atomistic.unitcell_structure_view:UnitcellStructureView",
    query_fields=None,
)

# This is the storage record, not the served OPTIMADE structures shape.
register_entry_record(
    name="atomistic-structure-record",
    record="httk.atomistic.structure_record:StructureRecord",
    definition_id=None,
)
