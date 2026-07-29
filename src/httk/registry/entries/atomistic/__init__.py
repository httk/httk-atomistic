"""Register entry providers and records implemented by :mod:`httk.atomistic`."""

from httk.core import register_entry_provider, register_entry_record

register_entry_provider(
    name="atomistic-structures",
    factory="httk.atomistic.structure_entries:StructureEntryProvider",
)

# This is the storage record, not the served OPTIMADE structures shape.
register_entry_record(
    name="atomistic-structure-record",
    record="httk.atomistic.structure_record:StructureRecord",
    definition_id=None,
)
