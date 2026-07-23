from httk.core.register import register_entry_provider

register_entry_provider(
    name="atomistic-structures",
    factory="httk.atomistic.structure_entries:StructureEntryProvider",
)
