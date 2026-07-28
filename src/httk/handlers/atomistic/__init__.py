from httk.core.register import register_entry_provider, register_format_adapter

register_entry_provider(
    name="atomistic-structures",
    factory="httk.atomistic.structure_entries:StructureEntryProvider",
)

register_format_adapter(
    name="atomistic-structures",
    adapter="httk.atomistic.vasp_structures:structure_from_payload",
    formats=("cif", "vasp-poscar"),
)
