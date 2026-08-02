"""Register entry providers and storage backings implemented by :mod:`httk.atomistic`."""

from httk.core import (
    register_entry_family,
    register_entry_provider,
    register_entry_record,
    register_optimade_entry_binding,
)

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

register_entry_family(
    name="structures",
    family="httk.atomistic.structure_entries:StructureEntry",
    definition_id="https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures",
)

register_entry_record(
    name="atomistic-unitcell-structure",
    family="structures",
    record="httk.atomistic.structure_record:UnitcellStructureRecord",
)

register_entry_record(
    name="atomistic-fundamental-domain-structure",
    family="structures",
    record="httk.atomistic.structure_record:FundamentalDomainStructureRecord",
)

register_entry_record(
    name="atomistic-asu-structure",
    family="structures",
    record="httk.atomistic.structure_record:ASUStructureRecord",
)
