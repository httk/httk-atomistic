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

register_optimade_entry_binding(
    name="atomistic-structure",
    definition_id="https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures",
    backend="httk.atomistic.models.structure.optimade:OptimadeStructure",
    view="httk.atomistic.models.structure.unitcell_view:UnitcellStructureView",
    query_fields=None,
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
register_entry_family(
    name="protochromas",
    family="httk.atomistic.entries.prototypes:ProtochromaEntry",
)
register_entry_family(
    name="crystallotypes",
    family="httk.atomistic.entries.prototypes:CrystallotypeEntry",
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
register_entry_record(
    name="atomistic-protochroma",
    family="protochromas",
    record="httk.atomistic.storage.records:ProtochromaRecord",
)
register_entry_record(
    name="atomistic-crystallotype",
    family="crystallotypes",
    record="httk.atomistic.storage.records:CrystallotypeRecord",
)
