"""The accepted-input union for prototypes."""

from typing import TYPE_CHECKING

import httk.atomistic.models.bareprototype.backend
import httk.atomistic.models.bareprototype.bareprototype
import httk.atomistic.models.bareprototype.label
import httk.atomistic.models.bareprototype.view_base

if TYPE_CHECKING:
    import httk.atomistic.models.bareprotostructure.backend
    import httk.atomistic.models.bareprotostructure.view_base
    import httk.atomistic.models.protostructure.backend
    import httk.atomistic.models.protostructure.view_base
    import httk.atomistic.models.prototype.backend
    import httk.atomistic.models.prototype.view_base
    import httk.atomistic.models.structure.backend
    import httk.atomistic.models.structure.view
    import httk.atomistic.models.structuretype.backend
    import httk.atomistic.models.structuretype.view_base

type BarePrototypeLike = (
    httk.atomistic.models.bareprototype.backend.BarePrototypeBackend
    | httk.atomistic.models.bareprototype.view_base.BarePrototypeViewBase
    | httk.atomistic.models.bareprototype.bareprototype.BarePrototype
    | httk.atomistic.models.bareprototype.label.BarePrototypeLabel
    | str
    | httk.atomistic.models.prototype.backend.PrototypeBackend
    | httk.atomistic.models.prototype.view_base.PrototypeViewBase
    | httk.atomistic.models.protostructure.backend.ProtostructureBackend
    | httk.atomistic.models.protostructure.view_base.ProtostructureViewBase
    | httk.atomistic.models.bareprotostructure.backend.BareProtostructureBackend
    | httk.atomistic.models.bareprotostructure.view_base.BareProtostructureViewBase
    | httk.atomistic.models.structure.backend.StructureBackend
    | httk.atomistic.models.structure.view.StructureView
    | httk.atomistic.models.structuretype.backend.StructuretypeBackend
    | httk.atomistic.models.structuretype.view_base.StructuretypeViewBase
)
