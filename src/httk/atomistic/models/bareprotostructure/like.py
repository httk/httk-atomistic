"""The accepted-input union for protostructures."""

from typing import TYPE_CHECKING

import httk.atomistic.models.bareprotostructure.backend
import httk.atomistic.models.bareprotostructure.bareprotostructure
import httk.atomistic.models.bareprotostructure.view_base

if TYPE_CHECKING:
    import httk.atomistic.models.bareprototype.view_base
    import httk.atomistic.models.protostructure.backend
    import httk.atomistic.models.protostructure.view_base
    import httk.atomistic.models.prototype.view_base
    import httk.atomistic.models.structure.backend
    import httk.atomistic.models.structure.view

type BareProtostructureLike = (
    httk.atomistic.models.bareprotostructure.backend.BareProtostructureBackend
    | httk.atomistic.models.bareprotostructure.view_base.BareProtostructureViewBase
    | httk.atomistic.models.bareprotostructure.bareprotostructure.BareProtostructure
    | str
    | httk.atomistic.models.protostructure.backend.ProtostructureBackend
    | httk.atomistic.models.protostructure.view_base.ProtostructureViewBase
    | httk.atomistic.models.bareprototype.view_base.BarePrototypeViewBase
    | httk.atomistic.models.prototype.view_base.PrototypeViewBase
    | httk.atomistic.models.structure.backend.StructureBackend
    | httk.atomistic.models.structure.view.StructureView
)
