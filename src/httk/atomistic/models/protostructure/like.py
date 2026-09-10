"""The accepted-input union for protostructures."""

from typing import TYPE_CHECKING

import httk.atomistic.models.protostructure.backend
import httk.atomistic.models.protostructure.protostructure
import httk.atomistic.models.protostructure.view_base

if TYPE_CHECKING:
    import httk.atomistic.models.bareprotostructure.view_base
    import httk.atomistic.models.bareprototype.view_base
    import httk.atomistic.models.prototype.view_base

type ProtostructureLike = (
    httk.atomistic.models.protostructure.backend.ProtostructureBackend
    | httk.atomistic.models.protostructure.view_base.ProtostructureViewBase
    | httk.atomistic.models.protostructure.protostructure.Protostructure
    | httk.atomistic.models.bareprototype.view_base.BarePrototypeViewBase
    | httk.atomistic.models.bareprotostructure.view_base.BareProtostructureViewBase
    | httk.atomistic.models.prototype.view_base.PrototypeViewBase
)
