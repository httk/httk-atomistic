"""The accepted-input union for prototypes."""

from typing import TYPE_CHECKING

import httk.atomistic.models.prototype.backend
import httk.atomistic.models.prototype.label
import httk.atomistic.models.prototype.prototype
import httk.atomistic.models.prototype.view_base

if TYPE_CHECKING:
    import httk.atomistic.models.bareprotostructure.view_base
    import httk.atomistic.models.bareprototype.view_base
    import httk.atomistic.models.protostructure.backend
    import httk.atomistic.models.protostructure.view_base

type PrototypeLike = (
    httk.atomistic.models.prototype.backend.PrototypeBackend
    | httk.atomistic.models.prototype.view_base.PrototypeViewBase
    | httk.atomistic.models.prototype.prototype.Prototype
    | httk.atomistic.models.prototype.label.PrototypeLabel
    | httk.atomistic.models.protostructure.backend.ProtostructureBackend
    | httk.atomistic.models.protostructure.view_base.ProtostructureViewBase
    | httk.atomistic.models.bareprototype.view_base.BarePrototypeViewBase
    | httk.atomistic.models.bareprotostructure.view_base.BareProtostructureViewBase
)
