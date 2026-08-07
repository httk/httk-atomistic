"""The accepted-input union for protostructures."""

import httk.atomistic.models.protostructure.backend
import httk.atomistic.models.protostructure.protostructure
import httk.atomistic.models.protostructure.view_base

type ProtostructureLike = (
    httk.atomistic.models.protostructure.backend.ProtostructureBackend
    | httk.atomistic.models.protostructure.view_base.ProtostructureViewBase
    | httk.atomistic.models.protostructure.protostructure.Protostructure
)
