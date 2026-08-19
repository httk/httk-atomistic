"""The accepted-input union for protopatterns."""

import httk.atomistic.models.protopattern.backend
import httk.atomistic.models.protopattern.protopattern
import httk.atomistic.models.protopattern.view_base

type ProtopatternLike = (
    httk.atomistic.models.protopattern.backend.ProtopatternBackend
    | httk.atomistic.models.protopattern.view_base.ProtopatternViewBase
    | httk.atomistic.models.protopattern.protopattern.Protopattern
    | str
)
