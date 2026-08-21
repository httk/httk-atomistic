"""The accepted-input union for prototemplates."""

import httk.atomistic.models.prototemplate.backend
import httk.atomistic.models.prototemplate.prototemplate
import httk.atomistic.models.prototemplate.view_base

type PrototemplateLike = (
    httk.atomistic.models.prototemplate.backend.PrototemplateBackend
    | httk.atomistic.models.prototemplate.view_base.PrototemplateViewBase
    | httk.atomistic.models.prototemplate.prototemplate.Prototemplate
    | str
)
