"""The accepted-input union for protonetstructures."""

import httk.atomistic.models.protonetstructure.backend
import httk.atomistic.models.protonetstructure.label
import httk.atomistic.models.protonetstructure.protonetstructure
import httk.atomistic.models.protonetstructure.view_base

type ProtonetstructureLike = (
    httk.atomistic.models.protonetstructure.backend.ProtonetstructureBackend
    | httk.atomistic.models.protonetstructure.view_base.ProtonetstructureViewBase
    | httk.atomistic.models.protonetstructure.protonetstructure.Protonetstructure
    | httk.atomistic.models.protonetstructure.label.ProtonetstructureLabel
    | str
)
