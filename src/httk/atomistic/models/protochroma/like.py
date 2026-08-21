"""The accepted-input union for protochromas."""

import httk.atomistic.models.protochroma.backend
import httk.atomistic.models.protochroma.protochroma
import httk.atomistic.models.protochroma.view_base

type ProtochromaLike = (
    httk.atomistic.models.protochroma.backend.ProtochromaBackend
    | httk.atomistic.models.protochroma.view_base.ProtochromaViewBase
    | httk.atomistic.models.protochroma.protochroma.Protochroma
    | str
)
