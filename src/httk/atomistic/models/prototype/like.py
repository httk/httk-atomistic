"""The accepted-input union for prototypes."""

import httk.atomistic.models.prototype.backend
import httk.atomistic.models.prototype.prototype
import httk.atomistic.models.prototype.view_base

type PrototypeLike = (
    httk.atomistic.models.prototype.backend.PrototypeBackend
    | httk.atomistic.models.prototype.view_base.PrototypeViewBase
    | httk.atomistic.models.prototype.prototype.Prototype
)
