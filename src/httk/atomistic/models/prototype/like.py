"""The accepted-input union for anonymous structures."""

import httk.atomistic.models.prototype.anonymous
import httk.atomistic.models.prototype.backend
import httk.atomistic.models.prototype.prototype
import httk.atomistic.models.prototype.view_base

type AnonymousStructureLike = (
    httk.atomistic.models.prototype.backend.AnonymousStructureBackend
    | httk.atomistic.models.prototype.view_base.AnonymousStructureViewBase
    | httk.atomistic.models.prototype.anonymous.AnonymousStructure
    | httk.atomistic.models.prototype.prototype.Prototype
)
