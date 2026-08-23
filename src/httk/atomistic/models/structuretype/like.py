"""The accepted-input union for structuretypes."""

import httk.atomistic.models.structuretype.backend
import httk.atomistic.models.structuretype.fundamental
import httk.atomistic.models.structuretype.structuretype
import httk.atomistic.models.structuretype.view_base

type StructuretypeLike = (
    httk.atomistic.models.structuretype.backend.StructuretypeBackend
    | httk.atomistic.models.structuretype.view_base.StructuretypeViewBase
    | httk.atomistic.models.structuretype.structuretype.Structuretype
    | httk.atomistic.models.structuretype.fundamental.FundamentalDomainTemplate
)
