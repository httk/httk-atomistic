"""The accepted-input union for crystal templates."""

import httk.atomistic.models.crystaltemplate.backend
import httk.atomistic.models.crystaltemplate.crystaltemplate
import httk.atomistic.models.crystaltemplate.fundamental
import httk.atomistic.models.crystaltemplate.view_base

type CrystalTemplateLike = (
    httk.atomistic.models.crystaltemplate.backend.CrystalTemplateBackend
    | httk.atomistic.models.crystaltemplate.view_base.CrystalTemplateViewBase
    | httk.atomistic.models.crystaltemplate.crystaltemplate.CrystalTemplate
    | httk.atomistic.models.crystaltemplate.fundamental.FundamentalDomainTemplate
)
