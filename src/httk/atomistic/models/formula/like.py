"""The accepted-input union for chemical-formula functions.

A ``str`` in this union is always a formula, never a filename.
"""

from collections.abc import Mapping
from typing import Any

import httk.atomistic.models.crystaltemplate.backend
import httk.atomistic.models.crystaltemplate.crystaltemplate
import httk.atomistic.models.crystaltemplate.fundamental
import httk.atomistic.models.crystaltemplate.view_base
import httk.atomistic.models.formula.backend
import httk.atomistic.models.formula.composition
import httk.atomistic.models.formula.formula
import httk.atomistic.models.formula.formulatemplate
import httk.atomistic.models.formula.view_base
import httk.atomistic.models.protostructure.backend
import httk.atomistic.models.protostructure.protostructure
import httk.atomistic.models.protostructure.view_base
import httk.atomistic.models.prototemplate.backend
import httk.atomistic.models.prototemplate.prototemplate
import httk.atomistic.models.prototemplate.view_base
import httk.atomistic.models.structure.backend
import httk.atomistic.models.structure.view
import httk.atomistic.storage.records

type ChemicalFormulaLike = (
    httk.atomistic.models.formula.backend.ChemicalFormulaBackend
    | httk.atomistic.models.formula.view_base.ChemicalFormulaViewBase
    | httk.atomistic.models.formula.composition.Composition
    | httk.atomistic.models.formula.formula.ChemicalFormula
    | httk.atomistic.models.formula.formulatemplate.Formulatemplate
    | httk.atomistic.models.crystaltemplate.backend.CrystalTemplateBackend
    | httk.atomistic.models.crystaltemplate.view_base.CrystalTemplateViewBase
    | httk.atomistic.models.crystaltemplate.crystaltemplate.CrystalTemplate
    | httk.atomistic.models.crystaltemplate.fundamental.FundamentalDomainTemplate
    | httk.atomistic.models.prototemplate.backend.PrototemplateBackend
    | httk.atomistic.models.prototemplate.view_base.PrototemplateViewBase
    | httk.atomistic.models.prototemplate.prototemplate.Prototemplate
    | httk.atomistic.models.protostructure.backend.ProtostructureBackend
    | httk.atomistic.models.protostructure.view_base.ProtostructureViewBase
    | httk.atomistic.models.protostructure.protostructure.Protostructure
    | httk.atomistic.storage.records.NormalizedCompositionRecord
    | httk.atomistic.models.structure.backend.StructureBackend
    | httk.atomistic.models.structure.view.StructureView
    | str
    | Mapping[str, Any]
)
