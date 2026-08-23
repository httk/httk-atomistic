"""The accepted-input union for chemical-formula functions.

A ``str`` in this union is always a formula, never a filename.
"""

from collections.abc import Mapping
from typing import Any

import httk.atomistic.models.formula.backend
import httk.atomistic.models.formula.composition
import httk.atomistic.models.formula.formula
import httk.atomistic.models.formula.formulatype
import httk.atomistic.models.formula.view_base
import httk.atomistic.models.protostructure.backend
import httk.atomistic.models.protostructure.protostructure
import httk.atomistic.models.protostructure.view_base
import httk.atomistic.models.prototype.backend
import httk.atomistic.models.prototype.prototype
import httk.atomistic.models.prototype.view_base
import httk.atomistic.models.structure.backend
import httk.atomistic.models.structure.view
import httk.atomistic.models.structuretype.backend
import httk.atomistic.models.structuretype.fundamental
import httk.atomistic.models.structuretype.structuretype
import httk.atomistic.models.structuretype.view_base
import httk.atomistic.storage.records

type ChemicalFormulaLike = (
    httk.atomistic.models.formula.backend.ChemicalFormulaBackend
    | httk.atomistic.models.formula.view_base.ChemicalFormulaViewBase
    | httk.atomistic.models.formula.composition.Composition
    | httk.atomistic.models.formula.formula.ChemicalFormula
    | httk.atomistic.models.formula.formulatype.Formulatype
    | httk.atomistic.models.structuretype.backend.StructuretypeBackend
    | httk.atomistic.models.structuretype.view_base.StructuretypeViewBase
    | httk.atomistic.models.structuretype.structuretype.Structuretype
    | httk.atomistic.models.structuretype.fundamental.FundamentalDomainTemplate
    | httk.atomistic.models.prototype.backend.PrototypeBackend
    | httk.atomistic.models.prototype.view_base.PrototypeViewBase
    | httk.atomistic.models.prototype.prototype.Prototype
    | httk.atomistic.models.protostructure.backend.ProtostructureBackend
    | httk.atomistic.models.protostructure.view_base.ProtostructureViewBase
    | httk.atomistic.models.protostructure.protostructure.Protostructure
    | httk.atomistic.storage.records.NormalizedCompositionRecord
    | httk.atomistic.models.structure.backend.StructureBackend
    | httk.atomistic.models.structure.view.StructureView
    | str
    | Mapping[str, Any]
)
