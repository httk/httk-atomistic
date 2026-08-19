"""The accepted-input union for chemical-formula functions.

A ``str`` in this union is always a formula, never a filename.
"""

from collections.abc import Mapping
from typing import Any

import httk.atomistic.models.crystalpattern.backend
import httk.atomistic.models.crystalpattern.crystalpattern
import httk.atomistic.models.crystalpattern.fundamental
import httk.atomistic.models.crystalpattern.view_base
import httk.atomistic.models.formula.backend
import httk.atomistic.models.formula.composition
import httk.atomistic.models.formula.formula
import httk.atomistic.models.formula.formulapattern
import httk.atomistic.models.formula.view_base
import httk.atomistic.models.protostructure.backend
import httk.atomistic.models.protostructure.protostructure
import httk.atomistic.models.protostructure.view_base
import httk.atomistic.models.structure.backend
import httk.atomistic.models.structure.view
import httk.atomistic.storage.records

type ChemicalFormulaLike = (
    httk.atomistic.models.formula.backend.ChemicalFormulaBackend
    | httk.atomistic.models.formula.view_base.ChemicalFormulaViewBase
    | httk.atomistic.models.formula.composition.Composition
    | httk.atomistic.models.formula.formula.ChemicalFormula
    | httk.atomistic.models.formula.formulapattern.Formulapattern
    | httk.atomistic.models.crystalpattern.backend.CrystalPatternBackend
    | httk.atomistic.models.crystalpattern.view_base.CrystalPatternViewBase
    | httk.atomistic.models.crystalpattern.crystalpattern.CrystalPattern
    | httk.atomistic.models.crystalpattern.fundamental.FundamentalDomainPattern
    | httk.atomistic.models.protostructure.backend.ProtostructureBackend
    | httk.atomistic.models.protostructure.view_base.ProtostructureViewBase
    | httk.atomistic.models.protostructure.protostructure.Protostructure
    | httk.atomistic.storage.records.NormalizedCompositionRecord
    | httk.atomistic.models.structure.backend.StructureBackend
    | httk.atomistic.models.structure.view.StructureView
    | str
    | Mapping[str, Any]
)
