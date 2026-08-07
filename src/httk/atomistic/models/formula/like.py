"""The accepted-input union for chemical-formula functions.

A ``str`` in this union is always a formula, never a filename.
"""

from collections.abc import Mapping
from typing import Any

import httk.atomistic.models.formula.anonymous
import httk.atomistic.models.formula.backend
import httk.atomistic.models.formula.composition
import httk.atomistic.models.formula.formula
import httk.atomistic.models.formula.view_base
import httk.atomistic.models.prototype.anonymous
import httk.atomistic.models.prototype.backend
import httk.atomistic.models.prototype.prototype
import httk.atomistic.models.prototype.view_base
import httk.atomistic.models.structure.backend
import httk.atomistic.models.structure.view
import httk.atomistic.storage.records

type ChemicalFormulaLike = (
    httk.atomistic.models.formula.backend.ChemicalFormulaBackend
    | httk.atomistic.models.formula.view_base.ChemicalFormulaViewBase
    | httk.atomistic.models.formula.composition.Composition
    | httk.atomistic.models.formula.formula.ChemicalFormula
    | httk.atomistic.models.formula.anonymous.AnonymousFormula
    | httk.atomistic.models.prototype.backend.AnonymousStructureBackend
    | httk.atomistic.models.prototype.view_base.AnonymousStructureViewBase
    | httk.atomistic.models.prototype.anonymous.AnonymousStructure
    | httk.atomistic.models.prototype.prototype.Prototype
    | httk.atomistic.storage.records.NormalizedCompositionRecord
    | httk.atomistic.models.structure.backend.StructureBackend
    | httk.atomistic.models.structure.view.StructureView
    | str
    | Mapping[str, Any]
)
