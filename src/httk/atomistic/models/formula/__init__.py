from typing import TYPE_CHECKING

from .formulapattern import Formulapattern
from .formulapattern_string import FormulapatternString
from .formulapattern_view import FormulapatternView

# Alias for discoverability; canonical name is Formulapattern (see docs/prototypes.md).
AnonymousFormula = Formulapattern
AnonymousFormulaView = FormulapatternView
AnonymousFormulaString = FormulapatternString
from .api import ChemicalFormulaAPI
from .backend import ChemicalFormulaBackend
from .composition import Composition
from .composition_view import CompositionView
from .diagnostics import CompositionDiagnostic
from .formula import ChemicalFormula
from .formula_string import FormulaString
from .formula_view import ChemicalFormulaView
from .notation import (
    anonymous_symbol,
    parse_anonymous_formula,
    parse_reduced_formula,
    reduced_coefficients,
    render_anonymous,
    render_reduced,
    try_parse_anonymous,
    try_parse_reduced,
)
from .view_base import ChemicalFormulaViewBase

__all__ = [
    "AnonymousFormula",
    "AnonymousFormulaString",
    "AnonymousFormulaView",
    "ChemicalFormula",
    "ChemicalFormulaAPI",
    "ChemicalFormulaBackend",
    "ChemicalFormulaLike",
    "ChemicalFormulaView",
    "ChemicalFormulaViewBase",
    "Composition",
    "CompositionDiagnostic",
    "CompositionView",
    "FormulaString",
    "Formulapattern",
    "FormulapatternString",
    "FormulapatternView",
    "RecordComposition",
    "StructureComposition",
    "anonymous_symbol",
    "parse_anonymous_formula",
    "parse_reduced_formula",
    "reduced_coefficients",
    "render_anonymous",
    "render_reduced",
    "try_parse_anonymous",
    "try_parse_reduced",
]

if TYPE_CHECKING:
    from .like import ChemicalFormulaLike
    from .record import RecordComposition
    from .structure import StructureComposition


def __getattr__(name: str) -> object:
    if name == "RecordComposition":
        from .record import RecordComposition

        globals()[name] = RecordComposition
        return RecordComposition
    if name == "StructureComposition":
        from .structure import StructureComposition

        globals()[name] = StructureComposition
        return StructureComposition
    if name == "ChemicalFormulaLike":
        from .like import ChemicalFormulaLike

        globals()[name] = ChemicalFormulaLike
        return ChemicalFormulaLike
    raise AttributeError(name)
