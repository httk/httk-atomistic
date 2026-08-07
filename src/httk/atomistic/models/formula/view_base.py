"""
The abstract base class for all chemical-formula views in httk-atomistic.
"""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.formula.backend import ChemicalFormulaBackend


class ChemicalFormulaViewBase(View[ChemicalFormulaBackend]):
    """
    Abstract base class for all views of chemical-formula data.
    """

    _backend_base_cls: ClassVar[type[ChemicalFormulaBackend]] = ChemicalFormulaBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


ChemicalFormulaViewBase._view_base_cls = ChemicalFormulaViewBase
