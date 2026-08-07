"""
The abstract base class for all chemical-formula backends in httk-atomistic.
"""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.formula.api import ChemicalFormulaAPI


class ChemicalFormulaBackend(Backend["ChemicalFormulaBackend"], ChemicalFormulaAPI):
    """
    Abstract base class for all backends of chemical-formula data.

    Concrete backends carry a native representation and produce the canonical
    chemical-formula accessors declared by ``ChemicalFormulaAPI`` from it.
    """

    backend_classes: ClassVar[list[type[Backend[Any]]]]
