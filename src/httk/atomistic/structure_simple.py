"""
Backend wrapping a Structure in the Simple representation.
"""

from typing import Any

from .cell import Cell
from .sites import Sites
from .species import Species
from .structure import Structure
from .structure_backend import StructureBackend


class StructureSimple(StructureBackend):
    """
    Backend for a crystal structure backed by an actual ``Structure`` object.

    Its quartet accessors delegate to the wrapped Structure, and ``unwrap`` returns
    that Structure.
    """

    _structure: Structure

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if not isinstance(obj, Structure):
            return None
        if hints and hints.get("kind", "simple") != "simple":
            return None
        return super().__new__(cls)

    def __init__(self, obj: Structure, **hints: Any) -> None:
        self._structure = obj

    @property
    def cell(self) -> Cell:
        return self._structure.cell

    @property
    def sites(self) -> Sites:
        return self._structure.sites

    @property
    def species(self) -> tuple[Species, ...]:
        return self._structure.species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return self._structure.species_at_sites

    def unwrap(self) -> Any:
        return self._structure
