"""Structure backend for :class:`~httk.atomistic.StructureRecord` snapshots."""

from typing import Any

from .cell import Cell
from .sites import Sites
from .species import Species
from .structure import Structure
from .structure_backend import StructureBackend
from .structure_record import StructureRecord


class StructureRecordBackend(StructureBackend):
    """Backend exposing a ``StructureRecord`` through the structure quartet."""

    _record: StructureRecord
    _structure_cache: Structure | None

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, StructureRecord):
            return None
        return super().__new__(cls)

    def __init__(self, obj: StructureRecord, **hints: Any) -> None:
        self._record = obj
        self._structure_cache = None

    def _structure(self) -> Structure:
        if self._structure_cache is None:
            self._structure_cache = self._record.to_structure()
        return self._structure_cache

    @property
    def cell(self) -> Cell:
        return self._structure().cell

    @property
    def sites(self) -> Sites:
        return self._structure().sites

    @property
    def species(self) -> tuple[Species, ...]:
        return self._structure().species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return self._structure().species_at_sites

    def unwrap(self) -> StructureRecord:
        """Return the stored snapshot rather than the reconstructed structure."""
        return self._record
