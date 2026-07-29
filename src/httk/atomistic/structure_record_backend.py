"""Structure backend for :class:`~httk.atomistic.StructureRecord` snapshots."""

from functools import cached_property
from typing import Any

from .cell import Cell
from .sites import Sites
from .species import Species
from .structure_backend import StructureBackend
from .structure_record import StructureRecord, _basis_vector


class StructureRecordBackend(StructureBackend):
    """Backend exposing a ``StructureRecord`` through the structure quartet."""

    _record: StructureRecord

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, StructureRecord):
            return None
        return super().__new__(cls)

    def __init__(self, obj: StructureRecord, **hints: Any) -> None:
        self._record = obj

    @cached_property
    def cell(self) -> Cell:  # pyright: ignore[reportIncompatibleMethodOverride]
        return Cell(
            _basis_vector(self._record.basis),
            precision=self._record.basis_precision,
            periodicity=self._record.periodicity,
        )

    @cached_property
    def sites(self) -> Sites:  # pyright: ignore[reportIncompatibleMethodOverride]
        return Sites(self._record.reduced_coords, precision=self._record.coordinate_precision)

    @cached_property
    def species(self) -> tuple[Species, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return tuple(value.to_species() for value in self._record.species)

    @cached_property
    def species_at_sites(self) -> tuple[str, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._record.species_at_sites

    def unwrap(self) -> StructureRecord:
        """Return the stored snapshot rather than the reconstructed structure."""
        return self._record
