"""Backend for an exact stored species record."""

from fractions import Fraction
from typing import Any

from .species_backend import SpeciesBackend
from .structure_record import SpeciesRecord, _concentration_precision_from_record


class RecordSpecies(SpeciesBackend):
    _record: SpeciesRecord

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, SpeciesRecord):
            return None
        return super().__new__(cls)

    def __init__(self, obj: SpeciesRecord, **hints: Any) -> None:
        self._record = obj

    @property
    def name(self) -> str:
        return self._record.name

    @property
    def chemical_symbols(self) -> tuple[str, ...]:
        return self._record.chemical_symbols

    @property
    def concentration(self) -> tuple[Fraction, ...]:
        return self._record.concentration

    @property
    def concentration_precision(self) -> tuple[Fraction | None, ...] | None:
        return _concentration_precision_from_record(self._record.concentration_precision)

    @property
    def mass(self) -> tuple[float, ...] | None:
        return self._record.mass

    @property
    def attached(self) -> tuple[str, ...] | None:
        return self._record.attached

    @property
    def nattached(self) -> tuple[int, ...] | None:
        return self._record.nattached

    @property
    def original_name(self) -> str | None:
        return self._record.original_name

    def unwrap(self) -> SpeciesRecord:
        return self._record
