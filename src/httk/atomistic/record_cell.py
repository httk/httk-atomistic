"""Backend for an exact stored cell record."""

from typing import Any

from httk.core import SurdScalar, SurdVector

from .cell_backend import CellBackend
from .structure_record import CellRecord, _basis_vector


class RecordCell(CellBackend):
    _record: CellRecord

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, CellRecord):
            return None
        return super().__new__(cls)

    def __init__(self, obj: CellRecord, **hints: Any) -> None:
        self._record = obj

    @property
    def basis(self) -> SurdVector:
        return _basis_vector(self._record.basis)

    @property
    def scale(self) -> SurdScalar:
        return SurdVector.one()

    @property
    def unscaled_basis(self) -> SurdVector:
        return self.basis

    @property
    def precision(self):
        return self._record.precision

    @property
    def periodicity(self) -> tuple[bool, bool, bool]:
        return self._record.periodicity  # type: ignore[return-value]

    def unwrap(self) -> CellRecord:
        return self._record
