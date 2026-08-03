"""Backend for an exact stored sites record."""

from typing import Any

from httk.core import FracVector

from .sites_backend import SitesBackend
from .structure_record import SitesRecord


class RecordSites(SitesBackend):
    _record: SitesRecord

    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, SitesRecord):
            return None
        return super().__new__(cls)

    def __init__(self, obj: SitesRecord, **hints: Any) -> None:
        self._record = obj

    @property
    def reduced_coords(self) -> FracVector:
        return self._record.reduced_coords

    @property
    def precision(self):
        return self._record.precision

    def unwrap(self) -> SitesRecord:
        return self._record
