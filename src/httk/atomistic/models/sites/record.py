import fractions
from typing import Any, Self

from httk.core import FracVector

from httk.atomistic.models.sites.backend import SitesBackend
from httk.atomistic.storage.records import SitesRecord


class RecordSites(SitesBackend):
    r"""Backend for sites stored in an exact record.

    :param obj: The stored sites record.
    :param \**hints: Backend-selection hints.
    """

    _record: SitesRecord

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a sites record.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, SitesRecord):
            return None
        return cls(obj, **hints)

    def __init__(self, obj: SitesRecord, **hints: Any) -> None:
        self._record = obj

    @property
    def reduced_coords(self) -> FracVector:
        """Return the stored reduced coordinates.

        :return: The exact reduced coordinates.
        """
        return self._record.reduced_coords

    @property
    def precision(self) -> fractions.Fraction | None:
        """Return the stored coordinate precision.

        :return: The fractional precision, or ``None`` when unknown.
        """
        return self._record.precision

    def unwrap(self) -> SitesRecord:
        """Return the stored sites record.

        :return: The source record.
        """
        return self._record
