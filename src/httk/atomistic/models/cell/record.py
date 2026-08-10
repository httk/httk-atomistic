"""Backend for an exact stored cell record."""

import fractions
from typing import Any, Self

from httk.core import SurdScalar, SurdVector

from httk.atomistic.models.cell.backend import CellBackend
from httk.atomistic.storage.records import CellRecord, _basis_vector


class RecordCell(CellBackend):
    r"""Backend for a cell stored in an exact record.

    :param obj: The stored cell record.
    :param \**hints: Backend-selection hints.
    """

    _record: CellRecord

    @classmethod
    def _backend_adopt(cls, obj: Any, **hints: Any) -> Self | None:
        r"""Adopt a cell record.

        :param obj: The source object to adopt.
        :param \**hints: Backend-selection hints.
        :return: An initialized backend, or ``None`` when this backend declines ``obj``.
        """
        if hints and hints.get("kind", "record") != "record":
            return None
        if not isinstance(obj, CellRecord):
            return None
        return cls(obj, **hints)

    def __init__(self, obj: CellRecord, **hints: Any) -> None:
        self._record = obj

    @property
    def basis(self) -> SurdVector:
        """Return the stored cell vectors.

        :return: The scaled lattice vectors.
        """
        return _basis_vector(self._record.basis)

    @property
    def scale(self) -> SurdScalar:
        """Return the unit scale factor.

        :return: The factor applied to ``unscaled_basis``.
        """
        return SurdVector.one()

    @property
    def unscaled_basis(self) -> SurdVector:
        """Return the stored basis before scaling.

        :return: The cell vectors.
        """
        return self.basis

    @property
    def precision(self) -> fractions.Fraction | None:
        """Return the stored basis precision.

        :return: The absolute precision, or ``None`` when unknown.
        """
        return self._record.precision

    @property
    def periodicity(self) -> tuple[bool, bool, bool]:
        """Return the stored periodicity flags.

        :return: Flags identifying the periodic basis rows.
        """
        return self._record.periodicity  # type: ignore[return-value]

    def unwrap(self) -> CellRecord:
        """Return the stored cell record.

        :return: The source record.
        """
        return self._record
