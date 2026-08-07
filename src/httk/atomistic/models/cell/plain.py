"""
Backend wrapping a raw 3x3 cell basis (the matrix of cell vectors).
"""

from typing import Any

from httk.core import SurdScalar, SurdVector

from httk.atomistic.models._vector_guards import is_basis_3x3, to_surdvector
from httk.atomistic.models.cell.backend import CellBackend


class PlainCell(CellBackend):
    r"""
    Backend for a cell backed by a raw 3x3 list or tuple of numbers (or any 3x3 vector-like).

    The native representation is preserved verbatim (one cell vector per row); the exact
    :class:`~httk.core.SurdVector` ``basis`` is built lazily and cached. This representation carries
    no separate length factor, so ``scale`` is the exact ``1`` and ``unscaled_basis == basis``.
    ``unwrap`` returns the original raw object.

    :param obj: The raw 3x3 basis representation.
    :param \**hints: Backend-selection hints.
    """

    _raw: Any
    _basis_cache: SurdVector | None

    # Cannot type annotate __new__ as `Self | None` for some reason
    def __new__(cls, obj: Any, **hints: Any) -> Any:
        if hints and hints.get("kind", "plain") != "plain":
            return None
        if not is_basis_3x3(obj):
            return None
        return super().__new__(cls)

    def __init__(self, obj: Any, **hints: Any) -> None:
        self._raw = obj
        self._basis_cache = None

    @property
    def basis(self) -> SurdVector:
        """Return the raw basis in the canonical representation.

        :return: The cell vectors.
        """
        if self._basis_cache is None:
            self._basis_cache = to_surdvector(self._raw)
        return self._basis_cache

    @property
    def scale(self) -> SurdScalar:
        """Return the unit scale factor.

        :return: The factor applied to ``unscaled_basis``.
        """
        return SurdVector.one()

    @property
    def unscaled_basis(self) -> SurdVector:
        """Return the basis before scaling.

        :return: The cell vectors.
        """
        return self.basis

    def unwrap(self) -> Any:
        """Return the original basis object.

        :return: The raw basis representation.
        """
        return self._raw
