"""
A view presenting any cell backend as a raw 3-row tuple of cell vectors.
"""

from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models._vector_guards import to_float_tuples
from httk.atomistic.models.cell.backend import CellBackend
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.cell.view_base import CellViewBase


class PlainCellView(CellViewBase, tuple):
    r"""
    A view presenting an underlying cell backend as the raw 3x3 basis matrix of floats.

    This view is a genuine tuple of three cell-vector rows (the scaled lattice vectors rendered to
    floats from the exact ``basis``), built eagerly and immutable.

    :param obj: The cell-like object to present.
    :param \**hints: Backend-selection hints.
    """

    _backend: CellBackend

    def __new__(cls, obj: CellLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        instance = super().__new__(cls, to_float_tuples(backend.basis))
        instance._backend = backend
        return instance

    def __init__(self, obj: CellLike, **hints: Any) -> None:
        super().__init__()

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> tuple[tuple[float, ...], ...]:
        """Return the presented basis as a plain tuple.

        :return: The three cell-vector rows.
        """
        # The view IS its presentation tuple; shed to a plain tuple (rows shared).
        return tuple(self)
