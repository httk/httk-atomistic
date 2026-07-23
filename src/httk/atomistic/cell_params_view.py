"""
A view presenting any cell backend as cell parameters (a, b, c, alpha, beta, gamma).
"""

from typing import Any, Self

from httk.core import unwrap

from .cell import Cell
from .cell_backend import CellBackend
from .cell_like import CellLike
from .cell_view import CellView


class CellParamsView(CellView, tuple):
    """
    A view presenting an underlying cell backend as cell parameters.

    This view is a genuine flat 6-tuple ``(a, b, c, alpha, beta, gamma)`` with the
    angles in degrees, built eagerly and immutable, with the elements also available
    as the named properties ``a``/``b``/``c``/``alpha``/``beta``/``gamma``.
    Parameters carry no orientation, so converting a cell to parameters is lossy:
    reconstructing a cell from this view reproduces the lengths, angles, and volume,
    but not the original cell-vector orientation.
    """

    _backend: CellBackend

    def __new__(cls, obj: CellLike, **hints: Any) -> Self:
        if isinstance(obj, cls):
            return obj
        backend = cls._prepare_backend(obj, hints)
        params = getattr(backend, "params", None)
        if params is not None:
            float_params = tuple(float(x) for x in params)
        else:
            reference = Cell(backend.unscaled_matrix, backend.scale)
            float_params = tuple(length.to_float() for length in reference.lengths) + tuple(
                float(angle) for angle in reference.angles
            )
        instance = super().__new__(cls, float_params)
        instance._backend = backend
        return instance

    def __init__(self, obj: CellLike, **hints: Any) -> None:
        super().__init__()

    @property
    def a(self) -> float:
        """The length of the first cell vector."""
        return self[0]

    @property
    def b(self) -> float:
        """The length of the second cell vector."""
        return self[1]

    @property
    def c(self) -> float:
        """The length of the third cell vector."""
        return self[2]

    @property
    def alpha(self) -> float:
        """The angle between the second and third cell vectors, in degrees."""
        return self[3]

    @property
    def beta(self) -> float:
        """The angle between the first and third cell vectors, in degrees."""
        return self[4]

    @property
    def gamma(self) -> float:
        """The angle between the first and second cell vectors, in degrees."""
        return self[5]

    def unwrap(self) -> Any:
        return unwrap(self._backend)
