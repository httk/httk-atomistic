"""
A view presenting any cell backend as cell parameters (a, b, c, alpha, beta, gamma).
"""

from typing import Any, Self

from httk.core import unwrap

from httk.atomistic.models.cell.backend import CellBackend
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.cell.view_base import CellViewBase


class CellParamsView(CellViewBase, tuple):
    r"""
    A view presenting an underlying cell backend as cell parameters.

    This view is a genuine flat 6-tuple ``(a, b, c, alpha, beta, gamma)`` with the
    angles in degrees, built eagerly and immutable, with the elements also available
    as the named properties ``a``/``b``/``c``/``alpha``/``beta``/``gamma``.
    Parameters carry no orientation, so converting a cell to parameters is lossy:
    reconstructing a cell from this view reproduces the lengths and angles, and reproduces volume
    only for a fully periodic source. The reconstruction inherits the fully periodic default, so
    this view discards the source periodicity as well as the original cell-vector orientation.

    :param obj: The cell-like object to present.
    :param \**hints: Backend-selection hints.
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
            reference = Cell(backend.unscaled_basis, backend.scale, backend.precision)
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
        """The length of the first cell vector.

        :return: The first vector length.
        """
        return self[0]

    @property
    def b(self) -> float:
        """The length of the second cell vector.

        :return: The second vector length.
        """
        return self[1]

    @property
    def c(self) -> float:
        """The length of the third cell vector.

        :return: The third vector length.
        """
        return self[2]

    @property
    def alpha(self) -> float:
        """The angle between the second and third cell vectors, in degrees.

        :return: The alpha angle.
        """
        return self[3]

    @property
    def beta(self) -> float:
        """The angle between the first and third cell vectors, in degrees.

        :return: The beta angle.
        """
        return self[4]

    @property
    def gamma(self) -> float:
        """The angle between the first and second cell vectors, in degrees.

        :return: The gamma angle.
        """
        return self[5]

    def unwrap(self) -> Any:
        """Return the raw object behind the backend.

        :return: The unwrapped source object.
        """
        return unwrap(self._backend)

    def unview(self) -> tuple[float, ...]:
        """Return the presented parameters as a plain tuple.

        :return: The six cell parameters.
        """
        # The view IS its presentation 6-tuple; shed to a plain tuple.
        return tuple(self)
