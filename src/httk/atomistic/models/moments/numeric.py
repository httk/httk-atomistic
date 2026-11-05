"""
The NumericSiteMoments value: per-site magnetic moments owned as plain floats.
"""

from typing import Any

from httk.core import NumericVector

from httk.atomistic.models._vector_guards import require_numpy
from httk.atomistic.models.cell.numeric import FloatRows, finite_float, float_precision, float_rows
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.moments.collinear import CollinearSiteMoments


class NumericSiteMoments:
    """
    Per-site magnetic moments owned as plain floats, in Bohr magnetons, an immutable value.

    A 1-D input (including an empty one) holds signed collinear scalars, kind ``"collinear"``;
    an N x 3 input (including a ``(0, 3)`` array) holds Cartesian vectors, kind
    ``"cartesian"``. Constructing one from an exact site-moments value is the explicit lossy
    step: collinear moments stay collinear, while Cartesian and crystal-axis moments become
    Cartesian floats; the source's precision is kept unless ``precision`` is given.

    Values are canonical: ``-0.0`` becomes ``0.0`` and non-finite numbers are rejected.
    Equality and hashing use the kind and values; the precision does not take part.
    Construction **requires numpy** and raises :class:`ImportError` eagerly when it is
    unavailable.

    :param moments: A 1-D (collinear) or N x 3 (Cartesian) real array-like, or site moments to
        convert.
    :param precision: How precisely the moments were stated, if known.
    :raises ValueError: If the moments are neither 1-D nor N x 3, or hold a non-finite value.
    """

    _kind: str
    _moments: tuple[float, ...] | FloatRows
    _precision: float | None

    def __init__(self, moments: Any, *, precision: Any = None) -> None:
        require_numpy()
        import numpy

        if isinstance(moments, NumericSiteMoments):
            precision = moments.precision if precision is None else precision
            self._kind, self._moments = moments._kind, moments._moments
        else:
            if isinstance(moments, SiteMomentsBackend):
                precision = moments.precision if precision is None else precision
                if isinstance(moments, CollinearSiteMoments):
                    array = numpy.asarray(moments.collinear_moments.to_floats(), dtype=float)
                else:  # Cartesian and crystal-axis moments both present Cartesian vectors
                    array = numpy.asarray(moments.cartesian_moments.to_floats(), dtype=float).reshape(-1, 3)
            else:
                array = numpy.asarray(moments, dtype=float)
            if array.ndim == 1:
                self._kind = "collinear"
                self._moments = tuple(finite_float(x, "NumericSiteMoments moments") for x in array.tolist())
            elif array.ndim == 2 and array.shape[1] == 3:
                self._kind = "cartesian"
                self._moments = float_rows(array, "NumericSiteMoments moments")
            else:
                raise ValueError(f"NumericSiteMoments moments must be 1-D or Nx3, got shape {array.shape}")
        self._precision = float_precision(precision)

    @property
    def kind(self) -> str:
        """Which representation this is, ``"collinear"`` or ``"cartesian"``.

        :return: The moments kind.
        """
        return self._kind

    @property
    def moments(self) -> NumericVector:
        """The moments as stored, as N scalars (collinear) or N x 3 vectors (Cartesian).

        :return: The moments as a read-only float array.
        """
        import numpy

        array = numpy.array(self._moments, dtype=float)
        if self._kind == "cartesian":
            array = array.reshape(-1, 3)
        array.flags.writeable = False
        return array

    @property
    def cartesian_moments(self) -> NumericVector:
        """The N x 3 Cartesian moments.

        :return: The moments as a read-only float array.
        :raises ValueError: If the moments are collinear, which carry no axis.
        """
        if self._kind == "collinear":
            raise ValueError(
                "a collinear scalar carries no axis and none is fabricated; "
                "construct a NumericSiteMoments from N x 3 rows explicitly to assign one"
            )
        return self.moments

    @property
    def precision(self) -> float | None:
        """How precisely the moments were stated, or ``None`` if unknown.

        :return: The precision as a floating-point value.
        """
        return self._precision

    def __len__(self) -> int:
        """Return the number of sites.

        :return: The number of moments.
        """
        return len(self._moments)

    def _key(self) -> tuple[Any, ...]:
        return (self._kind, self._moments)

    def __eq__(self, other: object) -> bool:
        """Equality of the kind and values; the precision does not take part.

        :param other: The object to compare with.
        :return: Whether the float values match.
        """
        if not isinstance(other, NumericSiteMoments):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        precision = "" if self._precision is None else f", precision={self._precision!r}"
        if self._kind == "cartesian" and not self._moments:
            return f"NumericSiteMoments(numpy.empty((0, 3)){precision})"
        values: Any = self._moments
        values = list(values) if self._kind == "collinear" else [list(row) for row in values]
        return f"NumericSiteMoments({values!r}{precision})"
