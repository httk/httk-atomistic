"""
The NumericSites value: reduced site coordinates owned as plain floats.
"""

from collections.abc import Iterator
from typing import Any

from httk.core import NumericVector

from httk.atomistic.models._vector_guards import require_numpy
from httk.atomistic.models.cell.numeric import FloatRows, float_precision, float_rows, readonly_array
from httk.atomistic.models.sites.api import SitesAPI


class NumericSites:
    """
    Reduced site coordinates owned as plain floats, an immutable value.

    Where a :class:`~httk.atomistic.Sites` holds its reduced coordinates exactly, a
    ``NumericSites`` holds the N x 3 coordinates as Python floats and returns read-only ``float64``
    numpy arrays. It is len/iter/indexable over its rows, each a read-only ``(3,)`` array.
    Constructing one from a ``Sites`` (or any sites backend) is the explicit lossy step.

    Values are canonical: ``-0.0`` becomes ``0.0`` and non-finite numbers are rejected.
    Construction **requires numpy** (the ``httk-atomistic[numpy]`` extra) and raises
    :class:`ImportError` eagerly when it is unavailable.

    :param reduced_coords: The reduced coordinates as an N x 3 real array-like, or sites to convert.
    :param precision: The fractional coordinate precision, if known (default taken from converted
        sites).
    :raises ValueError: If the coordinates are not N x 3 or hold a non-finite value.
    """

    _reduced_coords: FloatRows
    _precision: float | None

    def __init__(self, reduced_coords: Any, *, precision: Any = None) -> None:
        require_numpy()
        if isinstance(reduced_coords, NumericSites):
            precision = reduced_coords.precision if precision is None else precision
            reduced_coords = reduced_coords._reduced_coords
        elif isinstance(reduced_coords, SitesAPI):
            precision = reduced_coords.precision if precision is None else precision
            reduced_coords = reduced_coords.reduced_coords_floats()
        self._reduced_coords = float_rows(reduced_coords, "NumericSites reduced_coords")
        self._precision = float_precision(precision)

    @property
    def precision(self) -> float | None:
        """The fractional coordinate precision, or ``None`` if unknown.

        :return: The precision as a floating-point value.
        """
        return self._precision

    @property
    def reduced_coords(self) -> NumericVector:
        """The N x 3 reduced site coordinates.

        :return: The coordinates as a read-only float array.
        """
        import numpy

        array = numpy.array(self._reduced_coords, dtype=float).reshape(-1, 3)
        array.flags.writeable = False
        return array

    def __len__(self) -> int:
        """Return the number of sites.

        :return: The number of coordinate rows.
        """
        return len(self._reduced_coords)

    def __iter__(self) -> Iterator[NumericVector]:
        """Iterate over the reduced-coordinate rows.

        :yield: Each coordinate row as a read-only array.
        """
        for row in self._reduced_coords:
            yield readonly_array(row)

    def __getitem__(self, index: int) -> NumericVector:
        """Return one reduced-coordinate row.

        :param index: The row index.
        :return: The selected coordinate row as a read-only array.
        """
        return readonly_array(self._reduced_coords[index])

    def _key(self) -> tuple[Any, ...]:
        return (self._reduced_coords, self._precision)

    def __eq__(self, other: object) -> bool:
        """Equality of the coordinates and the precision.

        :param other: The object to compare with.
        :return: Whether the float values match.
        """
        if not isinstance(other, NumericSites):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        precision = "" if self._precision is None else f", precision={self._precision!r}"
        return f"NumericSites({[list(row) for row in self._reduced_coords]!r}{precision})"
