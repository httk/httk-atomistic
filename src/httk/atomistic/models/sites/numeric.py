"""
The NumericSites presentation: a Sites object exposed as plain numpy numbers.
"""

from collections.abc import Iterator

from httk.core import NumericVector, to_numeric

from httk.atomistic.models._vector_guards import require_numpy
from httk.atomistic.models.sites.like import SitesLike
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.sites.view import SitesView


class NumericSites:
    """
    A plain-numpy presentation of a :class:`~httk.atomistic.Sites` object.

    Where a ``Sites`` holds its reduced coordinates exactly (an Nx3 rational
    :class:`~httk.core.FracVector`), a ``NumericSites`` mirrors that interface but returns plain
    ``float64`` numpy arrays, for callers who do not need exact arithmetic and just want numpy arrays.
    It is len/iter/indexable over its rows, each yielded as a ``(3,)`` numpy array.

    The presentation is numpy-backed, so constructing a ``NumericSites`` **requires numpy** (the
    ``httk-atomistic[numpy]`` extra) and raises :class:`ImportError` eagerly when it is unavailable.
    The exact object is always one hop away via :attr:`exact`.

    :param sites: The sites or sites-like object to present.
    """

    _sites: Sites

    def __init__(self, sites: SitesLike) -> None:
        require_numpy()
        self._sites = sites if isinstance(sites, Sites) else SitesView(sites)

    @property
    def precision(self) -> float | None:
        """The fractional coordinate precision, or ``None`` if unknown.

        :return: The precision as a floating-point value.
        """
        return None if self._sites.precision is None else float(self._sites.precision)

    @property
    def reduced_coords(self) -> NumericVector:
        """The Nx3 reduced site coordinates.

        :return: The coordinates as floating-point values.
        """
        return to_numeric(self._sites.reduced_coords)

    def __len__(self) -> int:
        """Return the number of sites.

        :return: The number of coordinate rows.
        """
        return len(self._sites)

    def __iter__(self) -> Iterator[NumericVector]:
        """Iterate over the reduced-coordinate rows.

        :yield: Each coordinate row.
        """
        for row in self._sites:
            yield to_numeric(row)

    def __getitem__(self, index: int) -> NumericVector:
        """Return one reduced-coordinate row.

        :param index: The row index.
        :return: The selected coordinate row.
        """
        return to_numeric(self._sites[index])

    @property
    def exact(self) -> Sites:
        """The exact sites this presentation wraps.

        :return: The exact sites.
        """
        return self._sites

    def __repr__(self) -> str:
        return f"NumericSites(reduced_coords={self.reduced_coords!r})"
