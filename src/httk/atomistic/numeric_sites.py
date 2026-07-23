"""
The NumericSites presentation: a Sites object exposed as plain numpy numbers.
"""

from collections.abc import Iterator

from httk.core import NumericVector, to_numeric

from ._vector_guards import require_numpy
from .sites import Sites
from .sites_class_view import SitesClassView
from .sites_like import SitesLike


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
    """

    _sites: Sites

    def __init__(self, sites: SitesLike) -> None:
        require_numpy()
        self._sites = sites if isinstance(sites, Sites) else SitesClassView(sites)

    @property
    def reduced_coords(self) -> NumericVector:
        """The Nx3 reduced site coordinates as a ``float64`` numpy array."""
        return to_numeric(self._sites.reduced_coords)

    def __len__(self) -> int:
        return len(self._sites)

    def __iter__(self) -> Iterator[NumericVector]:
        for row in self._sites:
            yield to_numeric(row)

    def __getitem__(self, index: int) -> NumericVector:
        return to_numeric(self._sites[index])

    @property
    def exact(self) -> Sites:
        """The exact :class:`~httk.atomistic.Sites` this presentation wraps."""
        return self._sites

    def __repr__(self) -> str:
        return f"NumericSites(reduced_coords={self.reduced_coords!r})"
