"""
The Sites class for httk-atomistic.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

from httk.core import FracVector, VectorLike

from ._vector_guards import to_fracvector

if TYPE_CHECKING:
    from .numeric_sites import NumericSites


class Sites:
    """
    The sites of a crystal structure: the Nx3 matrix of reduced coordinates, held **exactly**.

    Reduced (fractional) coordinates are the symmetry-native frame: point-group operations are
    integer matrices and translations are rationals, so no radicals ever appear. They are therefore
    stored as an exact rational :class:`~httk.core.FracVector` of shape ``(N, 3)``. A Sites object
    is iterable and indexable over its length-3 coordinate rows (each a ``FracVector``), with
    ``len`` giving the number of sites.

    Inputs embed exactly: rationals (and rational-valued floats), rational strings, and numpy arrays
    all land on their exact rational value. An irrational :class:`~httk.core.SurdVector` input is
    reduced deterministically through the vector family's ``fractions`` hub (never raising on data);
    the exact Cartesian frame — where radicals belong — is obtained instead via
    :meth:`~httk.atomistic.Structure.cartesian_sites`.
    """

    _reduced_coords: FracVector

    def __init__(self, reduced_coords: VectorLike) -> None:
        coords = to_fracvector(reduced_coords)
        if coords.dim != () and not (len(coords.dim) == 2 and coords.dim[1] == 3):
            raise ValueError("Sites reduced_coords must be an Nx3 vector-like")
        self._reduced_coords = coords

    @property
    def reduced_coords(self) -> FracVector:
        """The Nx3 reduced site coordinates as an exact ``FracVector`` (one site per row)."""
        return self._reduced_coords

    def __len__(self) -> int:
        return len(self._reduced_coords)

    def __iter__(self) -> Iterator[FracVector]:
        return iter(self._reduced_coords)

    def __getitem__(self, index: int) -> FracVector:
        return self._reduced_coords[index]

    def numeric(self) -> "NumericSites":
        """A plain-numpy presentation of these sites (requires the ``httk-atomistic[numpy]`` extra)."""
        from .numeric_sites import NumericSites

        return NumericSites(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sites):
            return NotImplemented
        return self._reduced_coords == other._reduced_coords

    def __repr__(self) -> str:
        return f"Sites(reduced_coords={self._reduced_coords!r})"
