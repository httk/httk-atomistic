"""
The Sites class for httk-atomistic.
"""

import fractions
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from httk.core import FracVector, VectorLike

from httk.atomistic.models._vector_guards import to_fracvector, to_precision
from httk.atomistic.models.sites.backend import SitesBackend

if TYPE_CHECKING:
    from httk.atomistic.models.sites.numeric import NumericSites


class Sites(SitesBackend):
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
    :meth:`~httk.atomistic.UnitcellStructure.cartesian_sites`.

    :param reduced_coords: The reduced coordinates, one site per row.
    :param precision: The fractional precision carried from the source, if known.
    """

    _reduced_coords: FracVector
    _precision: fractions.Fraction | None

    def __init__(self, reduced_coords: VectorLike, precision: Any = None) -> None:
        coords = to_fracvector(reduced_coords)
        if coords.dim != () and not (len(coords.dim) == 2 and coords.dim[1] == 3):
            raise ValueError("Sites reduced_coords must be an Nx3 vector-like")
        self._reduced_coords = coords
        self._precision = to_precision(precision)

    @property
    def reduced_coords(self) -> FracVector:
        """The Nx3 reduced site coordinates as an exact ``FracVector`` (one site per row)."""
        return self._reduced_coords

    def __len__(self) -> int:
        """Return the number of sites.

        :return: The number of coordinate rows.
        """
        return len(self._reduced_coords)

    def __iter__(self) -> Iterator[FracVector]:
        """Iterate over the reduced-coordinate rows.

        :return: An iterator over the coordinate rows.
        """
        return iter(self._reduced_coords)

    def __getitem__(self, index: int) -> FracVector:
        """Return one reduced-coordinate row.

        :param index: The row index.
        :return: The selected coordinate row.
        """
        return self._reduced_coords[index]

    @property
    def precision(self) -> fractions.Fraction | None:
        """How precisely these coordinates were stated, in fractional units, or ``None``.

        Fractional and therefore dimensionless: reduced coordinates are fractions of a cell
        edge, and a ``Sites`` carries no cell to convert with. Use
        :meth:`~httk.atomistic.UnitcellStructure.cartesian_precision` for the corresponding length,
        which is the number an interatomic tolerance or an spglib ``symprec`` actually
        wants.

        It is the *coarsest* precision among the coordinates, since a structure is only as
        precisely stated as its least precisely stated number. ``None`` means unknown.

        :return: The fractional precision, or ``None`` when unknown.
        """
        return self._precision

    def numeric(self) -> "NumericSites":
        """Return a plain-numpy presentation of these sites.

        :return: The numpy-backed presentation.
        :raises ImportError: If numpy is unavailable.
        """
        from httk.atomistic.models.sites.numeric import NumericSites

        return NumericSites(self)

    def __eq__(self, other: object) -> bool:
        """Equality of the coordinates, and of nothing else.

        The stated ``precision`` does not take part: the same coordinates recorded from a
        more carefully written file are still the same coordinates.

        :param other: The object to compare with.
        :return: Whether the coordinate values match.
        """
        if not isinstance(other, Sites):
            return NotImplemented
        return self._reduced_coords == other._reduced_coords

    def __repr__(self) -> str:
        return f"Sites(reduced_coords={self._reduced_coords!r})"
