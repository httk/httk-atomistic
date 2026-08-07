"""
The minimal canonical sites interface for httk-atomistic.
"""

import fractions
from abc import ABC, abstractmethod

from httk.core import FracVector


class SitesAPI(ABC):
    """
    Abstract base class for the canonical sites interface.

    It declares the single ``reduced_coords`` accessor (the exact Nx3 rational
    :class:`~httk.core.FracVector` of reduced coordinates) that every sites backend produces from
    its own native representation and every sites view builds its presentation from. This is the
    single interchange format; there is no pairwise conversion between backends.
    """

    @property
    @abstractmethod
    def reduced_coords(self) -> FracVector:
        """Return the reduced site coordinates."""
        raise NotImplementedError

    @property
    def precision(self) -> fractions.Fraction | None:
        """How precisely the coordinates were stated, in fractional units, or ``None``.

        Fractional, not a length: reduced coordinates are dimensionless and a ``Sites`` has
        no cell to convert with. :meth:`~httk.atomistic.UnitcellStructure.cartesian_precision`
        does the conversion, where the cell is known.

        Concrete rather than abstract, so a backend with no source of precision inherits
        ``None`` instead of breaking.

        :return: The fractional precision, or ``None`` when unknown.
        """
        return None
