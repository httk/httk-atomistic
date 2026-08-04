"""
The minimal canonical site-moments interface for httk-atomistic.
"""

import fractions
from abc import ABC, abstractmethod

from httk.core import SurdVector


class SiteMomentsAPI(ABC):
    """
    Abstract base class for the canonical per-site magnetic-moments interface.

    ``cartesian_moments`` is the exact Nx3 Cartesian interchange format, in Bohr magnetons.
    A collinear representation has no axis and therefore raises from that accessor instead of
    fabricating one.
    """

    @property
    @abstractmethod
    def cartesian_moments(self) -> SurdVector:
        raise NotImplementedError

    @property
    def precision(self) -> fractions.Fraction | None:
        """How precisely the moments were stated, in Bohr magnetons, or ``None``.

        Concrete rather than abstract, so a backend with no source of precision inherits
        ``None`` instead of breaking.
        """
        return None

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
