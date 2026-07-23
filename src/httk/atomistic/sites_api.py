"""
The minimal canonical sites interface for httk-atomistic.
"""

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
        raise NotImplementedError
