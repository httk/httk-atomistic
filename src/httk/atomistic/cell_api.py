"""
The minimal canonical cell interface for httk-atomistic.
"""

from abc import ABC, abstractmethod

from httk.core import SurdScalar, SurdVector


class CellAPI(ABC):
    """
    Abstract base class for the canonical cell interface.

    It declares the exact accessors that every cell backend produces from its own native
    representation and every cell view builds its presentation from: the ``matrix`` of 3x3 lattice
    vectors (``scale * unscaled_matrix``), the positive ``scale``, and the ``unscaled_matrix``. All
    three are exact httk-core vectors; this is the single interchange format, with no pairwise
    conversion between backends.
    """

    @property
    @abstractmethod
    def matrix(self) -> SurdVector:
        raise NotImplementedError

    @property
    @abstractmethod
    def scale(self) -> SurdScalar:
        raise NotImplementedError

    @property
    @abstractmethod
    def unscaled_matrix(self) -> SurdVector:
        raise NotImplementedError
