"""
The minimal canonical cell interface for httk-atomistic.
"""

from abc import ABC, abstractmethod


class CellAPI(ABC):
    """
    Abstract base class for the canonical cell interface.

    It declares the single ``matrix`` accessor (the 3x3 cell vectors) that every cell
    backend produces from its own native representation and every cell view builds its
    presentation from. This is the single interchange format; there is no pairwise
    conversion between backends.
    """

    @property
    @abstractmethod
    def matrix(self) -> tuple[tuple[float, ...], ...]:
        raise NotImplementedError
