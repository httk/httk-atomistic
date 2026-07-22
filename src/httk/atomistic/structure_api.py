"""
The minimal canonical structure interface for httk-atomistic.
"""

from abc import ABC, abstractmethod

from .cell import Cell
from .sites import Sites
from .species import Species


class StructureAPI(ABC):
    """
    Abstract base class for the canonical structure interface.

    It declares the Simple quartet that every structure backend produces from its
    own native representation and every structure view builds its presentation
    from: ``cell``, ``sites``, ``species``, and ``species_at_sites``. This is the
    single interchange format; there is no pairwise conversion between backends.
    """

    @property
    @abstractmethod
    def cell(self) -> Cell:
        raise NotImplementedError

    @property
    @abstractmethod
    def sites(self) -> Sites:
        raise NotImplementedError

    @property
    @abstractmethod
    def species(self) -> tuple[Species, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def species_at_sites(self) -> tuple[str, ...]:
        raise NotImplementedError
