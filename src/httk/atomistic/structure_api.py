"""
The minimal canonical structure interface for httk-atomistic.
"""

from abc import ABC, abstractmethod

from .species import Species


class StructureAPI(ABC):
    """
    Abstract base class for the canonical structure interface.

    It declares the Simple quartet that every structure backend produces from its
    own native representation and every structure view builds its presentation
    from: ``basis``, ``sites``, ``species``, and ``species_at_sites``. This is the
    single interchange format; there is no pairwise conversion between backends.
    """

    @property
    @abstractmethod
    def basis(self) -> tuple[tuple[float, ...], ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def sites(self) -> tuple[tuple[float, ...], ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def species(self) -> tuple[Species, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def species_at_sites(self) -> tuple[str, ...]:
        raise NotImplementedError
