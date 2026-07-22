"""
The minimal canonical species interface for httk-atomistic.
"""

from abc import ABC, abstractmethod


class SpeciesAPI(ABC):
    """
    Abstract base class for the canonical single-species interface.

    It declares the accessors mirroring the OPTIMADE ``species`` fields that every
    species backend produces from its own native representation and every species view
    builds its presentation from: ``name``, ``chemical_symbols``, ``concentration``,
    and the optional ``mass``, ``attached``, ``nattached``, and ``original_name``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def chemical_symbols(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def concentration(self) -> tuple[float, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def mass(self) -> tuple[float, ...] | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def attached(self) -> tuple[str, ...] | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def nattached(self) -> tuple[int, ...] | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def original_name(self) -> str | None:
        raise NotImplementedError
