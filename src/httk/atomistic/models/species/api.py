"""
The minimal canonical species interface for httk-atomistic.
"""

from abc import ABC, abstractmethod
from fractions import Fraction


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
    def charges(self) -> tuple[Fraction | None, ...] | None:
        """Assigned charge numbers for the constituents, or ``None`` if unstated.

        A ``None`` element means the charge of that constituent is unstated; whole-
        ``None`` means no constituent charges are stated. Values use elementary-charge
        units, for example a formal oxidation state.

        :return: The constituent charges, or ``None`` when unstated.
        """
        return None

    @property
    def spins(self) -> tuple[Fraction | None, ...] | None:
        """Idealized signed spins assigned to the constituents, or ``None`` if unstated.

        A ``None`` element means the spin of that constituent is unstated; whole-``None``
        means no constituent spins are stated. This is distinct from a calculated site
        magnetic moment.

        :return: The constituent spins, or ``None`` when unstated.
        """
        return None

    @property
    def labels(self) -> tuple[str | None, ...] | None:
        """Free-form per-constituent labels, or ``None`` if unstated.

        A ``None`` element means that constituent has no stated label; whole-``None``
        means no constituent labels are stated.

        :return: The constituent labels, or ``None`` when unstated.
        """
        return None

    @property
    @abstractmethod
    def concentration(self) -> tuple[Fraction, ...]:
        raise NotImplementedError

    @property
    def is_ordered(self) -> bool:
        """Return whether every constituent has unit concentration.

        :return: ``True`` when all concentrations are exactly ``Fraction(1)``.
        """
        return all(value == Fraction(1) for value in self.concentration)

    @property
    @abstractmethod
    def concentration_precision(self) -> tuple[Fraction | None, ...] | None:
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
