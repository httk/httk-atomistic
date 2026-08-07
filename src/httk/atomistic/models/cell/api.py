"""
The minimal canonical cell interface for httk-atomistic.
"""

import fractions
from abc import ABC, abstractmethod

from httk.core import SurdScalar, SurdVector


class CellAPI(ABC):
    """
    Abstract base class for the canonical cell interface.

    It declares the exact accessors that every cell backend produces from its own native
    representation and every cell view builds its presentation from: the ``basis`` of 3x3 lattice
    vectors (``scale * unscaled_basis``), the positive ``scale``, and the ``unscaled_basis``. All
    three are exact httk-core vectors; this is the single interchange format, with no pairwise
    conversion between backends.
    """

    @property
    @abstractmethod
    def basis(self) -> SurdVector:
        """Return the scaled lattice vectors."""
        raise NotImplementedError

    @property
    @abstractmethod
    def scale(self) -> SurdScalar:
        """Return the positive factor applied to ``unscaled_basis``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def unscaled_basis(self) -> SurdVector:
        """Return the lattice vectors before applying ``scale``."""
        raise NotImplementedError

    @property
    def precision(self) -> fractions.Fraction | None:
        """How precisely the basis was stated, as an absolute length, or ``None`` if unknown.

        A backend that knows its source's precision overrides this; one that does not — a
        bare matrix of numbers with no provenance — inherits ``None``.

        :return: The absolute basis precision, or ``None`` when it is unknown.
        """
        return None

    @property
    def periodicity(self) -> tuple[bool, bool, bool]:
        """Which of the three basis rows is a genuine lattice translation.

        A backend that knows its periodicity overrides this; one that does not inherit
        ``(True, True, True)``. A cell described only by six lattice parameters or a bare
        matrix is interpreted as a fully periodic crystal.

        :return: Flags identifying the periodic basis rows.
        """
        return (True, True, True)
