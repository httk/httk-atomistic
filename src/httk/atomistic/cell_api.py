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
        raise NotImplementedError

    @property
    @abstractmethod
    def scale(self) -> SurdScalar:
        raise NotImplementedError

    @property
    @abstractmethod
    def unscaled_basis(self) -> SurdVector:
        raise NotImplementedError

    @property
    def precision(self) -> fractions.Fraction | None:
        """How precisely the basis was stated, as an absolute length, or ``None`` if unknown.

        Deliberately concrete rather than abstract. A backend that knows its source's
        precision overrides this; one that does not — a bare matrix of numbers with no
        provenance — inherits ``None``, which is the honest answer and keeps every existing
        backend, in this package or outside it, working unchanged.
        """
        return None

    @property
    def periodicity(self) -> tuple[bool, bool, bool]:
        """Which of the three basis rows is a genuine lattice translation.

        Concrete rather than abstract, for the same reason as :attr:`precision`: a backend
        that knows its periodicity overrides this, and one that does not inherits
        ``(True, True, True)``. That default is not a guess — a cell described by six
        lattice parameters, or by a bare matrix with no further provenance, is a crystal,
        which is what every cell in httk was before periodicity was recorded at all.
        """
        return (True, True, True)
