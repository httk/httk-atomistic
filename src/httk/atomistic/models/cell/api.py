"""
The minimal canonical cell interface for httk-atomistic.
"""

import fractions
from abc import ABC, abstractmethod

from httk.core import SurdScalar, SurdVector, exactmath
from httk.core.exactmath import integer_sqrt

_FALLBACK_PREC = exactmath.default_accuracy
_MAX_EXACT_RADICAND = 10**18


def _scalar_length(lsq: SurdScalar) -> SurdScalar:
    """Return a squared length's exact scalar root, with a deterministic fallback."""
    if lsq.is_rational:
        q = lsq._rational_fraction()
        num, den = q.numerator, q.denominator
        root_num, root_den = integer_sqrt(num), integer_sqrt(den)
        if root_num * root_num == num and root_den * root_den == den:
            return SurdVector(fractions.Fraction(root_num, root_den))._as_scalar()
        if num * den <= _MAX_EXACT_RADICAND:
            return SurdVector.sqrt_of(q)
        return SurdVector(exactmath.sqrt(q, prec=_FALLBACK_PREC, limit=True))._as_scalar()
    approx = lsq.to_fractions_approx(_FALLBACK_PREC)
    return SurdVector(exactmath.sqrt(approx, prec=_FALLBACK_PREC, limit=True))._as_scalar()


def _angle_from_gram(gram: SurdVector, i: int, j: int) -> fractions.Fraction:
    """Return one exact crystallographic angle from a Gram matrix."""
    dot = gram._element((i, j))
    li = _scalar_length(gram._element((i, i)))
    lj = _scalar_length(gram._element((j, j)))
    cosine = (dot * (li * lj)._as_scalar()._inverse())._as_scalar()
    try:
        exact = cosine.acos_degrees()
    except ValueError:
        exact = None
    if exact is not None:
        return exact
    cos_value = max(
        fractions.Fraction(-1),
        min(fractions.Fraction(1), cosine.to_fractions_approx(_FALLBACK_PREC)),
    )
    return fractions.Fraction(exactmath.acos(cos_value, degrees=True, prec=_FALLBACK_PREC, limit=False))


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

    def metric(self) -> SurdVector:
        """Return the exact Gram matrix of the scaled basis.

        :return: The Gram matrix of the cell vectors.
        """
        basis = self.basis
        return basis * basis.T()

    @property
    def lengths(self) -> tuple[SurdScalar, ...]:
        """Return the exact lengths of the three scaled basis vectors.

        :return: The three cell-vector lengths.
        """
        metric = self.metric()
        return tuple(_scalar_length(metric._element((i, i))) for i in range(3))

    @property
    def angles(self) -> tuple[fractions.Fraction, ...]:
        """Return ``(alpha, beta, gamma)`` in exact degrees.

        :return: The crystallographic angles in degrees.
        """
        gram = self.unscaled_basis * self.unscaled_basis.T()
        return (
            _angle_from_gram(gram, 1, 2),
            _angle_from_gram(gram, 0, 2),
            _angle_from_gram(gram, 0, 1),
        )

    @property
    def volume(self) -> SurdScalar:
        """Return the exact absolute determinant of the basis.

        :return: The cell volume.
        :raises ValueError: If the cell is not periodic in all three directions.
        """
        if self.periodicity != (True, True, True):
            raise ValueError(
                "volume is defined only for a fully 3D-periodic cell; this one is periodic in "
                f"{sum(self.periodicity)} of 3 directions ({self.periodicity}). "
                "Use periodic_measure for the area, length or volume of the periodic sublattice."
            )
        determinant = self.basis.det()
        return (-determinant)._as_scalar() if determinant.sign() < 0 else determinant
