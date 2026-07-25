"""Exact rational Wyckoff-position algebra.

A Wyckoff position is a family of symmetry-equivalent sites parameterised by a few free
coordinates: SG 15 letter ``e`` is ``0,y,1/4``, so one free parameter ``y`` places four
symmetry-equivalent atoms. This module goes both ways between free parameters and
coordinates, entirely over the rationals.

Two properties of the vendored tables make this much simpler than it looks, both asserted
in ``tests/test_symmetry_data.py``:

* Each Wyckoff position's ``orbit`` is already the **complete, deduplicated** list of
  affine maps, of length exactly ``multiplicity``, with centering translations folded in.
  Generating an orbit is therefore a plain loop with no coincidence testing and no
  tolerance.
* ``hasfreedom`` marks which of ``x``, ``y``, ``z`` are free, and the columns of every
  orbit matrix for the non-free variables are identically zero, with
  ``sum(hasfreedom) == rank``. So the free parameters are read straight off, and the
  ``first_orbit`` strings (``"1/8,y,-y+1/4"``) never need parsing — the same information
  is already present as an exact affine map.

Everything is expressed in the coordinates of whichever setting the record came from.
"""

import fractions
from typing import Any, Self, Sequence

from httk.core import FracVector

from .affine_operation import AffineOperation

__all__ = ["WyckoffBranch", "WyckoffPosition", "wyckoff_positions"]


class WyckoffBranch:
    """One member of a Wyckoff position's orbit: an affine map of the free parameters.

    A position of multiplicity *m* has *m* branches. The representative branch is the
    first, but a coordinate may lie on any of them, which is why
    :meth:`WyckoffPosition.parameters_of` tries them all.
    """

    _operation: AffineOperation
    _free: tuple[int, ...]
    _unimodular: FracVector
    _echelon: FracVector
    _pivot_inverse: FracVector

    def __init__(self, operation: AffineOperation, free: Sequence[int]) -> None:
        self._operation = operation
        self._free = tuple(free)
        self._unimodular, self._echelon = _row_hermite(operation.matrix, self._free)
        rank = len(self._free)
        if rank:
            pivot = [[self._echelon[row][column].to_fraction() for column in range(rank)] for row in range(rank)]
            self._pivot_inverse = FracVector.create(_invert_small(pivot))
        else:
            self._pivot_inverse = FracVector.create(())

    @property
    def operation(self) -> AffineOperation:
        """The affine map from a full ``(x, y, z)`` parameter triple to this branch's coordinate."""
        return self._operation

    @property
    def free(self) -> tuple[int, ...]:
        """Indices into ``(x, y, z)`` of the free parameters, ascending."""
        return self._free

    def coordinate(self, parameters: Any) -> FracVector:
        """The coordinate this branch places at the given free-parameter values.

        ``parameters`` has one entry per free parameter. Because the non-free columns of
        the matrix are zero, the values placed at the non-free positions are irrelevant.
        """
        placed = [fractions.Fraction(0)] * 3
        values = FracVector.create(parameters)
        if not self._free:
            # A fixed position has no parameters. FracVector has no zero-length shape, so
            # "no parameters" is the empty vector, whose dim reads as () or (0,).
            if values.dim not in ((), (0,)):
                raise ValueError(f"a fixed Wyckoff position takes no free parameters, got dim {values.dim}")
            return self._operation.apply(placed)
        if values.dim != (len(self._free),):
            raise ValueError(f"expected {len(self._free)} free parameter(s), got dim {values.dim}")
        for slot, index in enumerate(self._free):
            placed[index] = values[slot].to_fraction()
        return self._operation.apply(placed)

    def parameters_of(self, coordinate: Any) -> FracVector | None:
        """The free-parameter values putting this branch on ``coordinate``, or ``None``.

        ``None`` means the coordinate does not lie on this branch — for *any* lattice
        translation, not merely the one given. That completeness is what the row-Hermite
        form buys: with ``U`` unimodular over the integers, ``A t = d (mod Z^3)`` holds iff
        ``U A t = U d (mod Z^3)``, and the zero rows of ``U A`` turn the lattice-membership
        question into "are these components integers?" with no search over translations.

        The returned parameters are reduced into ``[0, 1)``, which is canonical: the pivot
        block has determinant ``±1`` throughout the vendored tables, so the solution is
        unique modulo one. The result is verified by re-evaluating the branch, so a table
        that ever violated that assumption would yield a clean miss rather than a wrong
        answer.
        """
        point = FracVector.create(coordinate)
        rank = len(self._free)
        difference = point - self._operation.vector
        mapped = difference * self._unimodular.T()

        # Rows below the rank carry no free parameter, so they must already be lattice
        # translations for the coordinate to lie on this branch at all.
        for row in range(rank, 3):
            if mapped[row].to_fraction().denominator != 1:
                return None

        if rank == 0:
            parameters = FracVector.create(())
        else:
            head = FracVector.create([mapped[row].to_fraction() for row in range(rank)])
            parameters = (head * self._pivot_inverse.T()).normalize()

        if self.coordinate(parameters).normalize() != point.normalize():
            return None
        return parameters


class WyckoffPosition:
    """A Wyckoff position of one space-group setting: a letter and its orbit."""

    _letter: str
    _multiplicity: int
    _site_symmetry: str
    _free: tuple[int, ...]
    _branches: tuple[WyckoffBranch, ...]

    def __init__(self, record: dict[str, Any]) -> None:
        self._letter = record["letter"]
        self._multiplicity = record["multiplicity"]
        self._site_symmetry = record["sitesym"]
        self._free = tuple(index for index, free in enumerate(record["hasfreedom"]) if free)
        self._branches = tuple(WyckoffBranch(AffineOperation.from_record(item), self._free) for item in record["orbit"])
        if len(self._branches) != self._multiplicity:
            raise ValueError(
                f"Wyckoff position {self._letter!r} has {len(self._branches)} orbit entries "
                f"but multiplicity {self._multiplicity}"
            )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Self:
        return cls(record)

    @property
    def letter(self) -> str:
        """The Wyckoff letter, e.g. ``"e"``. Bare, with no multiplicity prefix."""
        return self._letter

    @property
    def multiplicity(self) -> int:
        """How many sites one set of free-parameter values generates in the unit cell."""
        return self._multiplicity

    @property
    def site_symmetry(self) -> str:
        """The site-symmetry group in Hermann-Mauguin notation, e.g. ``"2"`` or ``"-1"``."""
        return self._site_symmetry

    @property
    def free(self) -> tuple[int, ...]:
        """Indices into ``(x, y, z)`` of the free parameters."""
        return self._free

    @property
    def free_count(self) -> int:
        """The number of degrees of freedom: 0 for a fixed position, 3 for the general one."""
        return len(self._free)

    @property
    def branches(self) -> tuple[WyckoffBranch, ...]:
        """The orbit, one branch per equivalent site. Complete and already deduplicated."""
        return self._branches

    @property
    def representative(self) -> WyckoffBranch:
        """The first orbit member, the one the tables print as ``first_orbit``."""
        return self._branches[0]

    def coordinates(self, parameters: Any) -> FracVector:
        """Every coordinate of the orbit, as an exact ``(multiplicity, 3)`` block.

        Not wrapped and not deduplicated: within one setting the tabulated orbit is
        already distinct, so wrapping is the caller's business (and matters only once a
        setting transform enters).
        """
        return FracVector.create([branch.coordinate(parameters).to_fractions() for branch in self._branches])

    def parameters_of(self, coordinate: Any) -> FracVector | None:
        """The free-parameter values placing *some* branch on ``coordinate``, or ``None``.

        Tries every branch, not only the representative. That matters: across the vendored
        tables, 11673 of the 20639 non-representative orbit members lie on a different
        branch than the representative, so a matcher that only tested ``first_orbit``
        would reject a majority of legitimate orbit points.
        """
        for branch in self._branches:
            parameters = branch.parameters_of(coordinate)
            if parameters is not None:
                return parameters
        return None

    def __repr__(self) -> str:
        return f"WyckoffPosition({self._multiplicity}{self._letter}, sitesym={self._site_symmetry!r})"


def wyckoff_positions(record: dict[str, Any]) -> tuple[WyckoffPosition, ...]:
    """The Wyckoff positions of a setting record, most specific first.

    Ordered by ``(free_count, multiplicity, letter)`` so that the first match found when
    identifying a coordinate is the most specific position it lies on. Ties do not arise:
    positions are affine subspaces, so a coordinate on two distinct positions of the same
    dimension also lies on their lower-dimensional intersection, which is covered by an
    earlier entry.
    """
    positions = [WyckoffPosition(entry) for entry in record["wyckoff"]]
    positions.sort(key=lambda position: (position.free_count, position.multiplicity, position.letter))
    return tuple(positions)


def _invert_small(rows: list[list[fractions.Fraction]]) -> list[list[fractions.Fraction]]:
    """The exact inverse of a small square rational matrix, by Gauss-Jordan elimination.

    :meth:`~httk.core.FracVector.inv` covers only scalars and 3x3 matrices, but the pivot
    blocks here are 1x1 and 2x2 as well (a Wyckoff position with one or two degrees of
    freedom). The sizes are at most 3, so the plain algorithm is entirely adequate.
    """
    size = len(rows)
    work = [list(row) + [fractions.Fraction(1 if i == j else 0) for j in range(size)] for i, row in enumerate(rows)]
    for column in range(size):
        pivot = next((r for r in range(column, size) if work[r][column] != 0), None)
        if pivot is None:
            raise ValueError("singular matrix: a Wyckoff pivot block must be invertible")
        work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(size):
            if row != column and work[row][column] != 0:
                factor = work[row][column]
                work[row] = [a - factor * b for a, b in zip(work[row], work[column])]
    return [row[size:] for row in work]


def _row_hermite(matrix: FracVector, free: Sequence[int]) -> tuple[FracVector, FracVector]:
    """Row-reduce the free columns of an orbit matrix to echelon form over the integers.

    Returns ``(U, H)`` with ``U`` unimodular (integer, determinant ``±1``) and
    ``H = U * A`` in row echelon form, where ``A`` is ``matrix`` restricted to the free
    columns. Unimodularity is the point: it preserves the integer lattice, so a
    congruence modulo ``Z^3`` may be applied to ``H`` and ``U d`` instead of ``A`` and
    ``d`` without changing which coordinates satisfy it.

    Uses gcd-based elimination (integer row reduction), so no division is ever inexact.
    Orbit matrices are integer with entries in ``{-2, ..., 2}``, so the values stay tiny.
    """
    rows = [[int(matrix[row][column].to_fraction()) for column in free] for row in range(3)]
    unimodular = [[1 if row == column else 0 for column in range(3)] for row in range(3)]

    pivot_row = 0
    for column in range(len(free)):
        # Drive all entries below the pivot to zero by repeated gcd steps.
        for row in range(pivot_row + 1, 3):
            while rows[row][column] != 0:
                if abs(rows[pivot_row][column]) > abs(rows[row][column]) or rows[pivot_row][column] == 0:
                    rows[pivot_row], rows[row] = rows[row], rows[pivot_row]
                    unimodular[pivot_row], unimodular[row] = unimodular[row], unimodular[pivot_row]
                if rows[pivot_row][column] == 0:
                    break
                factor = rows[row][column] // rows[pivot_row][column]
                rows[row] = [a - factor * b for a, b in zip(rows[row], rows[pivot_row])]
                unimodular[row] = [a - factor * b for a, b in zip(unimodular[row], unimodular[pivot_row])]
        if rows[pivot_row][column] != 0:
            pivot_row += 1
        if pivot_row == 3:
            break

    return FracVector.create(unimodular), FracVector.create(rows)
