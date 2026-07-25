"""Exact rational affine maps on fractional coordinates.

An :class:`AffineOperation` is a rotation part and a translation part held exactly as
:class:`~httk.core.FracVector` values. The same object serves for the two things this
package does with affine maps — a crystallographic symmetry operation, and a
change-of-basis between space-group settings — because they are the same algebra; only
their interpretation differs (see :class:`~httk.atomistic.SettingTransform` for the
latter).

Everything here is closed over the rationals: composing, inverting, conjugating, and
applying an operation to reduced coordinates all stay exact. That is the whole reason the
ASU machinery needs no tolerance.

**Convention.** The stored ``matrix`` is written for column vectors, matching how
crystallographic tables print an operation (``x' = W x + w``, i.e. ``-x+1/2,y,-z``).
httk holds coordinates as *rows*, so :meth:`apply` evaluates ``coords * matrix.T() +
vector``. Both spellings describe the same map; only one of them is ever written out in
code here.
"""

import fractions
from typing import Any, Iterable, Self

from httk.core import FracVector

__all__ = ["AffineOperation"]


class AffineOperation:
    """An exact affine map ``x -> W x + w`` on fractional coordinates.

    Two operations compare equal when their matrix and translation are exactly equal.
    Equality is *not* modulo lattice translations — use :meth:`wrapped` first when
    comparing symmetry operations as members of a space group, since ``x+1/2`` and
    ``x+3/2`` are the same operation there but different objects here.
    """

    _matrix: FracVector
    _vector: FracVector
    _transposed_cache: FracVector | None
    _inverse_cache: "AffineOperation | None"

    def __init__(self, matrix: Any, vector: Any = (0, 0, 0)) -> None:
        self._matrix = FracVector.create(matrix)
        self._vector = FracVector.create(vector)
        if self._matrix.dim != (3, 3):
            raise ValueError(f"AffineOperation matrix must be 3x3, got dim {self._matrix.dim}")
        if self._vector.dim != (3,):
            raise ValueError(f"AffineOperation vector must have 3 elements, got dim {self._vector.dim}")
        self._transposed_cache = None
        self._inverse_cache = None

    # --- constructors ---

    @classmethod
    def identity(cls) -> Self:
        """The identity operation."""
        return cls(FracVector.eye((3, 3)), (0, 0, 0))

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Self:
        """Build from a vendored ``affine_transformation`` record.

        The record's ``matrix`` and ``vector`` hold exact rational strings (``"1/2"``,
        ``"-1"``), which embed exactly.
        """
        return cls(record["matrix"], record["vector"])

    @classmethod
    def from_symop_record(cls, record: dict[str, Any]) -> Self:
        """Build from a vendored ``symops`` entry (which nests its affine part)."""
        return cls.from_record(record["affine_transformation"])

    # --- accessors ---

    @property
    def matrix(self) -> FracVector:
        """The 3x3 rotation part ``W``, in the column-vector convention."""
        return self._matrix

    @property
    def vector(self) -> FracVector:
        """The translation part ``w``."""
        return self._vector

    @property
    def _transposed(self) -> FracVector:
        """``W.T()``, cached: the form :meth:`apply` needs for row coordinates."""
        if self._transposed_cache is None:
            self._transposed_cache = self._matrix.T()
        return self._transposed_cache

    def determinant(self) -> fractions.Fraction:
        """The determinant of the rotation part, exactly.

        For a symmetry operation this is ``+1`` (proper) or ``-1`` (improper). For a
        change of basis it is the ratio of cell volumes, so a value other than ``±1``
        means the operation changes the lattice.
        """
        return self._matrix.det().to_fraction()

    def is_identity(self) -> bool:
        return self._matrix == FracVector.eye((3, 3)) and self._vector == FracVector.zeros((3,))

    # --- application ---

    def apply(self, coords: Any) -> FracVector:
        """Map reduced coordinates through this operation, exactly.

        ``coords`` is a single ``(3,)`` coordinate or an ``(N, 3)`` block of them; the
        result has the same shape. No wrapping is applied — see :meth:`apply_wrapped`.
        """
        reduced = FracVector.create(coords)
        if reduced.dim == (3,):
            return reduced * self._transposed + self._vector
        if len(reduced.dim) == 2 and reduced.dim[1] == 3:
            # Broadcasting the translation over rows keeps this a single exact expression.
            return reduced * self._transposed + FracVector.create([self._vector.to_fractions()] * reduced.dim[0])
        raise ValueError(f"expected a (3,) coordinate or an (N, 3) block, got dim {reduced.dim}")

    def apply_wrapped(self, coords: Any) -> FracVector:
        """:meth:`apply`, then wrap every component into ``[0, 1)``."""
        result: FracVector = self.apply(coords).normalize()
        return result

    # --- algebra ---

    def inverse(self) -> "AffineOperation":
        """The inverse map, exactly. Raises :class:`ZeroDivisionError` if singular."""
        if self._inverse_cache is None:
            inverse_matrix = self._matrix.inv()
            # x = W^-1 (x' - w)  =>  matrix W^-1, translation -W^-1 w
            inverse_vector = FracVector.create([0, 0, 0]) - self._vector * inverse_matrix.T()
            self._inverse_cache = AffineOperation(inverse_matrix, inverse_vector)
        return self._inverse_cache

    def __mul__(self, other: "AffineOperation") -> "AffineOperation":
        """Composition: ``(self * other)`` applies ``other`` first, then ``self``."""
        if not isinstance(other, AffineOperation):
            return NotImplemented
        return AffineOperation(
            self._matrix * other._matrix,
            other._vector * self._transposed + self._vector,
        )

    def conjugated_by(self, change: "AffineOperation") -> "AffineOperation":
        """This operation seen through a change of basis: ``change * self * change^-1``.

        If ``self`` is a symmetry operation expressed in one setting and ``change`` maps
        that setting's coordinates into another, the result is the same symmetry operation
        expressed in the other setting.
        """
        return change * self * change.inverse()

    def wrapped(self) -> "AffineOperation":
        """The same operation with its translation reduced into ``[0, 1)``.

        Two symmetry operations of a space group are the same element modulo lattice
        translations exactly when their wrapped forms are equal, which is what makes
        symop *sets* comparable.
        """
        return AffineOperation(self._matrix, self._vector.normalize())

    # --- identity and display ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineOperation):
            return NotImplemented
        return self._matrix == other._matrix and self._vector == other._vector

    def __hash__(self) -> int:
        """Hash consistent with :meth:`__eq__`, so operations deduplicate in a set.

        Relies on :class:`~httk.core.FracVector` hashing its canonical form, which is what
        makes ``(1, 0, 0)/2`` and ``(2, 0, 0)/4`` land in the same bucket. Arithmetic here
        does not reduce as it goes, so equal operations routinely arrive with different
        denominators, and the exact deduplication in ASU expansion depends on them being
        recognized as one.
        """
        return hash((self._matrix, self._vector))

    def __repr__(self) -> str:
        return f"AffineOperation({self.to_xyz()!r})"

    def to_xyz(self) -> str:
        """The operation in ``x,y,z`` notation, e.g. ``"-x+1/2,y,-z+1/2"``."""
        return ",".join(
            _component_to_xyz(row, translation)
            for row, translation in zip(self._matrix.to_fractions(), self._vector.to_fractions())
        )


def _signed(value: fractions.Fraction) -> str:
    """A rational with an explicit sign, e.g. ``+1/2``. ``Fraction`` has no ``+`` format spec."""
    return f"+{value}" if value >= 0 else str(value)


def _component_to_xyz(row: Iterable[fractions.Fraction], translation: fractions.Fraction) -> str:
    """One component of an ``x,y,z`` triplet, e.g. ``-x+1/2``."""
    parts: list[str] = []
    for coefficient, name in zip(row, ("x", "y", "z")):
        if coefficient == 0:
            continue
        if coefficient == 1:
            parts.append(f"+{name}")
        elif coefficient == -1:
            parts.append(f"-{name}")
        else:
            parts.append(f"{_signed(coefficient)}*{name}")
    if translation != 0:
        parts.append(_signed(translation))
    if not parts:
        return "0"
    text = "".join(parts)
    return text[1:] if text.startswith("+") else text
