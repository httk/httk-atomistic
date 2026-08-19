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
httk holds coordinates as *rows*, so :meth:`AffineOperation.apply` evaluates
``coords * matrix.T() + vector``. Both spellings describe the same map; only one of them is ever written out in
code here.
"""

import fractions
from collections.abc import Iterable, Mapping
from typing import Any, Self

from httk.core import FracVector

__all__ = ["AffineOperation"]


class AffineOperation:
    """Represent an exact affine map ``x -> W x + w`` on fractional coordinates.

    Two operations compare equal when their matrix and translation are exactly equal.
    Equality is *not* modulo lattice translations — use :meth:`wrapped` first when
    comparing symmetry operations as members of a space group, since ``x+1/2`` and
    ``x+3/2`` are the same operation there but different objects here.

    :param matrix: The 3x3 rotation part in the column-vector convention, or a comma-separated
        ``"x,y,z"`` operation string (as emitted by :meth:`to_xyz` and ``repr``), in which case
        ``vector`` is ignored.
    :param vector: The translation part in fractional coordinates.
    """

    _matrix: FracVector
    _vector: FracVector
    _transposed_cache: FracVector | None
    _inverse_cache: "AffineOperation | None"

    def __init__(self, matrix: Any, vector: Any = (0, 0, 0)) -> None:
        if isinstance(matrix, str):
            # Accept the "x,y,z" form that repr/to_xyz emit, so eval(repr(op)) round-trips.
            # Imported lazily: xyz.py imports this module, so a top-level import would be circular.
            from httk.atomistic.symmetry.xyz import operation_from_xyz

            parsed = operation_from_xyz(matrix)
            matrix, vector = parsed._matrix, parsed._vector
        self._matrix = FracVector(matrix)
        self._vector = FracVector(vector)
        if self._matrix.dim != (3, 3):
            raise ValueError(f"AffineOperation matrix must be 3x3, got dim {self._matrix.dim}")
        if self._vector.dim != (3,):
            raise ValueError(f"AffineOperation vector must have 3 elements, got dim {self._vector.dim}")
        self._transposed_cache = None
        self._inverse_cache = None

    # --- constructors ---

    @classmethod
    def identity(cls) -> Self:
        """Return the identity operation.

        :return: The identity affine operation.
        """
        return cls(FracVector.eye((3, 3)), (0, 0, 0))

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Self:
        """Build an operation from either vendored affine-record shape.

        The record's ``matrix`` and ``vector`` hold exact rational strings (``"1/2"``,
        ``"-1"``), which embed exactly.

        :param record: The affine record, or a mapping containing one under
            ``"affine_transformation"``.
        :return: The corresponding affine operation.
        """
        affine = record.get("affine_transformation", record)
        return cls(affine["matrix"], affine["vector"])

    # --- accessors ---

    @property
    def matrix(self) -> FracVector:
        """Return the 3x3 rotation part ``W`` in the column-vector convention.

        :return: The exact rotation matrix.
        """
        return self._matrix

    @property
    def vector(self) -> FracVector:
        """Return the translation part ``w``.

        :return: The exact translation vector.
        """
        return self._vector

    @property
    def _transposed(self) -> FracVector:
        """``W.T()``, cached: the form :meth:`apply` needs for row coordinates."""
        if self._transposed_cache is None:
            self._transposed_cache = self._matrix.T()
        return self._transposed_cache

    def determinant(self) -> fractions.Fraction:
        """Return the determinant of the rotation part exactly.

        For a symmetry operation this is ``+1`` (proper) or ``-1`` (improper). For a
        change of basis it is the ratio of cell volumes, so a value other than ``±1``
        means the operation changes the lattice.

        :return: The exact determinant.
        """
        return self._matrix.det().to_fraction()

    def is_identity(self) -> bool:
        """Report whether this operation is the identity.

        :return: Whether the matrix and translation are both identity values.
        """
        return self._matrix == FracVector.eye((3, 3)) and self._vector == FracVector.zeros((3,))

    # --- application ---

    def apply(self, coords: Any) -> FracVector:
        """Map reduced coordinates through this operation exactly.

        ``coords`` is a single ``(3,)`` coordinate or an ``(N, 3)`` block of them; the
        result has the same shape. No wrapping is applied — see :meth:`apply_wrapped`.

        :param coords: A single reduced coordinate or a block of reduced coordinates.
        :return: The transformed coordinates with the same shape as ``coords``.
        :raises ValueError: If ``coords`` is neither a length-three coordinate nor an
            ``(N, 3)`` block.
        """
        reduced = FracVector(coords)
        if reduced.dim == (3,):
            return reduced * self._transposed + self._vector
        if len(reduced.dim) == 2 and reduced.dim[1] == 3:
            # Broadcasting the translation over rows keeps this a single exact expression.
            return reduced * self._transposed + FracVector([self._vector.to_fractions()] * reduced.dim[0])
        raise ValueError(f"expected a (3,) coordinate or an (N, 3) block, got dim {reduced.dim}")

    def apply_wrapped(self, coords: Any) -> FracVector:
        """Map coordinates and wrap every component into ``[0, 1)``.

        :param coords: A single reduced coordinate or a block of reduced coordinates.
        :return: The transformed and wrapped coordinates.
        :raises ValueError: If ``coords`` has an unsupported shape.
        """
        result: FracVector = self.apply(coords).normalize()
        return result

    # --- algebra ---

    def inverse(self) -> "AffineOperation":
        """Return the inverse map exactly.

        :return: The inverse affine operation.
        :raises ZeroDivisionError: If the rotation part is singular.
        """
        if self._inverse_cache is None:
            inverse_matrix = self._matrix.inv()
            # x = W^-1 (x' - w)  =>  matrix W^-1, translation -W^-1 w
            inverse_vector = FracVector([0, 0, 0]) - self._vector * inverse_matrix.T()
            self._inverse_cache = AffineOperation(inverse_matrix, inverse_vector)
        return self._inverse_cache

    def __mul__(self, other: "AffineOperation") -> "AffineOperation":
        """Compose operations so ``self * other`` applies ``other`` first.

        :param other: The operation to apply first.
        :return: The composed affine operation.
        """
        if not isinstance(other, AffineOperation):
            return NotImplemented
        return AffineOperation(
            self._matrix * other._matrix,
            other._vector * self._transposed + self._vector,
        )

    def conjugated_by(self, change: "AffineOperation") -> "AffineOperation":
        """Rewrite this operation through a change of basis.

        If ``self`` is a symmetry operation expressed in one setting and ``change`` maps
        that setting's coordinates into another, the result is the same symmetry operation
        expressed in the other setting.

        :param change: The affine operation defining the coordinate change.
        :return: This operation conjugated by ``change``.
        """
        return change * self * change.inverse()

    def wrapped(self) -> "AffineOperation":
        """Return this operation with its translation reduced into ``[0, 1)``.

        Two symmetry operations of a space group are the same element modulo lattice
        translations exactly when their wrapped forms are equal, which is what makes
        symop *sets* comparable.

        :return: An operation with a normalized translation.
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

        :return: A hash consistent with :meth:`__eq__`.
        """
        return hash((self._matrix, self._vector))

    def __repr__(self) -> str:
        return f"AffineOperation({self.to_xyz()!r})"

    def to_xyz(self) -> str:
        """Render the operation in ``x,y,z`` notation.

        For example, return ``"-x+1/2,y,-z+1/2"``.

        :return: The crystallographic operation string.
        """
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
    return text.removeprefix("+")
