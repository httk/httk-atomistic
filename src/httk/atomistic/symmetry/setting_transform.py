"""The change of basis between a space-group setting and the IT standard setting.

A crystal structure can be written in any setting of its space group — a different choice
of axes, a different origin, or something that appears in no table at all. httk represents
such a structure by holding its Wyckoff data in the International Tables **standard**
setting and carrying a :class:`SettingTransform` that says how to get from there to the
setting the structure is actually in. That pairing is what lets an arbitrary,
non-tabulated setting be represented losslessly.

**Direction.** A :class:`SettingTransform` maps *standard* coordinates into the
structure's *own* setting::

    f_own = f_std * M.T() + v

with the reverse ``f_std = (f_own - v) * inv(M).T()`` and cell basis rows transforming as
``B_own = inv(M).T() * B_std``. Applying it backwards produces a structurally valid but
systematically wrong crystal.

**Never solve for a transform.** A transform is *stored*, never re-derived by searching
for one that maps one group onto another. Such a search is massively underdetermined — for
SG 15 alone there are 192 valid pairs with integer entries in ``{-1,0,1}`` and translations
on a 1/24 grid, and the true solution set is the full affine-normalizer coset, which is
infinite. Two structures with the same group, the same Wyckoff letters and the same free
parameters but different transforms are *different crystals*, so picking an arbitrary
member of that family would silently produce the wrong structure.
"""

import fractions
from typing import Any, Self

from httk.core import FracVector, SurdVector

from httk.atomistic import data
from httk.atomistic.symmetry._lattice import finite_translation_cosets
from httk.atomistic.symmetry.affine_operation import AffineOperation

__all__ = ["SettingTransform"]


class SettingTransform:
    """An exact rational change of basis from the IT standard setting to another setting.

    Wraps an :class:`~httk.atomistic.AffineOperation` and gives it the standard-to-own
    reading described in the module docstring, plus the cell-basis and symmetry-operation
    transformations that follow from it.
    """

    _operation: AffineOperation
    _hall_entry: str | None
    _cosets_cache: tuple[FracVector, ...] | None

    def __init__(self, matrix: Any, vector: Any = (0, 0, 0), *, hall_entry: str | None = None) -> None:
        self._operation = AffineOperation(matrix, vector)
        if self._operation.determinant() == 0:
            raise ValueError("a setting transform must be invertible, but its matrix is singular")
        self._hall_entry = hall_entry
        self._cosets_cache = None

    # --- constructors ---

    @classmethod
    def identity(cls) -> Self:
        """The transform of a structure already in its IT standard setting."""
        return cls(FracVector.eye((3, 3)), (0, 0, 0))

    @classmethod
    def for_hall_entry(cls, hall_entry: str) -> Self:
        """The tabulated transform for one of the 527 known settings.

        ``hall_entry`` is the normalized Hall symbol of the setting, which names it
        unambiguously — symbol, axes, and origin choice together.
        """
        record = data.setting_transform(hall_entry)
        affine = record["affine_transformation"]
        return cls(affine["matrix"], affine["vector"], hall_entry=hall_entry)

    # --- accessors ---

    @property
    def operation(self) -> AffineOperation:
        """The underlying affine map, standard to own setting."""
        return self._operation

    @property
    def matrix(self) -> FracVector:
        """The 3x3 rotation part ``M``."""
        return self._operation.matrix

    @property
    def vector(self) -> FracVector:
        """The origin shift ``v``."""
        return self._operation.vector

    @property
    def hall_entry(self) -> str | None:
        """The Hall entry this transform was looked up for, if it came from the tables."""
        return self._hall_entry

    def determinant(self) -> fractions.Fraction:
        """``det M``: the ratio of the own cell's volume to the standard cell's, inverted.

        ``1`` for 520 of the 527 tabulated settings. The exceptions are the seven
        rhombohedral-axes settings (IT numbers 146, 148, 155, 160, 161, 166, 167) where it
        is ``3``, because the standard hexagonal cell holds three primitive rhombohedral
        cells. A caller-supplied transform may have any non-zero value.
        """
        return self._operation.determinant()

    def is_identity(self) -> bool:
        """Whether the structure is already in its IT standard setting."""
        return self._operation.is_identity()

    # --- coordinates ---

    def to_setting(self, coords: Any) -> FracVector:
        """Map standard-setting reduced coordinates into this setting. Not wrapped."""
        return self._operation.apply(coords)

    def to_standard(self, coords: Any) -> FracVector:
        """Map this setting's reduced coordinates into the standard setting. Not wrapped."""
        return self._operation.inverse().apply(coords)

    # --- symmetry operations ---

    def symop_to_setting(self, operation: AffineOperation) -> AffineOperation:
        """A symmetry operation written in the standard setting, rewritten in this one."""
        return operation.conjugated_by(self._operation)

    def symop_to_standard(self, operation: AffineOperation) -> AffineOperation:
        """A symmetry operation written in this setting, rewritten in the standard one."""
        return operation.conjugated_by(self._operation.inverse())

    # --- cell basis ---

    def basis_to_setting(self, basis: Any) -> SurdVector:
        """Map a standard-setting cell basis (lattice vectors as rows) into this setting.

        Follows from coordinate invariance: if ``f_own = f_std * M.T()`` then
        ``B_own = inv(M).T() * B_std``, so that ``f * B`` is the same Cartesian point
        either way. The transform is rational, so an exact basis stays exact — a
        hexagonal cell keeps its ``sqrt(3)``.
        """
        return SurdVector.create(self.matrix.T().inv()) * SurdVector.create(basis)

    def basis_to_standard(self, basis: Any) -> SurdVector:
        """Map this setting's cell basis into the standard setting."""
        return SurdVector.create(self.matrix.T()) * SurdVector.create(basis)

    # --- lattice change ---

    def lattice_cosets(self) -> tuple[FracVector, ...]:
        """Translations of this setting's cell that are standard-lattice translations.

        Expanding an orbit generates points from the standard setting's symmetry
        operations, which carry the standard lattice's periodicity. When this setting's
        cell is *larger* than the standard one, that is not enough: points related by a
        standard lattice translation are genuinely distinct sites here, and the missing
        ones are recovered by also applying each translation returned by this method.

        The result is the finite subgroup of ``(Q/Z)^3`` generated by the columns of ``M``
        reduced modulo 1, always including the zero translation. It is trivial (just zero)
        whenever ``M`` is an integer matrix, which covers **all 527** tabulated settings —
        including the seven with ``det M == 3``, where this setting's cell is *smaller* and
        the surplus points collapse under wrapping instead. So this only ever does work for
        a caller-supplied transform to a supercell setting.
        """
        if self._cosets_cache is None:
            self._cosets_cache = self._compute_cosets()
        return self._cosets_cache

    def _compute_cosets(self) -> tuple[FracVector, ...]:
        columns = [self.matrix.T()[index].normalize() for index in range(3)]
        return finite_translation_cosets(columns)

    # --- algebra ---

    def inverse(self) -> "SettingTransform":
        """The transform in the opposite direction, from this setting to the standard one."""
        inverted = self._operation.inverse()
        return SettingTransform(inverted.matrix, inverted.vector)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SettingTransform):
            return NotImplemented
        return self._operation == other._operation

    def __hash__(self) -> int:
        return hash(self._operation)

    def __repr__(self) -> str:
        if self.is_identity():
            return "SettingTransform.identity()"
        if self._hall_entry is not None:
            return f"SettingTransform.for_hall_entry({self._hall_entry!r})"
        return f"SettingTransform({self._operation.to_xyz()!r})"
