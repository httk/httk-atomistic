"""Express an asymmetric-unit structure in its IT standard-setting cell.

The operation is exact after any optional recognition step. An ASU already carries the
standard-setting Wyckoff data and the stored transform from that setting to its own cell,
so the conventional cell is obtained by transforming the cell basis back and expanding a
new identity-transform ASU.
"""

import fractions
from dataclasses import dataclass

from httk.core import FracVector, unwrap

from .asu_recognition import recognize_asu
from .asu_structure import ASUStructure
from .cell import Cell
from .setting_transform import SettingTransform
from .spacegroup import Spacegroup
from .structure import Structure
from .structure_like import StructureLike
from .unitcell_structure_view import UnitcellStructureView

__all__ = ["ConventionalCellResult", "conventional_cell"]


@dataclass(frozen=True, slots=True)
class ConventionalCellResult:
    """A structure in its space group's IT standard-setting conventional cell.

    ``asu`` is the new standard-setting ASU that was expanded to make ``structure``.
    ``transform`` is the standard-to-own transform from the ASU that was supplied to, or
    recognized from, the operation; its orientation is :math:`f_own = f_std M^T + v`, so
    this operation undoes it for the cell basis. ``multiplier`` is the exact ratio of
    conventional-cell site count to input-cell site count. For the 527 vendored settings
    it is at least one; an untabulated, caller-supplied supercell transform may still
    produce a ratio below one.
    """

    structure: Structure
    asu: ASUStructure
    spacegroup: Spacegroup
    transform: SettingTransform
    multiplier: fractions.Fraction


def _scaled_precision(
    precision: fractions.Fraction | None,
    factor: fractions.Fraction,
) -> fractions.Fraction | None:
    return None if precision is None else precision * factor


def _matrix_row_sum_factor(matrix: FracVector) -> fractions.Fraction:
    rows = matrix.to_fractions()
    return max(sum((abs(value) for value in row), start=fractions.Fraction(0)) for row in rows)


def _matrix_column_sum_factor(matrix: FracVector) -> fractions.Fraction:
    rows = matrix.to_fractions()
    return max(sum((abs(rows[row][column]) for row in range(3)), start=fractions.Fraction(0)) for column in range(3))


def _as_existing_asu(structure: StructureLike | ASUStructure) -> ASUStructure | None:
    """Return an already-held ASU through any backend/view unwrap chain."""
    candidate: object = structure
    visited: set[int] = set()
    while id(candidate) not in visited:
        visited.add(id(candidate))
        if isinstance(candidate, ASUStructure):
            return candidate
        direct = getattr(candidate, "asu", None)
        if isinstance(direct, ASUStructure):
            return direct
        unwrapped = unwrap(candidate)
        if unwrapped is candidate:
            break
        candidate = unwrapped
    return None


def conventional_cell(
    structure: StructureLike | ASUStructure,
    *,
    tolerance: float | None = None,
    limit_denominator: int | None = None,
) -> ConventionalCellResult:
    """Return ``structure`` in its space group's IT standard-setting conventional cell.

    An existing :class:`~httk.atomistic.ASUStructure` (including an
    :class:`~httk.atomistic.ASUStructureView`, an ASU backend, or a full-cell view backed
    by one) is used exactly as stored. Supplying ``tolerance`` or ``limit_denominator`` for
    that path raises :class:`ValueError`, because those arguments belong to recognition.
    Any other :class:`~httk.atomistic.StructureLike` is first passed to
    :func:`~httk.atomistic.recognize_asu`. That tolerant step may snap measured coordinates
    onto symmetry positions and chooses the transform recorded in the result; it does not
    preserve an unstated input transform or promise a :func:`~httk.atomistic.same_crystal`
    match to noisy input coordinates. The optional tolerance is a Cartesian matching
    distance and the optional denominator limit idealises free parameters.

    The returned ``transform`` is the existing ASU's transform, or the transform chosen by
    recognition for a plain input; the returned ``asu`` has an identity transform.
    Construction and expansion are exact, including the rhombohedral case where the
    standard hexagonal cell contains three primitive cells. Basis precision is multiplied by
    ``M.T()`` and coordinate precision by the maximum absolute column sum of
    ``inv(M.T())``; unknown precision remains unknown. Requires a fully 3D-periodic
    structure.
    """
    original = UnitcellStructureView(structure)
    asu = _as_existing_asu(structure)
    if asu is not None:
        if tolerance is not None or limit_denominator is not None:
            raise ValueError("conventional_cell() tolerance and limit_denominator cannot be used with an existing ASU")
    else:
        asu = recognize_asu(
            original,
            tolerance=tolerance,
            limit_denominator=limit_denominator,
        )

    assert asu is not None
    transform = asu.transform
    basis_matrix = transform.matrix.T()
    coordinate_matrix = basis_matrix.inv()
    new_cell_precision = _scaled_precision(
        asu.cell.precision,
        _matrix_row_sum_factor(basis_matrix),
    )
    new_coordinate_precision = _scaled_precision(
        asu.coordinate_precision,
        _matrix_column_sum_factor(coordinate_matrix),
    )
    new_cell = Cell(
        transform.basis_to_standard(asu.cell.basis),
        precision=new_cell_precision,
        periodicity=asu.cell.periodicity,
    )
    standard_asu = ASUStructure(
        new_cell,
        asu.spacegroup,
        asu.asu_sites,
        asu.species,
        transform=SettingTransform.identity(),
        coordinate_precision=new_coordinate_precision,
    )
    result_structure = UnitcellStructureView(standard_asu)
    original_count = len(original.sites)
    if original_count == 0:
        raise ValueError("conventional_cell() cannot determine a site-count multiplier for an empty structure")
    multiplier = fractions.Fraction(len(result_structure.sites), original_count)

    # All tabulated transforms have determinant 1 or 3. Keep this invariant explicit while
    # allowing a caller-supplied untabulated transform to describe a larger own cell.
    if asu.setting() is not None:
        assert multiplier >= 1

    return ConventionalCellResult(
        result_structure,
        standard_asu,
        asu.spacegroup,
        transform,
        multiplier,
    )
