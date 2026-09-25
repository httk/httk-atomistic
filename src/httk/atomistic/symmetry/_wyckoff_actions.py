"""Compile exact affine-normalizer actions on Wyckoff parameter charts."""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any

from httk.core import FracVector

from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.spacegroup import Spacegroup
from httk.atomistic.symmetry.wyckoff import WyckoffBranch, WyckoffPosition

type _Matrix = tuple[tuple[Fraction, ...], ...]
type _Family = tuple[_Matrix, tuple[Fraction, ...]]


class UnsupportedWyckoffAction(ValueError):
    """Report an affine operation whose exact Wyckoff action cannot be compiled."""


@dataclass(frozen=True)
class WyckoffAction:
    """Map one Wyckoff letter and its free parameters exactly."""

    source_letter: str
    target_letter: str
    matrix: _Matrix
    vector: tuple[Fraction, ...]

    def apply(self, parameters: Any) -> FracVector:
        """Apply the compiled affine parameter map modulo one."""
        values = FracVector(parameters)
        rank = len(self.matrix)
        if not rank:
            if values.dim not in ((), (0,)):
                raise ValueError(f"fixed Wyckoff position takes no parameters, got dim {values.dim}")
            return FracVector(())
        if values.dim != (rank,):
            raise ValueError(f"expected {rank} Wyckoff parameter(s), got dim {values.dim}")
        fractions = values.to_fractions()
        return FracVector(
            [
                sum((fractions[row] * self.matrix[row][column] for row in range(rank)), Fraction())
                + self.vector[column]
                for column in range(rank)
            ]
        ).normalize()


def _multiply(left: _Matrix, right: _Matrix) -> _Matrix:
    if not left:
        return ()
    return tuple(
        tuple(
            sum((left[row][inner] * right[inner][column] for inner in range(len(right))), Fraction())
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _row_multiply(row: tuple[Fraction, ...], matrix: _Matrix) -> tuple[Fraction, ...]:
    return tuple(
        sum((row[inner] * matrix[inner][column] for inner in range(len(row))), Fraction())
        for column in range(len(matrix[0]))
    )


def _transpose(matrix: list[list[Fraction]] | _Matrix) -> _Matrix:
    return tuple(tuple(matrix[row][column] for row in range(len(matrix))) for column in range(len(matrix[0])))


def _branch_family(branch: WyckoffBranch) -> _Family:
    matrix = branch.operation.matrix.to_fractions()
    coefficients = tuple(tuple(matrix[coordinate][free] for coordinate in range(3)) for free in branch.free)
    return coefficients, tuple(branch.operation.vector.to_fractions())


def _transformed_family(branch: WyckoffBranch, operation: AffineOperation) -> _Family:
    coefficients, constant = _branch_family(branch)
    transform = operation.matrix.to_fractions()
    transformed = tuple(
        tuple(
            sum((transform[coordinate][inner] * coefficients[row][inner] for inner in range(3)), Fraction())
            for coordinate in range(3)
        )
        for row in range(len(coefficients))
    )
    translated = tuple(
        (
            sum((transform[coordinate][inner] * constant[inner] for inner in range(3)), Fraction())
            + operation.vector[coordinate].to_fraction()
        )
        for coordinate in range(3)
    )
    return transformed, translated


def _composed_family(branch: WyckoffBranch, matrix: _Matrix, vector: tuple[Fraction, ...]) -> _Family:
    coefficients, constant = _branch_family(branch)
    if not matrix:
        return (), tuple(value % 1 for value in constant)
    return _multiply(matrix, coefficients), tuple(
        (value + offset) % 1 for value, offset in zip(_row_multiply(vector, coefficients), constant, strict=True)
    )


def _family_key(family: _Family) -> _Family:
    coefficients, constant = family
    return coefficients, tuple(value % 1 for value in constant)


def _parameter_map(
    source: WyckoffPosition,
    target_branch: WyckoffBranch,
    operation: AffineOperation,
) -> tuple[_Matrix, tuple[Fraction, ...]] | None:
    coefficients, constant = _transformed_family(source.representative, operation)
    difference = tuple(
        left - right for left, right in zip(constant, target_branch.operation.vector.to_fractions(), strict=True)
    )
    unimodular = tuple(tuple(row) for row in target_branch._unimodular.to_fractions())
    mapped_coefficients = _multiply(coefficients, _transpose(unimodular))
    mapped_constant = _row_multiply(difference, _transpose(unimodular))
    rank = source.free_count
    if any(mapped_coefficients[row][column] for row in range(rank) for column in range(rank, 3)):
        return None
    if any(mapped_constant[column].denominator != 1 for column in range(rank, 3)):
        return None
    if not rank:
        return (), ()
    inverse = tuple(tuple(row) for row in target_branch._pivot_inverse.to_fractions())
    parameter_matrix = _multiply(tuple(row[:rank] for row in mapped_coefficients), _transpose(inverse))
    parameter_vector = _row_multiply(mapped_constant[:rank], _transpose(inverse))
    return parameter_matrix, tuple(value % 1 for value in parameter_vector)


def _orbit_matches(
    source: WyckoffPosition,
    target: WyckoffPosition,
    operation: AffineOperation,
    matrix: _Matrix,
    vector: tuple[Fraction, ...],
) -> bool:
    actual = Counter(_family_key(_transformed_family(branch, operation)) for branch in source.branches)
    expected = Counter(_family_key(_composed_family(branch, matrix, vector)) for branch in target.branches)
    return actual == expected


@lru_cache(maxsize=100_000)
def compile_wyckoff_action(
    spacegroup: Spacegroup,
    operation: AffineOperation,
    source_letter: str,
    *,
    target_spacegroup: Spacegroup | None = None,
) -> WyckoffAction:
    """Compile and prove one exact Wyckoff action without coordinate sampling.

    The proof compares complete affine orbit families coefficient by coefficient modulo lattice
    translations. Multiple target branches can be valid charts of one orbit; table order chooses a
    deterministic chart only after whole-orbit equality has been established.  Omitting
    ``target_spacegroup`` compiles the usual same-group normalizer action.

    :param spacegroup: The source standard-setting space group.
    :param operation: The exact affine operation on reduced coordinates.
    :param source_letter: The source Wyckoff letter.
    :param target_spacegroup: The target standard-setting group, or the source group.
    :return: The proven target letter and parameter map.
    """
    target_spacegroup = spacegroup if target_spacegroup is None else target_spacegroup
    source = spacegroup.wyckoff_position(source_letter)
    for target in target_spacegroup.wyckoff:
        if target.free_count != source.free_count or target.multiplicity != source.multiplicity:
            continue
        for branch in target.branches:
            parameter_map = _parameter_map(source, branch, operation)
            if parameter_map is None:
                continue
            matrix, vector = parameter_map
            if _orbit_matches(source, target, operation, matrix, vector):
                return WyckoffAction(source_letter, target.letter, matrix, vector)
    raise UnsupportedWyckoffAction(
        f"cannot compile exact Wyckoff action from {spacegroup.setting} to {target_spacegroup.setting} for "
        f"{operation.to_xyz()} on letter {source_letter!r}"
    )
