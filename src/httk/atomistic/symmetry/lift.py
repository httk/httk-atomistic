"""Exact one-hop backward lifts through Bärnighausen tables.

The public functions in this module invert one tabulated subgroup descent.  Coordinates,
Wyckoff parameters, affine maps, modular solves, and returned shifts are rational.  A
Cartesian tolerance is used only when accepting a measured structure that is not an exact
solution of the assembled equations. Cell-metric validation covers monoclinic, orthorhombic,
tetragonal, trigonal, hexagonal, and cubic systems; every tabulated trigonal and hexagonal parent
is in a hexagonal-axes standard setting, so their metric constraint is a=b with alpha=beta=90 and
gamma=120. Normalizer retry applies tabulated cosets to child fractional coordinates, maps
successful results back with the exact inverse, and follows tabulated coset order.
"""

import itertools
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from httk.core import FracVector, SurdVector

from httk.atomistic import data
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.models.structure.asu import ASUStructure, WyckoffSite
from httk.atomistic.symmetry._lattice import finite_translation_cosets
from httk.atomistic.symmetry._periodicity_guard import require_full_periodicity
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.recognition import (
    _cartesian_distance_squared,
    structure_tolerance,
)
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup
from httk.atomistic.symmetry.subgroups import (
    SubgroupTransform,
    _standard_input,
    minimal_supergroups,
    subgroup_closure,
    subgroup_representation,
    subgroup_transforms,
    supergroup_closure,
)

__all__ = [
    "COMPATIBLE_CRYSTAL_SYSTEMS",
    "LiftResult",
    "backward_lift",
    "canonicalize",
    "highest_symmetry",
    "lift_candidates",
    "rerepresent",
]


COMPATIBLE_CRYSTAL_SYSTEMS: dict[str, frozenset[str]] = {
    "triclinic": frozenset({"triclinic"}),
    "monoclinic": frozenset({"triclinic", "monoclinic"}),
    "orthorhombic": frozenset({"triclinic", "monoclinic", "orthorhombic"}),
    "tetragonal": frozenset({"triclinic", "monoclinic", "orthorhombic", "tetragonal"}),
    "trigonal": frozenset({"triclinic", "trigonal"}),
    "hexagonal": frozenset({"triclinic", "trigonal", "hexagonal"}),
    "cubic": frozenset({"triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "cubic"}),
}

_MAX_SOLVER_BRANCHES = 200_000
_MAX_FOURIER_MOTZKIN_INEQUALITIES = 20_000


@dataclass(frozen=True, slots=True)
class LiftResult:
    """One exact or tolerance-accepted parent representation.

    :param asu: The parent-standard-setting asymmetric unit.
    :param spacegroup: The parent space group in standard setting.
    :param path: Child-first tabulated parent-to-child subgroup transforms used.
    :param shift: The continuous-normalizer origin shift from the final hop, expressed in that
        hop's parent standard frame; earlier hops' shifts are already baked into the intermediate
        representations recorded by ``path``, so the result structure is fully determined.
    :param residual: The largest wrapped fractional residual accepted.
    """

    asu: ASUStructure
    spacegroup: Spacegroup
    path: tuple[SubgroupTransform, ...]
    shift: FracVector
    residual: Fraction


@dataclass(frozen=True, slots=True)
class _Orbit:
    index: int
    site: WyckoffSite
    position: Any
    coordinates: tuple[FracVector, ...]


@dataclass(frozen=True, slots=True)
class _Equation:
    matrix: tuple[tuple[Fraction, ...], ...]
    constant: tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class _Candidate:
    parent_letter: str
    species: str
    pieces: tuple[tuple[int, int], ...]
    equations: tuple[_Equation, ...]
    covered: frozenset[int]


@dataclass(frozen=True, slots=True)
class _MetricCell:
    cell: Cell
    cartesian_deviation: float
    fractional_deviation: Fraction


def _fractions(value: Any) -> Any:
    """Return nested Fraction values from a vector-like value."""
    return value.to_fractions() if hasattr(value, "to_fractions") else value


def _matrix(value: Any) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(Fraction(item) for item in row) for row in _fractions(value))


def _vector(value: Any) -> tuple[Fraction, ...]:
    return tuple(Fraction(item) for item in _fractions(value))


def _matvec(matrix: tuple[tuple[Fraction, ...], ...], vector: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    return tuple(sum((a * b for a, b in zip(row, vector)), Fraction(0)) for row in matrix)


def _matmul(
    left: tuple[tuple[Fraction, ...], ...], right: tuple[tuple[Fraction, ...], ...]
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(sum((left[i][k] * right[k][j] for k in range(len(right))), Fraction(0)) for j in range(len(right[0])))
        for i in range(len(left))
    )


def _transpose(matrix: tuple[tuple[Fraction, ...], ...]) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(row[column] for row in matrix) for column in range(len(matrix[0])))


def _wrapped(value: Fraction) -> Fraction:
    """Wrap one exact fraction into ``[-1/2, 1/2)``."""
    return value - math.floor(value + Fraction(1, 2))


def _wrapped_tuple(value: Any) -> tuple[Fraction, ...]:
    return tuple(_wrapped(item) for item in _vector(value))


def _ceil_fraction(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def _integer_options(row: tuple[Fraction, ...], constant: Fraction) -> range:
    low = constant + sum(min(Fraction(0), value) for value in row)
    high = constant + sum(max(Fraction(0), value) for value in row)
    # x is in [0, 1); including the upper endpoint only adds a harmless branch at a
    # boundary.  This is the finite per-row bound; no guessed global wrap range is used.
    return range(_ceil_fraction(low), math.floor(high) + 1)


@dataclass(frozen=True, slots=True)
class _Inequality:
    coefficients: tuple[Fraction, ...]
    bound: Fraction
    strict: bool


def _fourier_motzkin(inequalities: tuple[_Inequality, ...]) -> tuple[_Inequality, ...]:
    """Eliminate the first variable from a rational inequality system exactly."""
    positive = [item for item in inequalities if item.coefficients[0] > 0]
    negative = [item for item in inequalities if item.coefficients[0] < 0]
    zero = [
        _Inequality(item.coefficients[1:], item.bound, item.strict) for item in inequalities if not item.coefficients[0]
    ]
    result = list(zero)
    for upper in positive:
        for lower in negative:
            upper_coefficient = upper.coefficients[0]
            lower_coefficient = lower.coefficients[0]
            result.append(
                _Inequality(
                    tuple(
                        -lower_coefficient * a + upper_coefficient * b
                        for a, b in zip(upper.coefficients[1:], lower.coefficients[1:], strict=True)
                    ),
                    -lower_coefficient * upper.bound + upper_coefficient * lower.bound,
                    upper.strict or lower.strict,
                )
            )
    if len(result) > _MAX_FOURIER_MOTZKIN_INEQUALITIES:
        # ponytail: capped FM; replace with a polyhedral package only if table dimensions grow materially.
        raise ValueError("Fourier-Motzkin inequality cap exceeded")
    return tuple(result)


def _inequality_feasible(inequalities: tuple[_Inequality, ...]) -> bool:
    return all((0 < item.bound if item.strict else 0 <= item.bound) for item in inequalities)


def _choose_feasible_free_values(inequalities: tuple[_Inequality, ...], free_count: int) -> tuple[Fraction, ...] | None:
    stages = [inequalities]
    for _ in range(free_count):
        stages.append(_fourier_motzkin(stages[-1]))
    if not _inequality_feasible(stages[-1]):
        return None
    chosen: list[Fraction] = []
    for stage in reversed(stages[:-1]):
        lower: tuple[Fraction, bool] | None = None
        upper: tuple[Fraction, bool] | None = None
        for inequality in stage:
            coefficient = inequality.coefficients[0]
            remainder = sum(value * chosen[index] for index, value in enumerate(inequality.coefficients[1:]))
            bound = inequality.bound - remainder
            if coefficient > 0:
                candidate = (bound / coefficient, inequality.strict)
                if upper is None or candidate[0] < upper[0] or (candidate[0] == upper[0] and candidate[1]):
                    upper = candidate
            elif coefficient < 0:
                candidate = (bound / coefficient, inequality.strict)
                if lower is None or candidate[0] > lower[0] or (candidate[0] == lower[0] and candidate[1]):
                    lower = candidate
            elif (
                inequality.strict
                and not remainder < inequality.bound
                or not inequality.strict
                and not remainder <= inequality.bound
            ):
                return None
        if lower is None or upper is None:
            return None
        if lower[0] > upper[0] or (lower[0] == upper[0] and (lower[1] or upper[1])):
            return None
        chosen.append(lower[0] if lower[0] == upper[0] else (lower[0] + upper[0]) / 2)
    chosen.reverse()
    return tuple(chosen)


def _linear_solve(matrix: tuple[tuple[Fraction, ...], ...], rhs: tuple[Fraction, ...]) -> tuple[Fraction, ...] | None:
    """Return a deterministic exact solution of a boxed rational linear system.

    Free variables are selected at the midpoint of their exact Fourier--Motzkin-feasible interval.
    """
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    if not rows:
        return ()
    width = len(matrix[0])
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(width):
        pivot = next((row for row in range(pivot_row, len(rows)) if rows[row][column]), None)
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        divisor = rows[pivot_row][column]
        rows[pivot_row] = [item / divisor for item in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row or not rows[row][column]:
                continue
            factor = rows[row][column]
            rows[row] = [a - factor * b for a, b in zip(rows[row], rows[pivot_row])]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    if any(not any(row[:width]) and row[width] for row in rows):
        return None
    free_columns = tuple(column for column in range(width) if column not in pivot_columns)
    if len(free_columns) > 12:
        raise ValueError("Fourier-Motzkin free-variable cap exceeded")
    inequalities: list[_Inequality] = []
    for index in range(len(free_columns)):
        coefficients = [Fraction(0)] * len(free_columns)
        coefficients[index] = Fraction(-1)
        inequalities.append(_Inequality(tuple(coefficients), Fraction(0), False))
        coefficients[index] = Fraction(1)
        inequalities.append(_Inequality(tuple(coefficients), Fraction(1), True))
    for row, column in enumerate(pivot_columns):
        pivot_coefficients = tuple(-rows[row][free_column] for free_column in free_columns)
        constant = rows[row][width]
        inequalities.append(_Inequality(tuple(-value for value in pivot_coefficients), constant, False))
        inequalities.append(_Inequality(pivot_coefficients, Fraction(1) - constant, True))
    free_values = _choose_feasible_free_values(tuple(inequalities), len(free_columns))
    if free_values is None:
        return None
    solution = [Fraction(0)] * width
    for column, value in zip(free_columns, free_values, strict=True):
        solution[column] = value
    for row, column in enumerate(pivot_columns):
        solution[column] = rows[row][width] - sum(
            rows[row][free_column] * solution[free_column] for free_column in free_columns
        )
    if any(value < 0 or value >= 1 for value in solution):
        return None
    return tuple(solution)


def _least_squares(matrix: tuple[tuple[Fraction, ...], ...], rhs: tuple[Fraction, ...]) -> tuple[Fraction, ...] | None:
    transpose = _transpose(matrix)
    normal = _matmul(transpose, matrix)
    target = _matvec(transpose, rhs)
    return _linear_solve(normal, target)


def _rational_null_space(rows: tuple[tuple[Fraction, ...], ...], width: int) -> tuple[tuple[Fraction, ...], ...]:
    """Return an exact basis of ``{x : rows @ x = 0}`` by reduced row echelon elimination."""
    matrix = [list(row) for row in rows]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(width):
        if pivot_row == len(matrix):
            break
        pivot = next((row for row in range(pivot_row, len(matrix)) if matrix[row][column]), None)
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        divisor = matrix[pivot_row][column]
        matrix[pivot_row] = [item / divisor for item in matrix[pivot_row]]
        for row in range(len(matrix)):
            if row != pivot_row and matrix[row][column]:
                factor = matrix[row][column]
                matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[pivot_row])]
        pivot_columns.append(column)
        pivot_row += 1
    free_columns = [column for column in range(width) if column not in pivot_columns]
    basis: list[tuple[Fraction, ...]] = []
    for free_column in free_columns:
        vector = [Fraction(0)] * width
        vector[free_column] = Fraction(1)
        for row, column in enumerate(pivot_columns):
            vector[column] = -matrix[row][free_column]
        basis.append(tuple(vector))
    return tuple(basis)


def _rational_inverse(matrix: tuple[tuple[Fraction, ...], ...]) -> tuple[tuple[Fraction, ...], ...]:
    """Return the exact inverse of a nonsingular rational square matrix by Gauss--Jordan."""
    size = len(matrix)
    augmented = [
        list(matrix[row]) + [Fraction(1) if row == column else Fraction(0) for column in range(size)]
        for row in range(size)
    ]
    for column in range(size):
        pivot = next(row for row in range(column, size) if augmented[row][column])
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [item / divisor for item in augmented[column]]
        for row in range(size):
            if row != column and augmented[row][column]:
                factor = augmented[row][column]
                augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[column])]
    return tuple(tuple(row[size:]) for row in augmented)


def _integer_consistency_system(
    matrix: tuple[tuple[Fraction, ...], ...], constants: tuple[Fraction, ...]
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Return the integer Diophantine system ``A n = d`` that ``n - c in col(M)`` requires.

    Each left-null-space vector ``L`` of ``M`` gives one exact constraint ``L n = L c``; clearing
    denominators turns the rational rows into integers.
    """
    height = len(matrix)
    unknowns = len(matrix[0]) if matrix else 0
    if unknowns:
        left_null = _rational_null_space(_transpose(matrix), height)
    else:
        # Zero unknowns: ``col(M) = {0}``, so every row must equal its constant and the full
        # identity is the constraint set.
        left_null = tuple(
            tuple(Fraction(1) if index == row else Fraction(0) for index in range(height)) for row in range(height)
        )
    rows: list[tuple[int, ...]] = []
    targets: list[int] = []
    for vector in left_null:
        rhs = sum((coefficient * constant for coefficient, constant in zip(vector, constants)), Fraction(0))
        denominator = math.lcm(*(value.denominator for value in (*vector, rhs)))
        rows.append(tuple(int(value * denominator) for value in vector))
        targets.append(int(rhs * denominator))
    return tuple(rows), tuple(targets)


def _integer_diophantine(
    rows: tuple[tuple[int, ...], ...], targets: tuple[int, ...], width: int
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]] | None:
    """Solve ``A n = d`` exactly over the integers by unimodular row reduction of ``[A^T | I]``.

    Returns one particular integer solution and an integer basis of ``{v : A v = 0}``, or ``None``
    when the system has no integer solution.
    """
    height = len(rows)
    # Each work row is ``(left, right)`` with ``left`` an integer combination of ``A``'s columns and
    # ``right`` the coefficients producing it; unimodular row operations preserve that invariant.
    work: list[tuple[list[int], list[int]]] = [
        ([rows[constraint][column] for constraint in range(height)], [1 if k == column else 0 for k in range(width)])
        for column in range(width)
    ]
    pivots: list[int] = []
    pivot_row = 0
    for column in range(height):
        if pivot_row == len(work):
            break
        while True:
            candidates = [row for row in range(pivot_row, len(work)) if work[row][0][column]]
            if not candidates:
                break
            leader = min(candidates, key=lambda row: abs(work[row][0][column]))
            work[pivot_row], work[leader] = work[leader], work[pivot_row]
            settled = True
            for row in range(pivot_row + 1, len(work)):
                if work[row][0][column]:
                    factor = work[row][0][column] // work[pivot_row][0][column]
                    if factor:
                        work[row] = (
                            [a - factor * b for a, b in zip(work[row][0], work[pivot_row][0])],
                            [a - factor * b for a, b in zip(work[row][1], work[pivot_row][1])],
                        )
                    if work[row][0][column]:
                        settled = False
            if settled:
                break
        if work[pivot_row][0][column]:
            if work[pivot_row][0][column] < 0:
                work[pivot_row] = ([-value for value in work[pivot_row][0]], [-value for value in work[pivot_row][1]])
            pivots.append(pivot_row)
            pivot_row += 1
    null_basis = tuple(tuple(row[1]) for row in work if not any(row[0]))
    particular = [0] * width
    remainder = list(targets)
    for pivot in pivots:
        left = work[pivot][0]
        column = next(index for index, value in enumerate(left) if value)
        if remainder[column] % left[column]:
            return None
        factor = remainder[column] // left[column]
        if factor:
            remainder = [a - factor * b for a, b in zip(remainder, left)]
            particular = [a + factor * b for a, b in zip(particular, work[pivot][1])]
    if any(remainder):
        return None
    return tuple(particular), null_basis


def _lattice_box_points(
    particular: tuple[int, ...],
    null_basis: tuple[tuple[int, ...], ...],
    options: tuple[range, ...],
) -> tuple[tuple[int, ...], ...] | None:
    """Return every ``n = particular + sum(t_i * null_basis_i)`` lying in the per-row integer box.

    Returns ``None`` when the bounded coset enumeration would exceed the solver cap.
    """
    if any(not option for option in options):
        return ()
    lows = tuple(option[0] for option in options)
    highs = tuple(option[-1] for option in options)
    if not null_basis:
        point = particular
        return (point,) if all(low <= value <= high for value, low, high in zip(point, lows, highs)) else ()
    dimension = len(null_basis)
    columns = tuple(tuple(vector[row] for vector in null_basis) for row in range(len(particular)))
    gram = tuple(
        tuple(sum(a * b for a, b in zip(null_basis[i], null_basis[j])) for j in range(dimension))
        for i in range(dimension)
    )
    pseudo_inverse = _rational_inverse(tuple(tuple(Fraction(value) for value in row) for row in gram))
    # ``left = (K^T K)^-1 K^T`` maps a box point back to exact coset coefficients, so each
    # coefficient's finite range follows from the box corners componentwise.
    ranges: list[range] = []
    for index in range(dimension):
        projector = tuple(
            sum(pseudo_inverse[index][j] * null_basis[j][row] for j in range(dimension))
            for row in range(len(particular))
        )
        low_sum = Fraction(0)
        high_sum = Fraction(0)
        for row, coefficient in enumerate(projector):
            first = coefficient * (lows[row] - particular[row])
            second = coefficient * (highs[row] - particular[row])
            low_sum += min(first, second)
            high_sum += max(first, second)
        ranges.append(range(math.floor(low_sum), math.floor(high_sum) + 1))
    if math.prod(len(item) for item in ranges) > _MAX_SOLVER_BRANCHES:
        return None
    points: list[tuple[int, ...]] = []
    for coefficients in itertools.product(*ranges):
        point = tuple(
            particular[row] + sum(coefficients[index] * columns[row][index] for index in range(dimension))
            for row in range(len(particular))
        )
        if all(low <= value <= high for value, low, high in zip(point, lows, highs)):
            points.append(point)
    return tuple(sorted(points))


def _exact_modular_solution(
    matrix: tuple[tuple[Fraction, ...], ...],
    constants: tuple[Fraction, ...],
    options: tuple[range, ...],
) -> tuple[tuple[Fraction, ...], Fraction, bool] | None:
    """Return the lexicographically first exact boxed solution, or ``None`` to use the fallback.

    Only the integer wraps ``n`` that keep ``n - c`` in ``col(M)`` can yield an exact solution;
    they form a lattice coset enumerated inside the finite per-row box, in the same deterministic
    order the product enumeration visits, so the first-hit result is preserved.
    """
    diophantine_rows, diophantine_targets = _integer_consistency_system(matrix, constants)
    solved = _integer_diophantine(diophantine_rows, diophantine_targets, len(options))
    if solved is None:
        return None
    particular, null_basis = solved
    points = _lattice_box_points(particular, null_basis, options)
    if points is None:
        return None
    for integers in points:
        rhs = tuple(Fraction(integer) - constant for integer, constant in zip(integers, constants))
        solution = _linear_solve(matrix, rhs)
        if solution is not None:
            return solution, Fraction(0), True
    return None


def _solve_modular(equations: tuple[_Equation, ...]) -> tuple[tuple[Fraction, ...], Fraction, bool] | None:
    """Solve exact modular rows, then return the best exact rational least square fit."""
    if not equations:
        return (), Fraction(0), True
    matrix = tuple(row for equation in equations for row in equation.matrix)
    constants = tuple(value for equation in equations for value in equation.constant)
    options = tuple(_integer_options(row, constant) for row, constant in zip(matrix, constants))
    exact = _exact_modular_solution(matrix, constants, options)
    if exact is not None:
        return exact
    # Noisy inputs have no exact wrap, so fall back to the capped least-squares sweep over the full
    # per-row product.  The exact path above already ran unconditionally, so only this sweep is capped.
    branches = math.prod(len(item) for item in options)
    if branches > _MAX_SOLVER_BRANCHES:
        raise ValueError("exact modular lift solver branch cap exceeded")
    best: tuple[tuple[Fraction, ...], Fraction, bool] | None = None
    for integers in itertools.product(*(tuple(item) for item in options)):
        rhs = tuple(Fraction(integer) - constant for integer, constant in zip(integers, constants))
        solution = _linear_solve(matrix, rhs)
        if solution is not None:
            return solution, Fraction(0), True
        approximation = _least_squares(matrix, rhs)
        if approximation is None:
            continue
        residual = max(
            (
                abs(
                    sum(row[column] * approximation[column] for column in range(len(approximation)))
                    + constant
                    - integer
                )
                for row, constant, integer in zip(matrix, constants, integers)
            ),
            default=Fraction(0),
        )
        if best is None or residual < best[1]:
            best = (approximation, residual, False)
    return best


def _solve_for_transform(
    equations: tuple[_Equation, ...], transform: SubgroupTransform
) -> tuple[tuple[Fraction, ...], Fraction, bool] | None:
    try:
        return _solve_modular(equations)
    except ValueError as error:
        if "branch cap exceeded" not in str(error):
            raise
        raise ValueError(
            f"exact modular lift solver branch cap exceeded for "
            f"{transform.parent.setting} -> {transform.subgroup.setting}"
        ) from error


def _shift_basis(structure: ASUStructure) -> tuple[tuple[Fraction, ...], ...]:
    record = structure.spacegroup.it_number
    from httk.atomistic import data

    return tuple(
        tuple(Fraction(value) for value in vector)
        for vector in data.spacegroup_subgroup_record(record)["continuous_normalizer"]["basis_vectors"]
    )


def _equation(
    parent_position: Any,
    piece: Any,
    child_orbit: _Orbit,
    child_branch: int,
    shift_basis: tuple[tuple[Fraction, ...], ...],
) -> _Equation:
    parent_branch = parent_position.representative
    parent_matrix = _matrix(parent_branch.operation.matrix)
    parent_vector = _vector(parent_branch.operation.vector)
    piece_matrix = _matrix(piece.operation.matrix)
    piece_vector = _vector(piece.operation.vector)
    image_matrix = _matmul(piece_matrix, parent_matrix)
    free = parent_branch.free
    coefficients = tuple(tuple(image_matrix[row][column] for column in free) for row in range(3))
    shift_columns = tuple(tuple(-vector[row] for vector in shift_basis) for row in range(3))
    matrix = tuple(coefficients[row] + shift_columns[row] for row in range(3))
    image_constant = tuple(value + piece_vector[row] for row, value in enumerate(_matvec(piece_matrix, parent_vector)))
    observed = _vector(child_orbit.position.branches[child_branch].coordinate(child_orbit.site.free_params))
    return _Equation(matrix, tuple(a - b for a, b in zip(image_constant, observed)))


def _orbit_distance(predicted: FracVector, observed: tuple[FracVector, ...], cell: Cell) -> tuple[float, Fraction]:
    best_distance: float | None = None
    best_fraction = Fraction(0)
    for target in observed:
        difference = FracVector(_wrapped_tuple(predicted - target))
        distance = math.sqrt(_cartesian_distance_squared(difference, cell))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_fraction = max((abs(value) for value in difference.to_fractions()), default=Fraction(0))
    assert best_distance is not None
    return best_distance, best_fraction


def _validate_candidate(
    candidate: _Candidate,
    solution: tuple[Fraction, ...],
    parent_position: Any,
    pieces: tuple[Any, ...],
    orbits: tuple[_Orbit, ...],
    shift_basis: tuple[tuple[Fraction, ...], ...],
    translation_cosets: tuple[FracVector, ...],
    cell: Cell,
    tolerance: float,
    parent_count: int,
) -> tuple[float, Fraction] | None:
    shift = _matvec(_transpose(shift_basis), solution[parent_count:]) if shift_basis else (Fraction(0),) * 3
    predicted_points: list[FracVector] = []
    observed_points = [point for _, orbit_index in candidate.pieces for point in orbits[orbit_index].coordinates]
    for piece_index, _ in candidate.pieces:
        piece = pieces[piece_index]
        for branch in parent_position.branches:
            parent_point = branch.coordinate(solution[:parent_count])
            image = piece.operation.apply(parent_point) - FracVector(shift)
            predicted_points.extend((image + coset).normalize() for coset in translation_cosets)
    worst_distance = 0.0
    worst_fraction = Fraction(0)
    for predicted in predicted_points:
        distance, residual = _orbit_distance(predicted, tuple(observed_points), cell)
        if distance > tolerance:
            return None
        worst_distance = max(worst_distance, distance)
        worst_fraction = max(worst_fraction, residual)
    for observed in observed_points:
        distance, residual = _orbit_distance(observed, tuple(predicted_points), cell)
        if distance > tolerance:
            return None
        worst_distance = max(worst_distance, distance)
        worst_fraction = max(worst_fraction, residual)
    return worst_distance, worst_fraction


def _multiplicity_possible(structure: ASUStructure, transform: SubgroupTransform) -> bool:
    targets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for site in structure.wyckoff_sites:
        targets[site.species][site.wyckoff] += 1
    labels = tuple(sorted({label for target in targets.values() for label in target}))
    for target in targets.values():
        need = tuple(target[label] for label in labels)
        vectors = []
        for pieces in transform.splittings.values():
            counts: defaultdict[str, int] = defaultdict(int)
            for piece in pieces:
                counts[piece.letter] += 1
            vectors.append(tuple(counts[label] for label in labels))
        vectors = [vector for vector in vectors if any(vector)]

        seen: set[tuple[int, ...]] = set()

        def reachable(remaining: tuple[int, ...], *, _seen=seen, _vectors=vectors) -> bool:
            if not any(remaining):
                return True
            if remaining in _seen:
                return False
            _seen.add(remaining)
            for vector in _vectors:
                if all(value <= residual for value, residual in zip(vector, remaining)) and reachable(
                    tuple(residual - value for residual, value in zip(remaining, vector))
                ):
                    return True
            return False

        if not reachable(need):
            return False
    return True


def _metric_requirements(system: str) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, Fraction], ...]]:
    right_angle = Fraction(90)
    if system == "monoclinic":
        return (), ((0, right_angle), (2, right_angle))
    if system == "orthorhombic":
        return (), ((0, right_angle), (1, right_angle), (2, right_angle))
    if system == "tetragonal":
        return ((0, 1),), ((0, right_angle), (1, right_angle), (2, right_angle))
    if system == "cubic":
        return ((0, 1), (0, 2)), ((0, right_angle), (1, right_angle), (2, right_angle))
    if system in ("trigonal", "hexagonal"):
        # Every tabulated trigonal/hexagonal parent is reached in its hexagonal-axes standard
        # setting (all trigonal R groups are the ":H" settings), so the metric constraint is
        # a=b with alpha=beta=90, gamma=120. No rhombohedral-axes (":R") parent occurs.
        return ((0, 1),), ((0, right_angle), (1, right_angle), (2, Fraction(120)))
    return (), ()


def _cell_for_transform(structure: ASUStructure, transform: SubgroupTransform, tolerance: float) -> _MetricCell | None:
    child_matrix = transform.operation.matrix.T()
    parent_basis = SurdVector(child_matrix.inv()) * structure.cell.basis
    measured = Cell(parent_basis, precision=structure.cell.precision, periodicity=structure.cell.periodicity)
    equal_lengths, fixed_angles = _metric_requirements(transform.parent.crystal_system)
    if not equal_lengths and not fixed_angles:
        return _MetricCell(measured, 0.0, Fraction(0))
    metric = measured.metric()
    exact = all(metric._element((pair[0], pair[0])) == metric._element((pair[1], pair[1])) for pair in equal_lengths)
    exact = exact and all(measured.angles[index] == angle for index, angle in fixed_angles)
    if exact:
        return _MetricCell(measured, 0.0, Fraction(0))

    lengths = measured.lengths
    cartesian = 0.0
    fractional = Fraction(0)
    for first, second in equal_lengths:
        difference = abs(float(lengths[first]) - float(lengths[second]))
        cartesian = max(cartesian, difference)
        first_value = (
            lengths[first]._rational_fraction() if lengths[first].is_rational else lengths[first].to_fractions_approx()
        )
        second_value = (
            lengths[second]._rational_fraction()
            if lengths[second].is_rational
            else lengths[second].to_fractions_approx()
        )
        fractional = max(fractional, abs(first_value - second_value) / max(first_value, second_value, Fraction(1)))
    for index, angle in fixed_angles:
        measured_angle = measured.angles[index]
        if measured_angle == angle:
            continue
        other = ((1, 2), (0, 2), (0, 1))[index]
        cartesian = max(
            cartesian,
            min(float(lengths[other[0]]), float(lengths[other[1]]))
            * abs(math.sin(math.radians(float(measured_angle - angle)))),
        )
        fractional = max(fractional, abs(measured_angle - angle) / Fraction(180))
    if cartesian > tolerance:
        return None
    if any(not length.is_rational for length in lengths):
        return None

    params = [length._rational_fraction() for length in lengths] + list(measured.angles)
    for first, second in equal_lengths:
        params[second] = params[first]
    for index, angle in fixed_angles:
        params[3 + index] = angle
    snapped = Cell(
        CellParams(params).basis,
        precision=structure.cell.precision,
        periodicity=structure.cell.periodicity,
    )
    return _MetricCell(snapped, cartesian, fractional)


def _parent_charge(structure: ASUStructure, transform: SubgroupTransform) -> Fraction | None:
    """Undo descent's exact child/parent content scaling for one hop."""
    if structure.charge is None:
        return None
    multiplier = abs(transform.operation.determinant())
    if not multiplier:
        raise ValueError(f"singular subgroup transform {transform.parent.setting} -> {transform.subgroup.setting}")
    return structure.charge / multiplier


def _translation_cosets(transform: SubgroupTransform) -> tuple[FracVector, ...]:
    """Return child-coordinate representatives of parent-lattice translations."""
    matrix = transform.operation.matrix.T().inv()
    return finite_translation_cosets(tuple(FracVector(row) for row in matrix.to_fractions()))


def _candidate_list(
    structure: ASUStructure,
    transform: SubgroupTransform,
    tolerance: float,
    parent_cell: Cell,
) -> tuple[_Candidate, ...]:
    orbits = tuple(
        _Orbit(
            index,
            site,
            structure.spacegroup.wyckoff_position(site.wyckoff),
            tuple(
                FracVector(point).normalize()
                for point in structure.spacegroup.wyckoff_position(site.wyckoff).coordinates(site.free_params)
            ),
        )
        for index, site in enumerate(structure.wyckoff_sites)
    )
    shift_vectors = _shift_basis(structure)
    translation_cosets = _translation_cosets(transform)
    parent_labels = tuple(sorted(transform.splittings))
    target_counts: dict[str, int] = defaultdict(int)
    for orbit in orbits:
        target_counts[orbit.site.wyckoff] += 1
    candidates: list[_Candidate] = []
    for parent_label in parent_labels:
        parent_position = transform.parent.wyckoff_position(parent_label)
        pieces = transform.splittings[parent_label]
        piece_counts: dict[str, int] = defaultdict(int)
        for piece in pieces:
            piece_counts[piece.letter] += 1
        if any(label not in target_counts for label in piece_counts):
            continue
        piece_labels = {piece.letter for piece in pieces}
        species_values = sorted({orbit.site.species for orbit in orbits})
        for species in species_values:
            choices = [
                tuple(
                    orbit.index
                    for orbit in orbits
                    if orbit.site.species == species and orbit.site.wyckoff in piece_labels
                )
                for piece in pieces
            ]
            if any(not choice for choice in choices):
                continue
            order = sorted(range(len(pieces)), key=lambda index: (len(choices[index]), index))
            selected: list[tuple[int, int]] = []
            equations: list[_Equation] = []

            def visit(
                depth: int,
                *,
                _order=order,
                _selected=selected,
                _equations=equations,
                _pieces=pieces,
                _parent_label=parent_label,
                _species=species,
                _parent_position=parent_position,
                _choices=choices,
            ) -> None:
                if depth == len(_order):
                    selected_labels = [orbits[index].site.wyckoff for _, index in _selected]
                    if sorted(selected_labels) != sorted(piece.letter for piece in _pieces):
                        return
                    candidate = _Candidate(
                        _parent_label,
                        _species,
                        tuple(sorted(_selected)),
                        tuple(_equations),
                        frozenset(index for _, index in _selected),
                    )
                    solved = _solve_for_transform(candidate.equations, transform)
                    if solved is not None:
                        solution, _, _ = solved
                        check = _validate_candidate(
                            candidate,
                            solution,
                            _parent_position,
                            _pieces,
                            orbits,
                            shift_vectors,
                            translation_cosets,
                            structure.cell,
                            tolerance,
                            _parent_position.free_count,
                        )
                        if check is not None:
                            candidates.append(candidate)
                    return
                piece_index = _order[depth]
                for orbit_index in _choices[piece_index]:
                    _selected.append((piece_index, orbit_index))
                    # One anchor equation per split piece is enough; full branch validation
                    # below checks the complete orbit and removes false modular matches.
                    _equations.append(
                        _equation(_parent_position, _pieces[piece_index], orbits[orbit_index], 0, shift_vectors)
                    )
                    visit(depth + 1)
                    _equations.pop()
                    _selected.pop()

            visit(0)
    return tuple(candidates)


def _embed_equation(
    equation: _Equation, parent_offset: int, parent_count: int, total_parent: int, shift_dim: int
) -> _Equation:
    """Place one per-parent equation into a complete-cover variable vector."""
    width = total_parent + shift_dim
    rows = []
    for row in equation.matrix:
        embedded = [Fraction(0)] * width
        embedded[parent_offset : parent_offset + parent_count] = row[:parent_count]
        embedded[total_parent:] = row[parent_count:]
        rows.append(tuple(embedded))
    return _Equation(tuple(rows), equation.constant)


def _solve_with_fixed_shift(
    equations: tuple[_Equation, ...], total_parent: int, shift_dim: int, transform: SubgroupTransform
) -> tuple[Fraction, ...] | None:
    """Solve a second deterministic gauge when continuous shift freedom exists."""
    if not shift_dim:
        return None
    coefficients = tuple(Fraction(1, 2) if index == 0 else Fraction(0) for index in range(shift_dim))
    fixed: list[_Equation] = []
    for equation in equations:
        constants = tuple(
            value + sum(row[total_parent + index] * coefficients[index] for index in range(shift_dim))
            for row, value in zip(equation.matrix, equation.constant)
        )
        fixed.append(_Equation(tuple(row[:total_parent] for row in equation.matrix), constants))
    solved = _solve_for_transform(tuple(fixed), transform)
    return None if solved is None else solved[0] + coefficients


def _canonical_result_key(result: LiftResult) -> tuple[Any, ...]:
    sites = tuple(
        sorted(
            (site.species, site.wyckoff, tuple(site.free_params.to_fractions())) for site in result.asu.wyckoff_sites
        )
    )
    path_key = tuple((transform.index, transform.subgroup_type) for transform in result.path)
    metric = result.asu.cell.metric()
    gram = tuple(metric._element((row, column)) for row in range(3) for column in range(3))
    return (sites, tuple(result.shift.to_fractions()), gram, path_key)


def _lift_transform(structure: ASUStructure, transform: SubgroupTransform, tolerance: float) -> tuple[LiftResult, ...]:
    if not _multiplicity_possible(structure, transform):
        return ()
    metric_cell = _cell_for_transform(structure, transform, tolerance)
    if metric_cell is None:
        return ()
    parent_cell = metric_cell.cell
    candidates = _candidate_list(structure, transform, tolerance, parent_cell)
    if not candidates:
        return ()
    by_orbit: dict[int, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        for orbit in candidate.covered:
            by_orbit[orbit].append(index)
    all_orbits = frozenset(range(len(structure.wyckoff_sites)))
    shift_vectors = _shift_basis(structure)
    translation_cosets = _translation_cosets(transform)
    shift_dim = len(shift_vectors)
    orbits = tuple(
        _Orbit(
            index,
            site,
            structure.spacegroup.wyckoff_position(site.wyckoff),
            tuple(
                FracVector(point).normalize()
                for point in structure.spacegroup.wyckoff_position(site.wyckoff).coordinates(site.free_params)
            ),
        )
        for index, site in enumerate(structure.wyckoff_sites)
    )
    results: list[LiftResult] = []
    used: set[int] = set()
    chosen: list[_Candidate] = []

    def search(remaining: frozenset[int]) -> None:
        if not remaining:
            offsets: list[tuple[int, int]] = []
            total_parent = 0
            for candidate in chosen:
                count = transform.parent.wyckoff_position(candidate.parent_letter).free_count
                offsets.append((total_parent, count))
                total_parent += count
            equations = tuple(
                _embed_equation(equation, offsets[index][0], offsets[index][1], total_parent, shift_dim)
                for index, candidate in enumerate(chosen)
                for equation in candidate.equations
            )
            solved = _solve_for_transform(equations, transform)
            if solved is None:
                return
            solution, _, _ = solved
            parent_parameters: list[tuple[str, FracVector, str]] = []
            for candidate, (offset, count) in zip(chosen, offsets, strict=True):
                parent_parameters.append(
                    (candidate.parent_letter, FracVector(solution[offset : offset + count]), candidate.species)
                )
            sites = tuple(WyckoffSite(letter, params, species) for letter, params, species in parent_parameters)
            try:
                asu = ASUStructure(
                    parent_cell,
                    transform.parent,
                    sites,
                    structure.species,
                    transform=SettingTransform.identity(),
                    coordinate_precision=structure.coordinate_precision,
                    charge=_parent_charge(structure, transform),
                )
            except ValueError:
                return
            shift = _matvec(_transpose(shift_vectors), solution[total_parent:]) if shift_vectors else (Fraction(0),) * 3
            residual = metric_cell.fractional_deviation
            for candidate, (offset, count) in zip(chosen, offsets, strict=True):
                position = transform.parent.wyckoff_position(candidate.parent_letter)
                check = _validate_candidate(
                    candidate,
                    solution[offset : offset + count] + solution[total_parent:],
                    position,
                    transform.splittings[candidate.parent_letter],
                    orbits,
                    shift_vectors,
                    translation_cosets,
                    structure.cell,
                    tolerance,
                    count,
                )
                if check is None:
                    return
                residual = max(residual, check[1])
            results.append(LiftResult(asu, transform.parent, (transform,), FracVector(shift), residual))
            alternate = _solve_with_fixed_shift(equations, total_parent, shift_dim, transform)
            if alternate is not None and alternate != solution:
                alternate_sites = []
                for candidate, (offset, count) in zip(chosen, offsets, strict=True):
                    alternate_sites.append(
                        WyckoffSite(
                            candidate.parent_letter,
                            FracVector(alternate[offset : offset + count]),
                            candidate.species,
                        )
                    )
                try:
                    alternate_asu = ASUStructure(
                        parent_cell,
                        transform.parent,
                        alternate_sites,
                        structure.species,
                        transform=SettingTransform.identity(),
                        coordinate_precision=structure.coordinate_precision,
                        charge=_parent_charge(structure, transform),
                    )
                except ValueError:
                    alternate_asu = None
                if alternate_asu is not None:
                    alternate_shift = _matvec(_transpose(shift_vectors), alternate[total_parent:])
                    alternate_residual = metric_cell.fractional_deviation
                    valid = True
                    for candidate, (offset, count) in zip(chosen, offsets, strict=True):
                        check = _validate_candidate(
                            candidate,
                            alternate[offset : offset + count] + alternate[total_parent:],
                            transform.parent.wyckoff_position(candidate.parent_letter),
                            transform.splittings[candidate.parent_letter],
                            orbits,
                            shift_vectors,
                            translation_cosets,
                            structure.cell,
                            tolerance,
                            count,
                        )
                        if check is None:
                            valid = False
                            break
                        alternate_residual = max(alternate_residual, check[1])
                    if valid:
                        results.append(
                            LiftResult(
                                alternate_asu,
                                transform.parent,
                                (transform,),
                                FracVector(alternate_shift),
                                alternate_residual,
                            )
                        )
            return
        pivot = min(remaining, key=lambda item: sum(index not in used for index in by_orbit[item]))
        for candidate_index in by_orbit[pivot]:
            if candidate_index in used:
                continue
            candidate = candidates[candidate_index]
            if not candidate.covered <= remaining:
                continue
            used.add(candidate_index)
            chosen.append(candidate)
            search(remaining - candidate.covered)
            chosen.pop()
            used.remove(candidate_index)

    search(all_orbits)
    deduplicated: dict[tuple[Any, ...], LiftResult] = {}
    for result in results:
        deduplicated.setdefault(_canonical_result_key(result), result)
    return tuple(deduplicated[key] for key in sorted(deduplicated))


def _apply_normalizer(structure: ASUStructure, record: dict[str, Any]) -> ASUStructure | None:
    """Apply one exact child-setting normalizer and rematch its Wyckoff sites."""
    return _apply_normalizer_operation(structure, AffineOperation.from_record(record))


def _apply_normalizer_operation(structure: ASUStructure, operation: AffineOperation) -> ASUStructure | None:
    sites: list[WyckoffSite] = []
    for site in structure.wyckoff_sites:
        position = structure.spacegroup.wyckoff_position(site.wyckoff)
        original = tuple(FracVector(point).normalize() for point in position.coordinates(site.free_params))
        transformed = tuple(operation.apply_wrapped(point) for point in original)
        matches = [structure.spacegroup.identify_wyckoff(point) for point in transformed]
        if not matches or any(match is None for match in matches):
            return None
        first_match = matches[0]
        if first_match is None:
            return None
        first_position, first_parameters = first_match
        if any(match[0].letter != first_position.letter for match in matches[1:] if match is not None):
            return None
        expected = {
            tuple(FracVector(point).normalize().to_fractions())
            for point in first_position.coordinates(first_parameters)
        }
        actual = {tuple(point.to_fractions()) for point in transformed}
        if expected != actual:
            return None
        sites.append(WyckoffSite(first_position.letter, first_parameters, site.species))
    basis = SurdVector(operation.matrix.T().inv()) * structure.cell.basis
    try:
        return ASUStructure(
            Cell(basis, precision=structure.cell.precision, periodicity=structure.cell.periodicity),
            structure.spacegroup,
            sites,
            structure.species,
            transform=SettingTransform.identity(),
            coordinate_precision=structure.coordinate_precision,
            charge=structure.charge,
        )
    except ValueError:
        return None


def _normalizer_retries(structure: ASUStructure, target: Spacegroup, tolerance: float) -> tuple[LiftResult, ...]:
    try:
        record = data.affine_normalizer_coset_record(structure.spacegroup.hall_entry)
    except KeyError:
        return ()
    transforms = subgroup_transforms(target, structure.spacegroup)
    results: list[LiftResult] = []
    images: dict[tuple[Any, ...], ASUStructure | None] = {}
    for transform in transforms:
        if not _multiplicity_possible(structure, transform):
            continue
        for coset in record.get("affine_normalizer_cosets", ()):
            if target.crystal_system not in coset["compatible_systems"]:
                continue
            affine = coset["affine_transformation"]
            image_key = (tuple(tuple(row) for row in affine["matrix"]), tuple(affine["vector"]))
            if image_key not in images:
                images[image_key] = _apply_normalizer(structure, coset)
            image = images[image_key]
            if image is None:
                continue
            matches = tuple(
                result
                for result in _lift_transform(image, transform, tolerance)
                if result.path and result.path[0] == transform
            )
            operation = AffineOperation.from_record(coset)
            correction = transform.operation * operation.inverse() * transform.operation.inverse()
            for match in matches:
                restored = _apply_normalizer_operation(match.asu, correction)
                if restored is None:
                    continue
                results.append(LiftResult(restored, match.spacegroup, match.path, match.shift, match.residual))
    deduplicated: dict[tuple[Any, ...], LiftResult] = {}
    for result in results:
        deduplicated.setdefault(_canonical_result_key(result), result)
    return tuple(deduplicated[key] for key in sorted(deduplicated))


def _raw_lifts(structure: ASUStructure, target: Spacegroup, tolerance: float) -> tuple[LiftResult, ...]:
    if structure.spacegroup.crystal_system not in COMPATIBLE_CRYSTAL_SYSTEMS[target.crystal_system]:
        return ()
    results: list[LiftResult] = []
    for transform in subgroup_transforms(target, structure.spacegroup):
        results.extend(_lift_transform(structure, transform, tolerance))
    deduplicated: dict[tuple[Any, ...], LiftResult] = {}
    for result in results:
        deduplicated.setdefault(_canonical_result_key(result), result)
    return tuple(deduplicated[key] for key in sorted(deduplicated))


def backward_lift(
    structure: ASUStructure,
    supergroup: Spacegroup | int,
    *,
    tolerance: float | None = None,
) -> tuple[LiftResult, ...]:
    """Return all exact or tolerance-accepted lifts into one minimal supergroup.

    :param structure: The child-group asymmetric unit to lift.
    :param supergroup: The one-hop parent space group or IT number.
    :param tolerance: Cartesian acceptance tolerance, or the recognition-derived default.
    :return: Distinct parent representations in table order and canonical order.
    :raises ValueError: If the input is unsupported or the target is not one hop above it.

    A bounded normalizer retry applies tabulated cosets to child fractional coordinates and maps
    successful results back with the exact inverse, in tabulated coset order.
    """
    if not isinstance(structure, ASUStructure):
        raise TypeError(f"expected ASUStructure, got {type(structure).__name__}")
    require_full_periodicity(structure.cell, "backward_lift")
    if any(site.moment is not None for site in structure.wyckoff_sites):
        raise ValueError("backward_lift does not support structures with site moments")
    if structure.assemblies is not None:
        raise ValueError("backward_lift does not support structures with assemblies")
    if structure.molecular:
        raise ValueError("backward_lift does not support molecular structures")
    current = _standard_input(structure)
    target = supergroup if isinstance(supergroup, Spacegroup) else Spacegroup.standard(supergroup)
    if target.it_number not in minimal_supergroups(current.spacegroup):
        raise ValueError(f"space group {target.setting} is not a minimal supergroup of {current.spacegroup.setting}")
    accepted_tolerance = structure_tolerance(current) if tolerance is None else float(tolerance)
    results = list(_raw_lifts(current, target, accepted_tolerance))
    results.extend(_normalizer_retries(current, target, accepted_tolerance))
    deduplicated: dict[tuple[Any, ...], LiftResult] = {}
    for result in results:
        deduplicated.setdefault(_canonical_result_key(result), result)
    return tuple(deduplicated[key] for key in sorted(deduplicated))


def lift_candidates(structure: ASUStructure, *, tolerance: float | None = None) -> tuple[LiftResult, ...]:
    """Return all one-hop parent lifts in deterministic order.

    :param structure: The child-group asymmetric unit to lift.
    :param tolerance: Cartesian acceptance tolerance, or the recognition-derived default.
    :return: Results ordered by parent IT number, table order, and exact key.
    """
    parents = minimal_supergroups(structure.spacegroup)
    results: list[tuple[int, int, LiftResult]] = []
    for parent_number in parents:
        transforms = subgroup_transforms(parent_number, structure.spacegroup)
        table_indices = {transform: index for index, transform in enumerate(transforms)}
        for result in backward_lift(structure, parent_number, tolerance=tolerance):
            if result.path and result.path[0] in table_indices:
                results.append((parent_number, table_indices[result.path[0]], result))
    return tuple(
        result for _, _, result in sorted(results, key=lambda item: (item[0], item[1], _canonical_result_key(item[2])))
    )


def _highest_lifts(structure: ASUStructure, tolerance: float) -> tuple[LiftResult, ...]:
    results: list[tuple[int, int, LiftResult]] = []
    for parent_number in minimal_supergroups(structure.spacegroup):
        transforms = subgroup_transforms(parent_number, structure.spacegroup)
        indices = {transform: index for index, transform in enumerate(transforms)}
        target = Spacegroup.standard(parent_number)
        try:
            candidates = _raw_lifts(structure, target, tolerance) + _normalizer_retries(structure, target, tolerance)
        except ValueError as error:
            if "branch cap exceeded" not in str(error):
                raise
            # One hopeless parent target may saturate the modular solver's per-row product; skip it
            # so the breadth-first search still explores the remaining minimal supergroups.
            logging.getLogger(__name__).warning(
                "skipping lift into %s: %s", target.setting, error, extra={"context": "symmetry"}
            )
            continue
        for result in candidates:
            if result.path and result.path[0] in indices:
                results.append((parent_number, indices[result.path[0]], result))
    deduplicated: dict[tuple[Any, ...], tuple[int, int, LiftResult]] = {}
    for parent_number, table_index, result in results:
        key = (parent_number, table_index, _canonical_result_key(result))
        deduplicated.setdefault(key, (parent_number, table_index, result))
    return tuple(
        result
        for _, _, result in sorted(
            deduplicated.values(), key=lambda item: (item[0], item[1], _canonical_result_key(item[2]))
        )
    )


def _structure_signature(structure: ASUStructure) -> tuple[Any, ...]:
    sites = []
    for site in structure.wyckoff_sites:
        position = structure.spacegroup.wyckoff_position(site.wyckoff)
        orbit = tuple(sorted(_wrapped_tuple(point) for point in position.coordinates(site.free_params)))
        sites.append((site.species, site.wyckoff, orbit))
    metric = structure.cell.metric()
    gram = tuple(metric._element((row, column)) for row in range(3) for column in range(3))
    return tuple(sorted(sites)), gram


def _terminal_result(
    structure: ASUStructure, path: tuple[SubgroupTransform, ...], shift: FracVector, residual: Fraction
) -> LiftResult:
    return LiftResult(structure, structure.spacegroup, path, shift, residual)


def highest_symmetry(structure: ASUStructure, *, tolerance: float | None = None) -> tuple[LiftResult, ...]:
    """Return all terminal upward lifts reached by breadth-first search.

    :param structure: The starting asymmetric-unit structure.
    :param tolerance: Cartesian acceptance tolerance, or the recognition-derived default.
    :return: Deterministically ordered highest-symmetry representations.
    :raises ValueError: If the input is unsupported or the search cap is exceeded.

    A bounded normalizer retry applies tabulated cosets to child fractional coordinates and maps
    successful results back with the exact inverse, in tabulated coset order.
    """
    if not isinstance(structure, ASUStructure):
        raise TypeError(f"expected ASUStructure, got {type(structure).__name__}")
    require_full_periodicity(structure.cell, "highest_symmetry")
    if any(site.moment is not None for site in structure.wyckoff_sites):
        raise ValueError("highest_symmetry does not support structures with site moments")
    if structure.assemblies is not None:
        raise ValueError("highest_symmetry does not support structures with assemblies")
    if structure.molecular:
        raise ValueError("highest_symmetry does not support molecular structures")
    current = _standard_input(structure)
    accepted_tolerance = structure_tolerance(current) if tolerance is None else float(tolerance)
    queue: list[tuple[ASUStructure, tuple[SubgroupTransform, ...], FracVector, Fraction]] = [
        (current, (), FracVector((0, 0, 0)), Fraction(0))
    ]
    visited: set[tuple[int, tuple[Any, ...]]] = {(current.spacegroup.it_number, _structure_signature(current))}
    terminals: list[LiftResult] = []
    while queue:
        state, path, shift, residual = queue.pop(0)
        lifts = _highest_lifts(state, accepted_tolerance)
        if not lifts:
            terminals.append(_terminal_result(state, path, shift, residual))
            continue
        for result in lifts:
            next_state = result.asu
            key = (next_state.spacegroup.it_number, _structure_signature(next_state))
            if key in visited:
                continue
            if len(visited) >= 10_000:
                # ponytail: bounded state scan; add quotient-graph memoization if tables grow materially.
                raise ValueError(f"highest_symmetry search cap exceeded for {current.spacegroup.setting}")
            visited.add(key)
            queue.append(
                (
                    next_state,
                    path + result.path,
                    result.shift,
                    max(residual, result.residual),
                )
            )
    deduplicated: dict[tuple[Any, ...], LiftResult] = {}
    for result in terminals:
        deduplicated.setdefault((result.spacegroup.it_number, _structure_signature(result.asu), result.path), result)
    return tuple(
        sorted(
            deduplicated.values(),
            key=lambda result: (
                -len(result.spacegroup.symmetry_operations),
                result.spacegroup.it_number,
                result.residual,
                _canonical_result_key(result),
            ),
        )
    )


def canonicalize(structure: ASUStructure, *, tolerance: float | None = None) -> LiftResult:
    """Return the first deterministic highest-symmetry representation.

    :param structure: The structure to canonicalize.
    :param tolerance: Cartesian acceptance tolerance, or the recognition-derived default.
    :return: The canonical terminal lift.
    """
    return highest_symmetry(structure, tolerance=tolerance)[0]


def _supergroup_path(start: int, target: int) -> tuple[int, ...] | None:
    queue: list[tuple[int, tuple[int, ...]]] = [(start, ())]
    visited = {start}
    while queue:
        current, path = queue.pop(0)
        for parent in minimal_supergroups(current):
            if parent == target:
                return path + (parent,)
            if parent not in visited:
                visited.add(parent)
                queue.append((parent, path + (parent,)))
    return None


def rerepresent(
    structure: ASUStructure,
    target: Spacegroup | int,
    *,
    tolerance: float | None = None,
) -> ASUStructure:
    """Express a structure in a reachable subgroup or supergroup setting.

    :param structure: The input asymmetric-unit structure.
    :param target: The target space group or IT number.
    :param tolerance: Cartesian acceptance tolerance for upward lifts.
    :return: The target-group asymmetric unit.
    :raises ValueError: If the target is unrelated, an upward hop has no lift, or a cross-group
        rerepresentation requires descending or lifting a structure with site moments, assemblies,
        or molecular semantics.
    """
    if not isinstance(structure, ASUStructure):
        raise TypeError(f"expected ASUStructure, got {type(structure).__name__}")
    target_group = (target if isinstance(target, Spacegroup) else Spacegroup.standard(target)).standard_setting()
    structure = _standard_input(structure)
    current_number = structure.spacegroup.it_number
    if target_group.it_number == current_number:
        return structure
    if target_group.it_number in subgroup_closure(current_number):
        return subgroup_representation(structure, target_group).asu
    if target_group.it_number in supergroup_closure(current_number):
        path = _supergroup_path(current_number, target_group.it_number)
        if path is None:
            raise ValueError(f"no supergroup path from {structure.spacegroup.setting} to {target_group.setting}")
        current = structure
        for parent_number in path:
            results = backward_lift(current, parent_number, tolerance=tolerance)
            if not results:
                raise ValueError(
                    f"no lift from {current.spacegroup.setting} to {Spacegroup.standard(parent_number).setting}"
                )
            current = min(results, key=lambda result: (result.residual, _canonical_result_key(result))).asu
        return current
    raise ValueError(f"space groups {structure.spacegroup.setting} and {target_group.setting} are unrelated")
