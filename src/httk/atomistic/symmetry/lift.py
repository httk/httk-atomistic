"""Exact one-hop backward lifts through Bärnighausen tables.

The public functions in this module invert one tabulated subgroup descent.  Coordinates,
Wyckoff parameters, affine maps, modular solves, and returned shifts are rational.  A
Cartesian tolerance is used only when accepting a measured structure that is not an exact
solution of the assembled equations. Cell-metric validation covers monoclinic, orthorhombic,
tetragonal, trigonal, hexagonal, and cubic systems; every tabulated trigonal and hexagonal parent
is in a hexagonal-axes standard setting, so their metric constraint is a=b with alpha=beta=90 and
gamma=120. Normalizer retry applies tabulated cosets to child fractional coordinates, maps
successful results back with the exact inverse, and follows tabulated coset order.

On top of the one-hop lift, :func:`highest_symmetry` / :func:`canonicalize` search upward for the
highest-symmetry description of a crystal and return one deterministic, normalizer-canonical
representative.  Every search state is reduced to a normal form that collapses same-group
descriptions of the same crystal: mislabeled special sites are demoted, and the state is quotiented
by the group's continuous- and discrete-Euclidean-normalizer translations and its affine-normalizer
cosets.  A triclinic (SG 1 or 2) entry is first Niggli-reduced so the result is independent of the
input basis choice, and the returned cell is put in the standard orientation of its metric.  The
result is therefore invariant under origin shift, cell-basis choice (relabeling/shear), and site
order for the same crystal, and agrees with direct entry at the crystal's own space group.  For a
P1 / unit-cell start, build the ASU in SG 1 and canonicalize it::

    cell = Cell(((4, 0, 0), (0, 4, 0), (0, 0, 4)))
    sites = [WyckoffSite("a", FracVector((0, 0, 0)), "Cs"),
             WyckoffSite("a", FracVector((Fraction(1, 2),) * 3), "Cl")]
    p1 = ASUStructure(cell, 1, sites, [Species(...), Species(...)])
    result = canonicalize(p1)  # result.spacegroup.it_number == 221

The upward search lifts each state through three fail-only tiers, tried in order and only when the
earlier ones return nothing for a state: (1) the direct tabulated lift; (2) tabulated
affine-normalizer-coset retries; (3) a conventional-cell re-choice.  The third tier exists because
a centred-lattice parent can be presented, in the reduced cell the search carries, in an axis choice
that misses the parent's exact metric class even though the lattice admits a conforming cell -- an
F-centred cubic (NaCl from its Niggli primitive) is the motivating case.  It searches the candidate
parent lattice for a conventional basis meeting the parent metric exactly, derives the implied child
re-expression, and -- when that re-expression is an integer lattice normalizer -- applies it and
lifts the re-expressed child; the descent round-trip gate stays authoritative.  It is inert whenever
an earlier tier succeeds, so it never runs on a normally-climbing (e.g. P-lattice) input.  A centred
child whose conventional cell is an intrinsic supercell of its lattice (the R-centred trigonal case,
Bi-166) yields a non-integer, non-normalizer re-expression and is not landed by this tier.
"""

import itertools
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any

from httk.core import FracVector, SurdVector

from httk.atomistic import data
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.models.structure.asu import ASUStructure, WyckoffSite
from httk.atomistic.models.structure.comparison import same_crystal
from httk.atomistic.symmetry._lattice import finite_translation_cosets
from httk.atomistic.symmetry._periodicity_guard import require_full_periodicity
from httk.atomistic.symmetry._standardization_common import (
    _matrix_column_sum_factor,
    _matrix_row_sum_factor,
    _scaled_precision,
)
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.recognition import (
    _cartesian_distance_squared,
    structure_tolerance,
)
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup
from httk.atomistic.symmetry.subgroups import (
    SubgroupTransform,
    _child_sites,
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
# The noisy least-squares fallback only helps genuine recognition noise: the wrap box that carried an
# accepted noisy solution was <= 16 in the reviewed cases, while the full sweep on the same Bi
# 2-atom P1 -> P-1 hop enters boxes up to 1296 before this cap (and 46656 without it, ~90 s that only
# ever yields a rejected approximation).  The ceiling sits ~3.2x above that measured 1296.  It is a
# heuristic, not a proof: the box grows multiplicatively in the number of rows, so a genuinely noisy
# lift whose wrap box exceeds it would be silently missed -- hence the warning log at the skip.  The
# full cap still guards the exact path.
_MAX_NOISY_SWEEP_BRANCHES = 4_096
_MAX_FOURIER_MOTZKIN_INEQUALITIES = 20_000


@dataclass(frozen=True, slots=True)
class LiftResult:
    """One exact or tolerance-accepted parent representation.

    :param asu: The parent-standard-setting asymmetric unit.
    :param spacegroup: The parent space group in standard setting.
    :param path: Child-first tabulated parent-to-child subgroup transforms used.
    :param shift: The continuous-normalizer origin shift from the final hop, expressed in that
        hop's parent standard frame.  ``path`` and ``shift`` document the lift route; they do not by
        themselves reconstruct ``asu``, since :func:`highest_symmetry` additionally passes each state
        through an unrecorded normal form and canonical orientation.  ``asu`` is authoritative.
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
    if branches > _MAX_NOISY_SWEEP_BRANCHES:
        # No exact wrap exists (the exact path already ran); a noisy match this deep in the wrap box
        # is not a real lift, so fail the candidate cheaply rather than grinding the full sweep.  This
        # can in principle silently drop a genuinely noisy lift in a large box, so record it on the
        # same warning channel the branch-cap skip uses.
        logging.getLogger(__name__).warning(
            "skipping noisy least-squares sweep over %d wrap boxes (cap %d)",
            branches,
            _MAX_NOISY_SWEEP_BRANCHES,
            extra={"context": "symmetry"},
        )
        return None
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

            def visit(
                depth: int,
                *,
                _order=order,
                _selected=selected,
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
                    # The tabulated split maps each child orbit into the parent orbit, but the anchor
                    # need not close on the child's branch-0 representative: a correspondence that only
                    # solves on another branch was silently lost when this was hard-coded to branch 0.
                    # Enumerate every child anchor branch (deterministic product order); any spurious
                    # modular match this admits is removed by the exact descent round trip in
                    # _lift_transform, which is the authoritative correctness gate.
                    # ponytail: full branch product costs O(branches^pieces) solver calls -- negligible
                    # for the one-branch P-lattice majority, but it lengthens the failed-lift search on
                    # high-multiplicity trigonal parents; dedup combos by their solved placement if that
                    # class of input matters.
                    branch_counts = [len(orbits[index].position.branches) for _, index in _selected]
                    for combo in itertools.product(*(range(count) for count in branch_counts)):
                        equations = tuple(
                            _equation(
                                _parent_position,
                                _pieces[piece_index],
                                orbits[orbit_index],
                                combo[position],
                                shift_vectors,
                            )
                            for position, (piece_index, orbit_index) in enumerate(_selected)
                        )
                        candidate = _Candidate(
                            _parent_label,
                            _species,
                            tuple(sorted(_selected)),
                            equations,
                            frozenset(index for _, index in _selected),
                        )
                        solved = _solve_for_transform(candidate.equations, transform)
                        if solved is None:
                            continue
                        check = _validate_candidate(
                            candidate,
                            solved[0],
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
                    visit(depth + 1)
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


@lru_cache(maxsize=100_000)
def _canonical_representative_cached(
    spacegroup: Spacegroup, wyckoff: str, free_params: tuple[Fraction, ...]
) -> tuple[str, tuple[Fraction, ...]]:
    """Cached core of :func:`_canonical_representative`, keyed on hashable exact inputs.

    ``Spacegroup`` hashes and compares by setting, so states repeating a group and site share entries.
    """
    position = spacegroup.wyckoff_position(wyckoff)
    points = [FracVector(point).normalize() for point in position.coordinates(FracVector(free_params))]
    least = min(points, key=lambda point: tuple(point.to_fractions()))
    match = spacegroup.identify_wyckoff(least)
    if match is None:
        return wyckoff, free_params
    matched_position, parameters = match
    return matched_position.letter, tuple(parameters.to_fractions())


def _canonical_representative(spacegroup: Spacegroup, site: WyckoffSite) -> tuple[str, Any]:
    """Return ``(letter, free_params)`` derived from the site's lexicographically least orbit point.

    Two same-crystal descriptions can store different points of the same orbit (e.g. ``x`` versus
    ``-x``); re-identifying the least wrapped orbit point makes the reported letter and parameters a
    function of the orbit alone.
    """
    letter, parameters = _canonical_representative_cached(
        spacegroup, site.wyckoff, tuple(site.free_params.to_fractions())
    )
    return letter, FracVector(parameters)


def _site_key(structure: ASUStructure) -> tuple[tuple[str, str, tuple[Fraction, ...]], ...]:
    """Return the exact translation- and orbit-representative-invariant sorted-site key of one ASU.

    The key uses each site's canonical orbit representative, which may differ from the params stored
    on a structure that has not passed :func:`_canonical_sites`, so ``backward_lift``-family ordering
    follows this canonical key rather than the stored params.
    """
    entries: list[tuple[str, str, tuple[Fraction, ...]]] = []
    for site in structure.wyckoff_sites:
        letter, parameters = _canonical_representative(structure.spacegroup, site)
        entries.append((site.species, letter, tuple(parameters.to_fractions())))
    return tuple(sorted(entries))


def _canonical_sites(structure: ASUStructure) -> ASUStructure:
    """Rewrite each site with the deterministic orbit representative from :func:`_canonical_representative`."""
    sites: list[WyckoffSite] = []
    changed = False
    for site in structure.wyckoff_sites:
        letter, parameters = _canonical_representative(structure.spacegroup, site)
        if letter != site.wyckoff or tuple(parameters.to_fractions()) != tuple(site.free_params.to_fractions()):
            changed = True
        sites.append(WyckoffSite(letter, parameters, site.species))
    if not changed:
        return structure
    try:
        return ASUStructure(
            structure.cell,
            structure.spacegroup,
            sites,
            structure.species,
            transform=SettingTransform.identity(),
            coordinate_precision=structure.coordinate_precision,
            charge=structure.charge,
        )
    except ValueError:
        return structure


def _canonical_result_key(result: LiftResult) -> tuple[Any, ...]:
    sites = _site_key(result.asu)
    path_key = tuple((transform.index, transform.subgroup_type) for transform in result.path)
    metric = result.asu.cell.metric()
    gram = tuple(metric._element((row, column)) for row in range(3) for column in range(3))
    return (sites, tuple(result.shift.to_fractions()), gram, path_key)


def _crystals_match_within(left: ASUStructure, right: ASUStructure, tolerance: float) -> bool:
    """Return whether two same-cell crystals agree atom-for-atom within ``tolerance`` (Cartesian).

    A bijective, species-labelled, minimum-image match -- the tolerant counterpart of
    :func:`same_crystal` for the recognition-snapped path, where an accepted lift reproduces the
    child only up to the hop's tolerance rather than exactly.

    The basis-equality guard below is a defensive precondition, not a working branch: the round-trip
    caller (:func:`_round_trip_reproduces`) always passes two structures built on the child's own
    cell, so it never fires there.  It is kept so the tolerant match stays sound if reused with
    mismatched cells, where a Cartesian comparison would be meaningless.
    """
    from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

    left_view = UnitcellStructureView(left)
    right_view = UnitcellStructureView(right)
    if left_view.cell.basis != right_view.cell.basis:
        return False
    left_points = list(zip(left_view.species_at_sites, left_view.sites.reduced_coords.to_fractions()))
    remaining = list(zip(right_view.species_at_sites, right_view.sites.reduced_coords.to_fractions()))
    if len(left_points) != len(remaining):
        return False
    for name, coordinate in left_points:
        match_index: int | None = None
        for index, (other_name, other_coordinate) in enumerate(remaining):
            if other_name != name:
                continue
            difference = FracVector(_wrapped_tuple(FracVector(coordinate) - FracVector(other_coordinate)))
            # ponytail: greedy first-match within tolerance; sites sit far more than one tolerance
            # apart, so the first admissible partner is the only one -- no assignment search needed.
            if math.sqrt(_cartesian_distance_squared(difference, left_view.cell)) <= tolerance:
                match_index = index
                break
        if match_index is None:
            return False
        remaining.pop(match_index)
    return True


def _round_trip_reproduces(
    child: ASUStructure, parent: ASUStructure, transform: SubgroupTransform, tolerance: float
) -> bool:
    """Return whether descending ``parent`` through ``transform`` reproduces ``child``.

    A backward-lift candidate is correct exactly when the descent it claims to invert recovers the
    child crystal.  Descent (:func:`_child_sites`) is exact and already trusted, so this is the
    authoritative correctness gate: the per-orbit distance check in :func:`_validate_candidate` is
    only a cheap pre-filter, because a modular anchor match on any child branch can pass it while
    placing the other sites on a different crystal.  Exact input reproduces the child exactly, so the
    fast path is :func:`same_crystal`; a recognition-snapped input is reproduced only up to the hop's
    Cartesian ``tolerance``, so the fallback matches within it -- one rule, exact when it can be.

    The lift's continuous-normalizer origin freedom (:attr:`LiftResult.shift`, and the residual
    origin a polar or P1 child carries) leaves the descended child at a canonical origin the raw
    input need not share.  Both sides are reduced through :func:`_translation_normal_form` first, so
    the comparison is invariant to exactly that freedom -- a no-op for the non-polar majority, where
    the continuous normalizer is trivial.  It shifts only along continuous directions, so a genuinely
    different crystal from a spurious branch match still cannot be reconciled.
    """
    try:
        rebuilt = ASUStructure(
            child.cell,
            transform.subgroup,
            _child_sites(parent, transform),
            child.species,
            transform=SettingTransform.identity(),
            coordinate_precision=child.coordinate_precision,
            charge=child.charge,
        )
    except ValueError:
        return False
    rebuilt = _translation_normal_form(rebuilt)
    target = _translation_normal_form(child)
    return same_crystal(rebuilt, target) or _crystals_match_within(rebuilt, target, tolerance)


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
    # Authoritative correctness gate: keep only candidates whose exact descent reproduces the child.
    # Branch-free anchoring above admits more modular matches; the ones that do not round-trip -- the
    # unsound lifts branch-0 anchoring used to mask by never proposing them -- are removed here.
    results = [result for result in results if _round_trip_reproduces(structure, result.asu, transform, tolerance)]
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


def _integer_diagonalize(rows: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], list[list[int]]]:
    """Diagonalize an integer ``n x 3`` matrix by unimodular row and column operations.

    Returns the positive diagonal entries ``(d0, d1, d2)`` and the accumulated column transform
    ``V`` (``3 x 3`` unimodular), so that ``M t`` is integral iff ``d_i * (V^-1 t)_i`` is integral;
    a zero diagonal marks a continuous direction (column of ``V``).  Euclidean row/column swaps
    drive each pivot to the gcd, so this terminates on any integer input.
    """
    work = [list(row) for row in rows]
    height = len(work)
    transform = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for pivot in range(3):
        while True:
            if pivot >= height or work[pivot][pivot] == 0:
                swapped = False
                for row in range(pivot, height):
                    for column in range(pivot, 3):
                        if work[row][column]:
                            if row != pivot:
                                work[pivot], work[row] = work[row], work[pivot]
                            if column != pivot:
                                for line in work:
                                    line[pivot], line[column] = line[column], line[pivot]
                                for line in transform:
                                    line[pivot], line[column] = line[column], line[pivot]
                            swapped = True
                            break
                    if swapped:
                        break
                if not swapped:
                    break
            leader = work[pivot][pivot]
            settled = True
            for row in range(pivot + 1, height):
                if work[row][pivot]:
                    factor = work[row][pivot] // leader
                    for column in range(3):
                        work[row][column] -= factor * work[pivot][column]
                    if work[row][pivot]:
                        work[pivot], work[row] = work[row], work[pivot]
                        settled = False
                        break
            if not settled:
                continue
            for column in range(pivot + 1, 3):
                if work[pivot][column]:
                    factor = work[pivot][column] // leader
                    for line in work:
                        line[column] -= factor * line[pivot]
                    for line in transform:
                        line[column] -= factor * line[pivot]
                    if work[pivot][column]:
                        for line in work:
                            line[pivot], line[column] = line[column], line[pivot]
                        for line in transform:
                            line[pivot], line[column] = line[column], line[pivot]
                        settled = False
                        break
            if settled:
                break
    diagonal = tuple(abs(work[index][index]) if index < height else 0 for index in range(3))
    return diagonal, transform


def _ext_gcd(first: int, second: int) -> tuple[int, int, int]:
    """Return ``(g, x, y)`` with ``g = gcd`` positive and ``x*first + y*second = g``."""
    old_r, r = first, second
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    if old_r < 0:
        return -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def _lattice_basis(generators: list[tuple[int, ...]]) -> list[list[int]]:
    """Return a 3x3 integer basis (rows) of the rank-3 sublattice of ``Z^3`` spanned by generators."""
    pivots: list[list[int]] = []
    leads: list[int] = []
    for generator in generators:
        vector = list(generator)
        for pivot, lead in zip(pivots, leads):
            if vector[lead]:
                gcd, left, right = _ext_gcd(pivot[lead], vector[lead])
                combined = [left * pivot[k] + right * vector[k] for k in range(3)]
                reduced = [(vector[lead] // gcd) * pivot[k] - (pivot[lead] // gcd) * vector[k] for k in range(3)]
                pivot[:] = combined
                vector = reduced
        lead = next((k for k in range(3) if vector[k]), None)
        if lead is not None:
            if vector[lead] < 0:
                vector = [-value for value in vector]
            pivots.append(vector)
            leads.append(lead)
            order = sorted(range(len(leads)), key=lambda index: leads[index])
            pivots = [pivots[index] for index in order]
            leads = [leads[index] for index in order]
    basis = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for pivot, lead in zip(pivots, leads):
        basis[lead] = pivot
    return basis


def _translation_lattice(spacegroup: Spacegroup) -> tuple[tuple[Fraction, ...], ...]:
    """Return a 3x3 rational matrix whose columns are a basis of the group's translation lattice.

    The lattice is ``Z^3`` plus the centring vectors -- the translation parts of the ``W = I``
    symmetry operations -- so for a P lattice this is the identity and for a centred lattice it is
    finer than ``Z^3``.
    """
    identity = FracVector.eye((3, 3))
    centrings = [
        tuple(FracVector(operation.vector).normalize().to_fractions())
        for operation in spacegroup.symmetry_operations
        if operation.matrix == identity
    ]
    denominators = [value.denominator for vector in centrings for value in vector]
    scale = math.lcm(*denominators) if denominators else 1
    generators = [tuple(int(value * scale) for value in vector) for vector in centrings]
    generators += [tuple(scale if i == j else 0 for j in range(3)) for i in range(3)]
    rows = _lattice_basis(generators)
    vectors = [tuple(Fraction(value, scale) for value in row) for row in rows]
    return tuple(tuple(vectors[column][row] for column in range(3)) for row in range(3))


_DISCRETE_NORMALIZER_CACHE: dict[str, tuple[tuple[Fraction, ...], ...]] = {}


def _discrete_normalizer_translations(spacegroup: Spacegroup) -> tuple[tuple[Fraction, ...], ...]:
    """Return the finite group of discrete Euclidean-normalizer translations, reps in ``[0, 1)^3``.

    A translation ``t`` normalizes ``G`` iff ``(I - W) t`` lies in the group's translation lattice
    ``T_G`` for every distinct linear part ``W`` (conjugation sends ``(W, w)`` to
    ``(W, w + (I - W) t)``, and ops sharing ``W`` differ only by translations in ``T_G``).  For a
    centred lattice ``T_G`` is finer than ``Z^3``, so the criterion uses ``T_G`` rather than ``Z^3``
    -- otherwise quarter-translations such as ``(1/4, 1/4, 1/4)`` for F-43m are missed.  With ``C``
    a column basis of ``T_G``, the substitution ``t = C u`` turns ``(I - W) t in T_G`` into the
    integral ``M' = C^-1 (I - W) C`` acting on ``u in Z^3``; diagonalizing ``M'`` enumerates the
    finite solution set exactly, mapped back by ``C``.  Continuous null-space directions are excluded
    here -- the continuous-translation quotient owns them.  Always includes the identity.
    """
    key = spacegroup.hall_entry
    cached = _DISCRETE_NORMALIZER_CACHE.get(key)
    if cached is not None:
        return cached
    lattice = _translation_lattice(spacegroup)
    lattice_inverse = _rational_inverse(lattice)
    seen: set[tuple[tuple[int, ...], ...]] = set()
    rows: list[tuple[int, ...]] = []
    for operation in spacegroup.symmetry_operations:
        linear = tuple(tuple(int(value) for value in row) for row in operation.matrix.to_fractions())
        if linear in seen:
            continue
        seen.add(linear)
        i_minus_w = tuple(tuple(Fraction((1 if i == j else 0) - linear[i][j]) for j in range(3)) for i in range(3))
        conjugated = _matmul(_matmul(lattice_inverse, i_minus_w), lattice)
        for row in conjugated:
            assert all(value.denominator == 1 for value in row)
            rows.append(tuple(value.numerator for value in row))
    diagonal, transform = _integer_diagonalize(tuple(rows))
    axes = [[Fraction(step, divisor) for step in range(divisor)] if divisor else [Fraction(0)] for divisor in diagonal]
    representatives: set[tuple[Fraction, ...]] = set()
    for first in axes[0]:
        for second in axes[1]:
            for third in axes[2]:
                coefficients = (first, second, third)
                unit = tuple(
                    sum((transform[row][column] * coefficients[column] for column in range(3)), Fraction(0))
                    for row in range(3)
                )
                representatives.add(
                    tuple(
                        sum((lattice[row][column] * unit[column] for column in range(3)), Fraction(0)) % 1
                        for row in range(3)
                    )
                )
    result = tuple(sorted(representatives))
    _DISCRETE_NORMALIZER_CACHE[key] = result
    return result


def _translation_normal_form(structure: ASUStructure) -> ASUStructure:
    """Return the origin-canonical image under the group's continuous-normalizer translations.

    The continuous normalizer basis spans the directions along which the whole structure may be
    translated while staying a valid same-group description.  Candidate origins are, for every
    expanded orbit point of every site, the pure translation that cancels that point's components
    along the continuous directions; the least exact sorted-site key wins, and the identity
    translation is always a candidate so the result never regresses.  Pure translations leave the
    cell unchanged, so the site key alone is a sound comparison.
    """
    shift_basis = _shift_basis(structure)
    if not shift_basis:
        return structure
    # Every tabulated continuous-normalizer vector is an axis-aligned unit vector, so the continuous
    # directions are simply the axes those vectors point along.
    axes = sorted({index for vector in shift_basis for index in range(3) if vector[index]})
    identity = FracVector.eye((3, 3))
    candidates: set[tuple[Fraction, ...]] = {(Fraction(0), Fraction(0), Fraction(0))}
    for site in structure.wyckoff_sites:
        position = structure.spacegroup.wyckoff_position(site.wyckoff)
        for point in position.coordinates(site.free_params):
            values = FracVector(point).normalize().to_fractions()
            translation = [Fraction(0), Fraction(0), Fraction(0)]
            for index in axes:
                translation[index] = (-values[index]) % 1
            candidates.add(tuple(translation))
    best = structure
    best_key = _site_key(structure)
    for translation in sorted(candidates):
        if not any(translation):
            continue
        image = _apply_normalizer_operation(structure, AffineOperation(identity, FracVector(translation)))
        if image is None:
            continue
        key = _site_key(image)
        if key < best_key:
            best, best_key = image, key
    return best


def _demote_sites(structure: ASUStructure) -> ASUStructure:
    """Re-label any site whose expanded orbit degenerates onto a more-special Wyckoff position.

    A lift can leave an atom on a special coordinate while still carrying a general (or less
    special) Wyckoff letter; its expanded orbit then contains coincident points.  Re-identifying the
    representative coordinate through the exact Wyckoff machinery demotes it to its true most-special
    letter and free params, so equal crystals carry equal site keys.
    """
    demoted: list[WyckoffSite] = []
    changed = False
    for site in structure.wyckoff_sites:
        position = structure.spacegroup.wyckoff_position(site.wyckoff)
        points = [FracVector(point).normalize() for point in position.coordinates(site.free_params)]
        if len({tuple(point.to_fractions()) for point in points}) == len(points):
            demoted.append(site)
            continue
        match = structure.spacegroup.identify_wyckoff(points[0])
        if match is None:
            demoted.append(site)
            continue
        letter_position, parameters = match
        demoted.append(WyckoffSite(letter_position.letter, parameters, site.species))
        changed = True
    if not changed:
        return structure
    try:
        return ASUStructure(
            structure.cell,
            structure.spacegroup,
            demoted,
            structure.species,
            transform=SettingTransform.identity(),
            coordinate_precision=structure.coordinate_precision,
            charge=structure.charge,
        )
    except ValueError:
        return structure


def _normal_form(structure: ASUStructure) -> ASUStructure:
    """Return a deterministic canonical representative of one state within its own space group.

    Sites mislabeled onto a special coordinate are demoted first, then the normalizer quotients are
    collapsed: the candidate images are every affine-normalizer coset crossed with every discrete
    Euclidean-normalizer translation (both including the identity), each followed by the
    continuous-translation quotient.  Every accepted image is the same Cartesian crystal re-expressed
    in the same group -- ``_apply_normalizer_operation`` re-identifies the Wyckoff sites and rejects
    anything that is not -- so collapsing them to the least sorted-site key cannot lose a reachable
    terminal and yields a description-invariant representative.  The least key wins in a deterministic
    order (tabulated coset order, then sorted translations); each site is finally stored at its
    canonical orbit representative.  Basis-choice invariance for the same lattice comes separately,
    from the Niggli reduction of triclinic (SG 1/2) entries in :func:`highest_symmetry`, not from
    this per-group normal form.
    """
    structure = _demote_sites(structure)
    identity_matrix = FracVector.eye((3, 3))
    operations = [AffineOperation.identity()]
    try:
        record = data.affine_normalizer_coset_record(structure.spacegroup.hall_entry)
    except KeyError:
        record = None
    if record is not None:
        system = structure.spacegroup.crystal_system
        operations.extend(
            AffineOperation.from_record(coset)
            for coset in record.get("affine_normalizer_cosets", ())
            if system in coset["compatible_systems"]
        )
    translations = _discrete_normalizer_translations(structure.spacegroup)
    best: ASUStructure | None = None
    best_key: tuple[Any, ...] | None = None
    for operation in operations:
        image = _apply_normalizer_operation(structure, operation)
        if image is None:
            continue
        for translation in translations:
            if any(translation):
                shifted = _apply_normalizer_operation(image, AffineOperation(identity_matrix, FracVector(translation)))
            else:
                shifted = image
            if shifted is None:
                continue
            # A coset or discrete shift can change which origin is canonical, so re-run the
            # continuous-translation quotient on each candidate.
            reduced = _translation_normal_form(shifted)
            if reduced.cell.basis.det().sign() < 0:
                # A det=-1 coset yields a left-handed basis that the final canonical orientation would
                # flip; normalize handedness (inversion is a same-crystal basis change) before keying,
                # so the selected minimum matches the right-handed representative that is returned.
                flipped = _apply_normalizer_operation(
                    reduced, AffineOperation(FracVector(((-1, 0, 0), (0, -1, 0), (0, 0, -1))), (0, 0, 0))
                )
                if flipped is not None:
                    reduced = _translation_normal_form(flipped)
            key = _site_key(reduced)
            if best_key is None or key < best_key:
                best, best_key = reduced, key
    # Store each site at its canonical orbit representative so the returned free params, not just the
    # comparison key, are independent of which orbit point the input happened to carry.
    return _canonical_sites(best) if best is not None else _canonical_sites(structure)


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


def _bilinear(gram: tuple[tuple[Fraction, ...], ...], left: tuple[int, ...], right: tuple[int, ...]) -> Fraction:
    return sum((left[i] * gram[i][j] * right[j] for i in range(3) for j in range(3)), Fraction(0))


def _integer_determinant(rows: tuple[tuple[int, ...], ...]) -> int:
    a, b, c = rows
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])


@lru_cache(maxsize=50_000)
def _search_conventional_basis(
    gram: tuple[tuple[Fraction, ...], ...], system: str
) -> tuple[tuple[int, ...], ...] | None:
    """Return a canonical integer basis of ``gram``'s lattice meeting the system's exact metric.

    The lattice is the candidate parent lattice ``inv(M^T)*B_child`` (as its exact rational gram);
    the returned rows are integer coordinates on it whose cell satisfies the parent crystal system's
    :func:`_metric_requirements` exactly.  Enumeration is a bounded exact short-vector search --
    ``bound`` 5 was what the trigonal Bi case needed in the parameterization these grams use, kept as
    a documented bounded search rather than a proven radius.  The choice is deterministic: least
    ``(abs(det), row-order squared lengths, rows)``, then flipped to a right-handed (positive
    determinant) basis so the applied re-expression is orientation-preserving.  ``None`` when no such
    basis is found in the bound.
    """
    equal_lengths, fixed_angles = _metric_requirements(system)
    if not equal_lengths and not fixed_angles:
        return None
    everything = [n for n in itertools.product(range(-5, 6), repeat=3) if any(n)]
    everything.sort(key=lambda n: (_bilinear(gram, n, n), n))
    norms = sorted({_bilinear(gram, n, n) for n in everything})
    cap = norms[min(len(norms) - 1, 9)]
    short = [n for n in everything if _bilinear(gram, n, n) <= cap]
    right = Fraction(90)
    # Every target system that reaches here has alpha = beta = 90 (parents are orthorhombic or more
    # symmetric), so the c axis is perpendicular to whichever earlier axes those angles name -- a
    # precomputed perpendicular set keeps the innermost loop to genuine candidates instead of all of
    # ``short``, which is what makes the exact search tractable inside the breadth-first search.
    perpendicular = {vector: {other for other in short if _bilinear(gram, vector, other) == 0} for vector in short}
    c_perpendicular_to_a = any(index == 1 and degrees == right for index, degrees in fixed_angles)
    c_perpendicular_to_b = any(index == 0 and degrees == right for index, degrees in fixed_angles)
    basal_angles = tuple((index, degrees) for index, degrees in fixed_angles if index == 2)

    def angle_ok(rows: tuple[tuple[int, ...], ...], index: int, degrees: Fraction) -> bool:
        first, second = ((1, 2), (0, 2), (0, 1))[index]
        product = _bilinear(gram, rows[first], rows[second])
        if degrees == right:
            return product == 0
        length = _bilinear(gram, rows[first], rows[first])
        return 2 * product == -length and length == _bilinear(gram, rows[second], rows[second])

    best: tuple[tuple[Any, ...], tuple[tuple[int, ...], ...]] | None = None
    for a in short:
        length_a = _bilinear(gram, a, a)
        for b in short:
            # Prune on the constraints that involve only the first two axes before the c loop.
            if (0, 1) in equal_lengths and _bilinear(gram, b, b) != length_a:
                continue
            if any(not angle_ok((a, b, a), index, degrees) for index, degrees in basal_angles):
                continue
            candidates: Any = short
            if c_perpendicular_to_a:
                candidates = perpendicular[a]
            if c_perpendicular_to_b:
                candidates = candidates & perpendicular[b] if c_perpendicular_to_a else perpendicular[b]
            for c in candidates:
                rows = (a, b, c)
                if _integer_determinant(rows) == 0:
                    continue
                if not all(
                    _bilinear(gram, rows[i], rows[i]) == _bilinear(gram, rows[j], rows[j]) for i, j in equal_lengths
                ):
                    continue
                if not all(angle_ok(rows, index, degrees) for index, degrees in fixed_angles):
                    continue
                key = (
                    abs(_integer_determinant(rows)),
                    tuple(_bilinear(gram, rows[k], rows[k]) for k in range(3)),
                    rows,
                )
                if best is None or key < best[0]:
                    best = (key, rows)
    if best is None:
        return None
    rows = best[1]
    if _integer_determinant(rows) < 0:
        # The abs(det) key leaves the handedness of the winner undetermined; negate one axis so the
        # returned basis is right-handed and every re-expression built from it preserves orientation.
        rows = (rows[0], rows[1], tuple(-value for value in rows[2]))
    return rows


def _exact_rational_matrix(matrix: Any) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(value for value in row) for row in matrix.to_fractions())


def _matmul3(
    left: tuple[tuple[Fraction, ...], ...], right: tuple[tuple[Fraction, ...], ...]
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(sum((left[i][k] * right[k][j] for k in range(3)), Fraction(0)) for j in range(3)) for i in range(3)
    )


def _recell_lifts(structure: ASUStructure, target: Spacegroup, tolerance: float) -> tuple[LiftResult, ...]:
    """Third, fail-only lift tier: re-choose the child's conventional cell so a metric-rejected hop fits.

    A tabulated transform is rejected by :func:`_cell_for_transform` when the child, carried through
    the breadth-first search in a reduced cell, presents the parent lattice in an axis choice that
    misses the parent's exact metric class -- even though the lattice does admit a conforming cell.
    For each such transform this searches the candidate parent lattice for a conventional basis
    meeting the parent metric exactly, derives the implied child re-expression ``U = M^T*N*inv(M^T)``,
    and -- when ``U`` is a unimodular (``abs(det) == 1``) integer lattice normalizer of the child --
    applies ``U^T`` through the validity-gated :func:`_apply_normalizer_operation` and lifts the
    re-expressed child.  A ``U`` that is non-integer or volume-changing (a centred child whose
    conventional cell is an intrinsic supercell, e.g. the R-centred trigonal case) is not a normalizer
    and is skipped here.  The committed descent round-trip gate in :func:`_lift_transform` remains
    authoritative against the original child.

    ``_lift_transform`` is called directly, deliberately bypassing the same-setting
    :data:`COMPATIBLE_CRYSTAL_SYSTEMS` filter that cell re-choice is designed to escape.
    """
    child_gram_surd = SurdVector(structure.cell.basis) * SurdVector(structure.cell.basis).T()
    if not child_gram_surd.is_rational:
        return ()
    child_gram = tuple(tuple(value for value in row) for row in child_gram_surd.to_fractions_approx())
    results: list[LiftResult] = []
    for transform in subgroup_transforms(target, structure.spacegroup):
        if not _multiplicity_possible(structure, transform):
            continue
        if _cell_for_transform(structure, transform, tolerance) is not None:
            continue
        matrix_transpose = _exact_rational_matrix(transform.operation.matrix.T())
        matrix_transpose_inverse = _exact_rational_matrix(transform.operation.matrix.T().inv())
        parent_gram = _matmul3(
            _matmul3(matrix_transpose_inverse, child_gram),
            tuple(tuple(matrix_transpose_inverse[j][i] for j in range(3)) for i in range(3)),
        )
        basis = _search_conventional_basis(parent_gram, transform.parent.crystal_system)
        if basis is None:
            continue
        rechoice = _matmul3(
            _matmul3(matrix_transpose, tuple(tuple(Fraction(v) for v in row) for row in basis)),
            matrix_transpose_inverse,
        )
        if any(value.denominator != 1 for row in rechoice for value in row):
            continue
        determinant = (
            rechoice[0][0] * (rechoice[1][1] * rechoice[2][2] - rechoice[1][2] * rechoice[2][1])
            - rechoice[0][1] * (rechoice[1][0] * rechoice[2][2] - rechoice[1][2] * rechoice[2][0])
            + rechoice[0][2] * (rechoice[1][0] * rechoice[2][1] - rechoice[1][1] * rechoice[2][0])
        )
        # A re-expression is a same-crystal re-choice only if it preserves the cell volume: an integer
        # but volume-changing matrix (real runs produced det -2 and -6) is a supercell, not a lattice
        # normalizer, and must not be applied even though the downstream re-identification would also
        # reject it.  The right-handed search basis makes the surviving determinant exactly +1.
        if abs(determinant) != 1:
            continue
        # Pre-check the rebased parent metric before paying for the re-expression and lift: applying
        # the operation sets the child basis to inv(U)*B, so the parent cell it derives is
        # inv(M^T)*inv(U)*B.  Most searched re-choices do not clear the parent metric here (8 of 13
        # firings in review), so this exact check -- the same one _cell_for_transform applies, on the
        # exact rational grams this tier is guarded to -- skips them cheaply.
        rebased_parent = SurdVector(transform.operation.matrix.T().inv()) * (
            SurdVector(FracVector(rechoice).inv()) * structure.cell.basis
        )
        measured = Cell(rebased_parent)
        equal_lengths, fixed_angles = _metric_requirements(transform.parent.crystal_system)
        metric = measured.metric()
        if not all(metric._element((i, i)) == metric._element((j, j)) for i, j in equal_lengths):
            continue
        if not all(measured.angles[index] == angle for index, angle in fixed_angles):
            continue
        transpose = FracVector(tuple(tuple(rechoice[j][i] for j in range(3)) for i in range(3)))
        image = _apply_normalizer_operation(structure, AffineOperation(transpose, (0, 0, 0)))
        if image is None:
            continue
        results.extend(_lift_transform(image, transform, tolerance))
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
    parents = minimal_supergroups(structure.spacegroup)
    indices_by_parent: dict[int, dict[SubgroupTransform, int]] = {}
    for parent_number in parents:
        transforms = subgroup_transforms(parent_number, structure.spacegroup)
        indices = {transform: index for index, transform in enumerate(transforms)}
        indices_by_parent[parent_number] = indices
        target = Spacegroup.standard(parent_number)
        try:
            raw = _raw_lifts(structure, target, tolerance)
            # The state normal form now collapses the origin/normalizer-equivalent variants that
            # unconditional retries used to contribute, so only fall back to retries when the direct
            # lift found nothing for this parent (backward_lift keeps unconditional retries).
            candidates = raw or _normalizer_retries(structure, target, tolerance)
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
    if not results:
        # Only a genuinely stranded state -- no direct or retry lift into any parent -- pays for the
        # exact conventional-cell re-choice search, so its cost never touches states that climb
        # normally (the search is dormant on every P-lattice battery).
        for parent_number in parents:
            for result in _recell_lifts(structure, Spacegroup.standard(parent_number), tolerance):
                if result.path and result.path[0] in indices_by_parent[parent_number]:
                    results.append((parent_number, indices_by_parent[parent_number][result.path[0]], result))
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


def _canonical_orientation(structure: ASUStructure) -> ASUStructure:
    """Put the cell in the standard crystallographic orientation for its exact Gram matrix.

    Equal crystals reached through different lifts can carry the same metric in different rotated
    frames; fractional coordinates are orientation-independent, so re-expressing the cell in the
    unique standard orientation of its parameters makes the returned basis a function of the metric
    alone.  A det<0 (left-handed) basis -- which a det=-1 normalizer coset can leave behind -- is
    first re-expressed through inversion so the standard-orientation rebuild does not silently
    produce the enantiomorph.  In an enantiomorphic group inversion conjugates to the other group and
    is rejected; the exact left-handed cell is then kept as-is (chirality-preserving, only cosmetic
    orientation canonicity is lost).  The rebuild is otherwise accepted only when it reproduces the
    Gram matrix exactly (the stored angles are approximate fractions, so it need not).
    """
    if structure.cell.basis.det().sign() < 0:
        # Inversion as a basis change: basis -> -B (right-handed), coords -> -x, Cartesian identical.
        inverted = _apply_normalizer_operation(
            structure, AffineOperation(FracVector(((-1, 0, 0), (0, -1, 0), (0, 0, -1))), (0, 0, 0))
        )
        if inverted is None:
            # Enantiomorphic group: cannot invert without changing the crystal.  Keep the exact
            # left-handed cell rather than letting the CellParams rebuild produce the enantiomorph.
            return structure
        structure = inverted
    cell = structure.cell
    lengths = cell.lengths
    if any(not length.is_rational for length in lengths):
        return structure
    params = [length._rational_fraction() for length in lengths] + list(cell.angles)
    try:
        oriented = Cell(CellParams(params).basis, precision=cell.precision, periodicity=cell.periodicity)
    except ValueError:
        return structure
    if oriented.basis == cell.basis or oriented.metric() != cell.metric():
        return structure
    return ASUStructure(
        oriented,
        structure.spacegroup,
        structure.wyckoff_sites,
        structure.species,
        transform=SettingTransform.identity(),
        coordinate_precision=structure.coordinate_precision,
        charge=structure.charge,
    )


def _exact_translations(atoms: list[tuple[str, tuple[Fraction, ...]]]) -> list[tuple[Fraction, ...]]:
    """Return every exact non-lattice translation that maps a P1 atom set onto itself.

    Candidates are the differences within the smallest species class (any self-translation must send
    that class's first atom to another of its atoms), and each is verified against the full per-species
    wrapped-coordinate **multisets** by exact rational equality -- no tolerance.  The multiset (not a
    set) matters: a duplicated ``(species, coordinate)`` site finds no self-translation and passes
    through untouched instead of reaching the reduction's internal consistency error.
    """
    by_species: dict[str, Counter[tuple[Fraction, ...]]] = defaultdict(Counter)
    for species, coordinate in atoms:
        by_species[species][coordinate] += 1
    smallest = min(by_species.values(), key=lambda counts: counts.total())
    anchor = FracVector(next(iter(smallest)))
    translations: list[tuple[Fraction, ...]] = []
    for point in smallest:
        candidate = (FracVector(point) - anchor).normalize()
        if not any(candidate.to_fractions()):
            continue
        if all(
            Counter(
                {
                    tuple((FracVector(coordinate) + candidate).normalize().to_fractions()): count
                    for coordinate, count in counts.items()
                }
            )
            == counts
            for counts in by_species.values()
        ):
            translations.append(tuple(candidate.to_fractions()))
    return translations


def _primitive_reduced_entry(structure: ASUStructure) -> ASUStructure:
    """Collapse an exact P1 supercell to its unique primitive description before the search.

    Any exact supercell -- any multiplicity, any (diagonal or sheared) sublattice orientation -- has
    a translation group finer than the cell lattice.  The finer lattice ``L'`` is generated by ``Z^3``
    together with **all** exact self-translations.  ``L'`` itself is unique, but the basis emitted here
    (``_lattice_basis`` echelon output) is order-dependent, so it is *not* a canonical cell: this step
    only reaches *a* primitive description.  Description-invariance of the final result comes from the
    downstream Niggli reduction and normal form, not from this helper, which must not be reused
    standalone as a canonicalizer.  With ``S`` the new lattice basis in old fractional units (rows),
    the new cell is ``B' = S B`` and coordinates remap as ``c' = c S^-1``; each primitive atom then
    appears exactly ``n = 1/|det S|`` times and is deduplicated.  Fractional coordinate precision is
    rescaled through the subdivision (conservatively, as one hop does) so ``cartesian_precision`` and
    the derived tolerance are preserved.

    Only exact rational invariance fires this: a noisy (near-coincident) supercell is left untouched
    here -- snapping such input is :func:`~httk.atomistic.canonical_asu`'s tolerant job.
    """
    atoms = [
        (
            site.species,
            tuple(
                FracVector(structure.spacegroup.wyckoff_position(site.wyckoff).coordinates(site.free_params)[0])
                .normalize()
                .to_fractions()
            ),
        )
        for site in structure.wyckoff_sites
    ]
    if len(atoms) < 2:
        return structure
    translations = _exact_translations(atoms)
    if not translations:
        return structure

    denominator = math.lcm(*(value.denominator for translation in translations for value in translation))
    generators = [tuple(denominator if i == j else 0 for j in range(3)) for i in range(3)]
    generators += [tuple(int(value * denominator) for value in translation) for translation in translations]
    hermite = _lattice_basis(generators)
    lattice = tuple(tuple(Fraction(hermite[i][j], denominator) for j in range(3)) for i in range(3))
    determinant = (
        hermite[0][0] * (hermite[1][1] * hermite[2][2] - hermite[1][2] * hermite[2][1])
        - hermite[0][1] * (hermite[1][0] * hermite[2][2] - hermite[1][2] * hermite[2][0])
        + hermite[0][2] * (hermite[1][0] * hermite[2][1] - hermite[1][1] * hermite[2][0])
    )
    multiplicity, remainder = divmod(denominator**3, abs(determinant))
    assert remainder == 0

    inverse = _rational_inverse(lattice)
    collapsed: dict[tuple[str, tuple[Fraction, ...]], int] = defaultdict(int)
    for species, coordinate in atoms:
        reduced = tuple((FracVector(coordinate) * FracVector(inverse)).normalize().to_fractions())
        collapsed[(species, reduced)] += 1
    if any(count != multiplicity for count in collapsed.values()):
        raise ValueError("exact supercell reduction produced uneven site copies; this is a logic error")

    sites = [WyckoffSite("a", FracVector(coordinate), species) for species, coordinate in collapsed]
    charge = None if structure.charge is None else structure.charge / multiplicity
    # B' = S * B, so scale precisions exactly as one subgroup hop does for a basis change: the cell's
    # absolute precision by the row-sum factor of S, and the fractional coordinate precision by the
    # column-sum factor of S^-1 -- the conservative bound that keeps cartesian_precision equal-or-wider.
    matrix = FracVector(lattice)
    return ASUStructure(
        Cell(
            SurdVector(lattice) * structure.cell.basis,
            precision=_scaled_precision(structure.cell.precision, _matrix_row_sum_factor(matrix)),
            periodicity=structure.cell.periodicity,
        ),
        structure.spacegroup,
        sites,
        structure.species,
        transform=SettingTransform.identity(),
        coordinate_precision=_scaled_precision(
            structure.coordinate_precision, _matrix_column_sum_factor(FracVector(inverse))
        ),
        charge=charge,
    )


def _niggli_reduced_entry(structure: ASUStructure) -> ASUStructure:
    """Re-express a triclinic-entry (SG 1 or 2) structure in its exact Niggli-reduced basis.

    Different bases of the same lattice reduce to the same Niggli cell, so this removes the
    basis-choice freedom before the search.  A non-rational (surd) Gram matrix that
    ``niggli_reduce`` cannot handle degrades gracefully to the unreduced structure.
    """
    # Local import: reduction is a sibling top-level module; importing it at module load would run
    # before httk.atomistic finishes wiring symmetry, so keep it lazy.
    from httk.atomistic.reduction import niggli_reduce

    try:
        reduced = niggli_reduce(structure.cell)
    except (ValueError, ArithmeticError):
        return structure
    if reduced.transform == FracVector.eye((3, 3)):
        return structure
    rebased = _apply_normalizer_operation(structure, AffineOperation(reduced.transform.T().inv(), (0, 0, 0)))
    return rebased if rebased is not None else structure


def _canonical_entry(structure: ASUStructure) -> ASUStructure:
    """Return the normal-form state ``highest_symmetry`` starts its search from.

    Standardizes the input, collapses an exact P1 supercell to its primitive cell, Niggli-reduces a
    triclinic (SG 1/2) cell, and reduces to the normalizer-canonical normal form.
    """
    current = _standard_input(structure)
    if current.spacegroup.it_number == 1:
        current = _primitive_reduced_entry(current)
    if current.spacegroup.it_number in (1, 2):
        current = _niggli_reduced_entry(current)
    return _normal_form(current)


def _canonical_without_bfs(structure: ASUStructure) -> ASUStructure:
    """Return the canonical representative of ``structure`` within its own group -- no upward search.

    This is exactly the terminal ``highest_symmetry`` would emit if the entry state had no lifts:
    the canonical-entry normal form, placed in the standard orientation of its metric.  It is the
    deterministic, fully invariant representation of the *recognized* symmetry, without hunting for
    pseudosymmetry above it.
    """
    return _canonical_orientation(_canonical_entry(structure))


def highest_symmetry(
    structure: ASUStructure, *, tolerance: float | None = None, all_paths: bool = False
) -> tuple[LiftResult, ...]:
    """Return all terminal upward lifts reached by breadth-first search.

    :param structure: The starting asymmetric-unit structure.
    :param tolerance: Cartesian acceptance tolerance, or the recognition-derived default.
    :param all_paths: When ``False`` (default) the visited set collapses alternate Bärnighausen
        routes to one entry per state, so each terminal appears once.  When ``True`` the visited set
        also keys on the accumulated path, so every distinct ``(terminal, path)`` pair is returned;
        the ``.asu`` representatives of the extra results are identical, only ``path`` differs.  The
        state cap therefore binds sooner under the flag.
    :return: Deterministically ordered highest-symmetry representations.
    :raises ValueError: If the input is unsupported, or if the breadth-first search exceeds its
        visited-state cap.  A per-parent modular-solver branch-cap failure is not raised: that parent
        target is skipped and reported through the ``"symmetry"`` warning channel, so in that rare
        case the returned symmetry may be lower than the true maximum.  The noisy least-squares
        fallback is capped the same way -- a candidate whose integer-wrap box exceeds the noisy cap is
        skipped (also on the ``"symmetry"`` warning channel), which could likewise lower the returned
        symmetry for a genuinely noisy large-box lift.

    An exact P1 supercell entry is first collapsed to its unique primitive description (any
    multiplicity or sublattice orientation), and a triclinic (SG 1 or 2) entry is then Niggli-reduced
    so the search is independent of the input basis choice.  Each search state is then reduced to its
    normalizer-canonical normal form --
    special-site demotion plus the continuous- and discrete-Euclidean-normalizer translation
    quotients and the affine-normalizer coset quotient -- collapsing origin-, basis- and
    normalizer-equivalent representations to one visited entry so the search terminates from a raw
    P1 input.  The returned ``asu`` is that normalizer-canonical representative, its cell placed in
    the standard orientation of its metric, so the result is invariant under origin shift, cell-basis
    choice, and site order for the same crystal.  ``path`` records the tabulated hops of the route
    that reached it, and a bounded normalizer retry along that route applies tabulated cosets to
    child coordinates and maps results back with the exact inverse, in tabulated coset order.
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
    # Standardize, collapse exact P1 supercells, Niggli-reduce triclinic cells, and take the normal
    # form -- the state the breadth-first search starts from.
    current = _canonical_entry(structure)
    accepted_tolerance = structure_tolerance(current) if tolerance is None else float(tolerance)
    queue: list[tuple[ASUStructure, tuple[SubgroupTransform, ...], FracVector, Fraction]] = [
        (current, (), FracVector((0, 0, 0)), Fraction(0))
    ]
    start_signature = (current.spacegroup.it_number, _structure_signature(current))
    visited: set[tuple[Any, ...]] = {(*start_signature, ()) if all_paths else start_signature}
    terminals: list[LiftResult] = []
    while queue:
        state, path, shift, residual = queue.pop(0)
        lifts = _highest_lifts(state, accepted_tolerance)
        if not lifts:
            terminals.append(_terminal_result(_canonical_orientation(state), path, shift, residual))
            continue
        for result in lifts:
            next_state = _normal_form(result.asu)
            next_path = path + result.path
            signature = (next_state.spacegroup.it_number, _structure_signature(next_state))
            # all_paths keys the visited set on the route too, so alternate routes to a state are all
            # explored rather than collapsed to the first one reached.
            key = (*signature, next_path) if all_paths else signature
            if key in visited:
                continue
            if len(visited) >= 10_000:
                # ponytail: bounded state scan; add quotient-graph memoization if tables grow materially.
                raise ValueError(f"highest_symmetry search cap exceeded for {current.spacegroup.setting}")
            visited.add(key)
            queue.append((next_state, next_path, result.shift, max(residual, result.residual)))
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

    The result is the normalizer-canonical representative of the input's crystal: the same exact
    ``(it_number, sorted (species, wyckoff, free_params), cell basis)`` for any origin shift,
    cell-basis choice (relabeling/shear), or site ordering of that crystal, and coherent with direct
    entry at its own space group.  See :func:`highest_symmetry` for the full contract; for a
    P1/unit-cell start build the ASU in SG 1 and pass it here.

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
