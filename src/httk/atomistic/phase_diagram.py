"""Float linear-programming phase diagrams for atomistic compositions.

The structure layer keeps coordinates and cells exact. Phase-diagram construction is a
deliberate numerical boundary instead: formula-unit counts and total energies are normalized
to per-atom floats and a small dense numpy simplex solver constructs the lower convex hull.
"""

import fractions
import math
from collections.abc import Mapping, Sequence
from typing import Any, Self

from ._vector_guards import require_numpy
from .structure_like import StructureLike

__all__ = ["PhaseDiagram"]

_PIVOT_TOLERANCE = 1e-11
_WEIGHT_TOLERANCE = 1e-10
_COMPOSITION_TOLERANCE = 1e-9


class _LPInfeasibleError(ValueError):
    """The equality-constrained linear program has no feasible point."""


class _LPUnboundedError(ValueError):
    """The equality-constrained linear program is unbounded below."""


def _basis_is_well_conditioned(matrix: Any, basis: Sequence[int], tolerance: float) -> bool:
    """Whether a candidate basis is numerically safe enough to solve."""
    import numpy as np

    if not basis:
        return True
    condition = float(np.linalg.cond(matrix[:, basis]))
    limit = 1.0 / max(tolerance, 100.0 * np.finfo(np.float64).eps)
    return math.isfinite(condition) and condition <= limit


def _matrix_rank(matrix: Any, threshold: float) -> int:
    """SVD rank against one caller-supplied absolute singular-value threshold."""
    import numpy as np

    if matrix.size == 0:
        return 0
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.count_nonzero(singular_values > threshold))


def _scaled_independent_equalities(
    matrix: Any,
    rhs: Any,
    tolerance: float,
) -> tuple[Any, Any, Any, Any]:
    """Scale equalities, reject inconsistency, and retain independent rows."""
    import numpy as np

    if matrix.shape[0] == 0:
        return matrix.copy(), rhs.copy(), matrix.copy(), rhs.copy()

    if matrix.shape[1] == 0:
        coefficient_sizes = np.zeros(matrix.shape[0], dtype=np.float64)
    else:
        coefficient_sizes = np.max(np.abs(matrix), axis=1)
    row_sizes = np.maximum(coefficient_sizes, np.abs(rhs))
    row_sizes[row_sizes == 0.0] = 1.0
    scaled_matrix = matrix / row_sizes[:, None]
    scaled_rhs = rhs / row_sizes
    augmented = np.column_stack((scaled_matrix, scaled_rhs))
    largest = float(np.linalg.svd(augmented, compute_uv=False)[0])
    rank_threshold = tolerance * max(1.0, largest)
    coefficient_rank = _matrix_rank(scaled_matrix, rank_threshold)
    augmented_rank = _matrix_rank(augmented, rank_threshold)
    if augmented_rank > coefficient_rank:
        raise _LPInfeasibleError("linear program is infeasible")

    selected: list[int] = []
    selected_rank = 0
    for row in range(scaled_matrix.shape[0]):
        trial = scaled_matrix[[*selected, row], :]
        trial_rank = _matrix_rank(trial, rank_threshold)
        if trial_rank > selected_rank:
            selected.append(row)
            selected_rank = trial_rank
        if selected_rank == coefficient_rank:
            break
    if selected_rank != coefficient_rank:
        raise RuntimeError("could not select independent equality constraints")
    return (
        scaled_matrix[selected, :],
        scaled_rhs[selected],
        scaled_matrix,
        scaled_rhs,
    )


def _simplex_iterations(
    matrix: Any,
    rhs: Any,
    costs: Any,
    basis: list[int],
    tolerance: float,
) -> tuple[list[int], Any]:
    """Run revised-simplex pivots from a feasible basis using Bland's rule."""
    import numpy as np

    row_count, variable_count = matrix.shape
    if row_count == 0:
        if np.any(costs < -tolerance):
            raise _LPUnboundedError("linear program is unbounded below")
        return basis, np.zeros(variable_count, dtype=np.float64)

    max_iterations = max(10_000, 100 * (row_count + variable_count))
    for _ in range(max_iterations):
        basis_matrix = matrix[:, basis]
        try:
            basic_values = np.linalg.solve(basis_matrix, rhs)
            multipliers = np.linalg.solve(basis_matrix.T, costs[basis])
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("simplex basis became singular") from exc

        small_negative = (basic_values < 0.0) & (np.abs(basic_values) <= tolerance)
        basic_values[small_negative] = 0.0
        if np.any(basic_values < -tolerance):
            raise RuntimeError("simplex basis lost primal feasibility")

        reduced_costs = costs - matrix.T @ multipliers
        reduced_costs[basis] = 0.0
        basis_set = set(basis)
        entering_candidates = [
            index for index in range(variable_count) if index not in basis_set and reduced_costs[index] < -tolerance
        ]
        if not entering_candidates:
            solution = np.zeros(variable_count, dtype=np.float64)
            solution[basis] = basic_values
            return basis, solution

        pivoted = False
        for entering in entering_candidates:
            direction = np.linalg.solve(basis_matrix, matrix[:, entering])
            eligible = [row for row in range(row_count) if direction[row] > tolerance]
            if not eligible:
                raise _LPUnboundedError("linear program is unbounded below")

            ratios = {row: basic_values[row] / direction[row] for row in eligible}
            minimum = min(ratios.values())
            machine_tolerance = 16.0 * np.finfo(np.float64).eps
            minimizers = [
                row
                for row in eligible
                if abs(ratios[row] - minimum) <= machine_tolerance * max(1.0, abs(minimum), abs(ratios[row]))
            ]
            for leaving_row in sorted(minimizers, key=basis.__getitem__):
                candidate_basis = basis.copy()
                candidate_basis[leaving_row] = entering
                if not _basis_is_well_conditioned(matrix, candidate_basis, tolerance):
                    continue
                basis = candidate_basis
                pivoted = True
                break
            if pivoted:
                break
        if not pivoted:
            raise RuntimeError("simplex found no numerically safe pivot")

    raise RuntimeError("simplex iteration limit exceeded")


def _solve_equality_lp(
    costs: Any,
    matrix: Any,
    rhs: Any,
    *,
    pivot_tolerance: float = _PIVOT_TOLERANCE,
) -> tuple[float, tuple[float, ...]]:
    """Minimize ``costs @ x`` subject to ``matrix @ x == rhs`` and ``x >= 0``.

    This is a deterministic dense two-phase revised simplex. Equalities are first
    max-scaled row by row, checked for consistency by comparing coefficient and augmented
    SVD ranks, and reduced to an independent row set. Artificial variables provide the
    phase-one feasible basis; zero artificial variables are then pivoted out before phase
    two. Both entering and genuinely tied leaving variables follow Bland's anti-cycling
    rule, and candidate pivots whose basis is too ill-conditioned are skipped.

    Raises:
        _LPInfeasibleError: If the equality constraints cannot be satisfied.
        _LPUnboundedError: If the objective is unbounded below.
    """
    require_numpy()
    import numpy as np

    tolerance = float(pivot_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("pivot_tolerance must be a finite positive number")

    objective = np.asarray(costs, dtype=np.float64)
    coefficients = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(rhs, dtype=np.float64)
    if objective.ndim != 1 or coefficients.ndim != 2 or target.ndim != 1:
        raise ValueError("costs and rhs must be vectors and matrix must be two-dimensional")
    row_count, variable_count = coefficients.shape
    if objective.shape != (variable_count,) or target.shape != (row_count,):
        raise ValueError("linear-program dimensions do not agree")
    if not (np.all(np.isfinite(objective)) and np.all(np.isfinite(coefficients)) and np.all(np.isfinite(target))):
        raise ValueError("linear-program inputs must be finite")
    coefficients, target, scaled_coefficients, scaled_target = _scaled_independent_equalities(
        coefficients,
        target,
        tolerance,
    )
    row_count = coefficients.shape[0]
    if variable_count == 0:
        return 0.0, ()

    negative_rhs = target < 0.0
    coefficients = coefficients.copy()
    target = target.copy()
    coefficients[negative_rhs] *= -1.0
    target[negative_rhs] *= -1.0

    artificial = np.eye(row_count, dtype=np.float64)
    phase_one_matrix = np.concatenate((coefficients, artificial), axis=1)
    phase_one_costs = np.concatenate(
        (
            np.zeros(variable_count, dtype=np.float64),
            np.ones(row_count, dtype=np.float64),
        )
    )
    basis = list(range(variable_count, variable_count + row_count))
    basis, phase_one_solution = _simplex_iterations(
        phase_one_matrix,
        target,
        phase_one_costs,
        basis,
        tolerance,
    )
    artificial_sum = float(np.sum(phase_one_solution[variable_count:]))
    if artificial_sum > tolerance:
        raise _LPInfeasibleError("linear program is infeasible")

    # A zero artificial basic variable either pivots onto an original column or marks a
    # redundant equality. The latter is common here because sum(composition) == sum(w).
    while any(index >= variable_count for index in basis):
        artificial_row = min(
            (row for row, index in enumerate(basis) if index >= variable_count),
            key=basis.__getitem__,
        )
        basis_matrix = phase_one_matrix[:, basis]
        tableau_original = np.linalg.solve(basis_matrix, phase_one_matrix[:, :variable_count])
        basis_set = set(basis)
        entering_candidates = [
            index
            for index in range(variable_count)
            if index not in basis_set and abs(tableau_original[artificial_row, index]) > tolerance
        ]
        pivoted = False
        for entering in entering_candidates:
            candidate_basis = basis.copy()
            candidate_basis[artificial_row] = entering
            if not _basis_is_well_conditioned(phase_one_matrix, candidate_basis, tolerance):
                continue
            basis = candidate_basis
            pivoted = True
            break
        if pivoted:
            continue
        if entering_candidates:
            raise RuntimeError("simplex found no numerically safe artificial-variable pivot")
        phase_one_matrix = np.delete(phase_one_matrix, artificial_row, axis=0)
        target = np.delete(target, artificial_row)
        del basis[artificial_row]

    phase_two_matrix = phase_one_matrix[:, :variable_count]
    basis, solution = _simplex_iterations(
        phase_two_matrix,
        target,
        objective,
        basis,
        tolerance,
    )
    del basis
    residual = scaled_coefficients @ solution - scaled_target
    if np.any(np.abs(residual) > 100.0 * tolerance):
        raise _LPInfeasibleError("linear program did not reach a feasible point")
    return float(objective @ solution), tuple(float(value) for value in solution)


def _formula_label(composition: Mapping[str, int | float | fractions.Fraction]) -> str:
    """Return a compact deterministic unit-cell formula label."""
    parts: list[str] = []
    for element in sorted(composition):
        count = float(composition[element])
        if count <= 0.0:
            continue
        parts.append(element)
        if not math.isclose(count, 1.0, rel_tol=0.0, abs_tol=1e-12):
            parts.append(f"{count:.12g}")
    return "".join(parts) or "empty"


class PhaseDiagram:
    """A normalized float convex-hull phase diagram.

    Construct with :meth:`from_compositions` or :meth:`from_structures`. Energies supplied
    to either factory are total formula-unit or unit-cell energies and are divided by the
    corresponding atom count. All exposed compositions and energies are plain floats.

    Hull membership uses the requested per-atom energy ``tolerance``. At duplicated
    compositions only the lowest-energy polymorph is stable, except that every polymorph
    tied within the tolerance is reported stable. ``None`` decompositions identify stable
    phases, including uncontested phases whose composition lies outside the convex hull of
    every other input.

    Tie-lines are every midpoint-supported pair of distinct stable compositions. A stable
    phase lying genuinely between two others on both their composition and energy segments
    subsumes the long segment. Distinct pair endpoints are compared exactly; the internal
    ``1e-9`` composition tolerance is used only to decide whether a third phase lies on
    their open segment. With four or more exactly coplanar stable phases, crossing diagonals
    can describe equally valid triangulations; this class reports all midpoint-supported
    pairs rather than choosing one triangulation.
    """

    _elements: tuple[str, ...]
    _ids: tuple[str, ...]
    _compositions: tuple[tuple[float, ...], ...]
    _energies_per_atom: tuple[float, ...]
    _tolerance: float
    _hull_indices: tuple[int, ...]
    _energy_above_hull: tuple[float, ...]
    _phase_lines: tuple[tuple[int, int], ...]
    _decompositions: tuple[tuple[tuple[int, float], ...] | None, ...]

    def __init__(
        self,
        elements: tuple[str, ...],
        ids: tuple[str, ...],
        compositions: tuple[tuple[float, ...], ...],
        energies_per_atom: tuple[float, ...],
        tolerance: float,
    ) -> None:
        require_numpy()
        self._elements = elements
        self._ids = ids
        self._compositions = compositions
        self._energies_per_atom = energies_per_atom
        self._tolerance = tolerance
        self._analyze()

    @classmethod
    def from_compositions(
        cls,
        compositions: Sequence[Mapping[str, int | float | fractions.Fraction]],
        energies: Sequence[float],
        ids: Sequence[str] | None = None,
        *,
        tolerance: float = 1e-8,
    ) -> Self:
        """Build a diagram from formula-unit compositions and total energies.

        Each composition maps an element label to its count in the formula unit to which
        the matching total energy refers. Counts must be finite and non-negative, with a
        strictly positive total atom count. Elements are sorted independently of mapping
        insertion order before rows are normalized to atomic fractions.
        """
        require_numpy()
        composition_rows = tuple(compositions)
        energy_values = tuple(float(energy) for energy in energies)
        if not composition_rows:
            raise ValueError("a phase diagram requires at least one phase")
        if len(composition_rows) != len(energy_values):
            raise ValueError("compositions and energies must have the same length")
        energy_tolerance = float(tolerance)
        if not math.isfinite(energy_tolerance) or energy_tolerance < 0.0:
            raise ValueError("tolerance must be a finite non-negative number")
        if not all(math.isfinite(energy) for energy in energy_values):
            raise ValueError("energies must be finite")

        elements = tuple(sorted({element for row in composition_rows for element in row}))
        if not elements:
            raise ValueError("compositions must contain at least one element")

        normalized: list[tuple[float, ...]] = []
        per_atom: list[float] = []
        for row, energy in zip(composition_rows, energy_values, strict=True):
            counts = tuple(float(row.get(element, 0.0)) for element in elements)
            if not all(math.isfinite(count) and count >= 0.0 for count in counts):
                raise ValueError("composition counts must be finite and non-negative")
            atom_count = sum(counts)
            if not math.isfinite(atom_count):
                raise ValueError("a phase composition must have a finite atom count")
            if atom_count <= 0.0:
                raise ValueError("a phase composition must have a positive atom count")
            largest_count = max(counts)
            scaled_total = sum(count / largest_count for count in counts)
            fraction_row = tuple((count / largest_count) / scaled_total for count in counts)
            if not all(math.isfinite(fraction) for fraction in fraction_row):
                raise ValueError("normalized composition fractions must be finite")
            normalized.append(fraction_row)
            normalized_energy = energy / atom_count
            if not math.isfinite(normalized_energy):
                raise ValueError("energy per atom must be finite")
            per_atom.append(normalized_energy)

        if ids is None:
            normalized_ids = tuple(_formula_label(row) for row in composition_rows)
        else:
            normalized_ids = tuple(str(identifier) for identifier in ids)
            if len(normalized_ids) != len(composition_rows):
                raise ValueError("ids must have the same length as compositions")

        return cls(
            elements,
            normalized_ids,
            tuple(normalized),
            tuple(per_atom),
            energy_tolerance,
        )

    @classmethod
    def from_structures(
        cls,
        structures: Sequence[StructureLike],
        energies: Sequence[float],
        ids: Sequence[str] | None = None,
        *,
        tolerance: float = 1e-8,
    ) -> Self:
        """Build a diagram from structures and their total unit-cell energies.

        Each site contributes its named species' ``chemical_symbols`` weighted by
        ``concentration``. ``"vacancy"`` contributes no atoms; ``"X"`` is rejected because
        an unknown element cannot define a composition coordinate. Default identifiers are
        deterministic alphabetically sorted labels using the (possibly fractional) unit-cell
        counts without reducing them.
        """
        require_numpy()
        from .unitcell_structure_view import UnitcellStructureView

        structure_values = tuple(structures)
        composition_rows: list[dict[str, float]] = []
        for structure_like in structure_values:
            structure = UnitcellStructureView(structure_like)
            species_by_name = {species.name: species for species in structure.species}
            composition: dict[str, float] = {}
            for name in structure.species_at_sites:
                species = species_by_name[name]
                for symbol, concentration in zip(
                    species.chemical_symbols,
                    species.concentration,
                    strict=True,
                ):
                    if not math.isfinite(concentration) or concentration < 0.0:
                        raise ValueError("species concentrations must be finite and non-negative")
                    if symbol == "vacancy":
                        continue
                    if symbol == "X":
                        raise ValueError("cannot build a phase diagram from unknown chemical symbol 'X'")
                    composition[symbol] = composition.get(symbol, 0.0) + float(concentration)
            composition_rows.append(composition)

        if ids is None:
            ids = tuple(_formula_label(row) for row in composition_rows)
        return cls.from_compositions(
            composition_rows,
            energies,
            ids,
            tolerance=tolerance,
        )

    def _mixture(
        self,
        indices: Sequence[int],
        target: tuple[float, ...],
    ) -> tuple[float, tuple[float, ...]]:
        # Normalized rows sum to one, so the final element equality is exactly
        # redundant with sum(weights) == 1. Drop it deterministically before phase I.
        matrix = [
            [self._compositions[index][element] for index in indices]
            for element in range(max(0, len(self._elements) - 1))
        ]
        matrix.append([1.0] * len(indices))
        rhs = [*target[:-1], 1.0]
        costs = [self._energies_per_atom[index] for index in indices]
        return _solve_equality_lp(costs, matrix, rhs)

    def _analyze(self) -> None:
        phase_count = len(self)
        stable: list[int] = []
        above_hull: list[float] = []
        for index in range(phase_count):
            competitors = tuple(candidate for candidate in range(phase_count) if candidate != index)
            try:
                value, _ = self._mixture(competitors, self._compositions[index])
            except _LPInfeasibleError:
                stable.append(index)
                above_hull.append(0.0)
                continue
            difference = self._energies_per_atom[index] - value
            above_hull.append(max(0.0, difference))
            if difference <= self._tolerance:
                stable.append(index)

        self._hull_indices = tuple(stable)
        self._energy_above_hull = tuple(above_hull)

        decompositions: list[tuple[tuple[int, float], ...] | None] = []
        stable_set = set(stable)
        for index in range(phase_count):
            if index in stable_set:
                decompositions.append(None)
                continue
            try:
                _, weights = self._mixture(stable, self._compositions[index])
            except _LPInfeasibleError as exc:
                raise RuntimeError("stable phases do not span an unstable composition") from exc
            decompositions.append(
                tuple(
                    (stable[position], weight) for position, weight in enumerate(weights) if weight > _WEIGHT_TOLERANCE
                )
            )
        self._decompositions = tuple(decompositions)

        lines: list[tuple[int, int]] = []
        for position, first in enumerate(stable):
            for second in stable[position + 1 :]:
                first_composition = self._compositions[first]
                second_composition = self._compositions[second]
                if first_composition == second_composition:
                    continue
                midpoint = tuple(
                    (left + right) / 2.0 for left, right in zip(first_composition, second_composition, strict=True)
                )
                value, _ = self._mixture(stable, midpoint)
                pair_value = (self._energies_per_atom[first] + self._energies_per_atom[second]) / 2.0
                if pair_value > value + self._tolerance:
                    continue
                if self._is_subsumed(first, second, stable):
                    continue
                lines.append((first, second))
        self._phase_lines = tuple(lines)

    def _is_subsumed(self, first: int, second: int, stable: Sequence[int]) -> bool:
        left = self._compositions[first]
        right = self._compositions[second]
        difference = tuple(a - b for a, b in zip(left, right, strict=True))
        axis = max(range(len(difference)), key=lambda index: abs(difference[index]))
        if abs(difference[axis]) <= _COMPOSITION_TOLERANCE:
            return False
        for middle in stable:
            if middle == first or middle == second:
                continue
            candidate = self._compositions[middle]
            if _rows_close(candidate, left) or _rows_close(candidate, right):
                continue
            fraction = (candidate[axis] - right[axis]) / difference[axis]
            if not _COMPOSITION_TOLERANCE < fraction < 1.0 - _COMPOSITION_TOLERANCE:
                continue
            interpolated = tuple(fraction * a + (1.0 - fraction) * b for a, b in zip(left, right, strict=True))
            if not _rows_close(candidate, interpolated):
                continue
            energy = fraction * self._energies_per_atom[first] + (1.0 - fraction) * self._energies_per_atom[second]
            if abs(self._energies_per_atom[middle] - energy) <= self._tolerance:
                return True
        return False

    @property
    def elements(self) -> tuple[str, ...]:
        """Element-coordinate order used by :attr:`compositions`."""
        return self._elements

    @property
    def ids(self) -> tuple[str, ...]:
        """Phase identifiers in input order."""
        return self._ids

    @property
    def compositions(self) -> tuple[tuple[float, ...], ...]:
        """Per-atom composition-fraction rows in :attr:`elements` order."""
        return self._compositions

    @property
    def energies_per_atom(self) -> tuple[float, ...]:
        """Input energies normalized by formula-unit or unit-cell atom count."""
        return self._energies_per_atom

    @property
    def hull_indices(self) -> tuple[int, ...]:
        """Stable phase indices in input order."""
        return self._hull_indices

    @property
    def energy_above_hull(self) -> tuple[float, ...]:
        """Non-negative energy distance from the lower hull in per-atom units."""
        return self._energy_above_hull

    @property
    def phase_lines(self) -> tuple[tuple[int, int], ...]:
        """Sorted midpoint-supported stable tie-lines as ``(smaller, larger)`` indices."""
        return self._phase_lines

    def decomposition(self, index: int) -> tuple[tuple[int, float], ...] | None:
        """Optimal stable-phase ``(index, weight)`` pairs, or ``None`` when stable."""
        return self._decompositions[index]

    def is_stable(self, index: int) -> bool:
        """Whether the phase at ``index`` is stable within the energy tolerance."""
        return index in self._hull_indices

    def __len__(self) -> int:
        """Number of phases."""
        return len(self._compositions)

    def plot(
        self,
        *,
        ax: Any = None,
        show_unstable: bool = True,
        label_stable: bool = True,
    ) -> Any:
        """Plot the phase diagram and return its matplotlib Axes.

        Binary diagrams use the fraction of the second element on the x axis. The y axis is
        formation energy relative to the lowest stable pure-element endpoints when both are
        present, and raw energy per atom otherwise. One-element diagrams use raw energy at a
        single composition point.

        Diagrams with three or more elements use a regular composition polygon whose corner
        ``k`` is at angle ``2*pi*k/N``. Phase lines are drawn individually in black; stable
        phases are filled and unstable phases are open. Matplotlib is imported only here and
        this method never calls ``show`` or writes a file.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("PhaseDiagram.plot() requires matplotlib; install matplotlib") from exc

        if ax is None:
            _, ax = plt.subplots()
        if len(self._elements) <= 2:
            self._plot_binary(ax, show_unstable=show_unstable, label_stable=label_stable)
        else:
            self._plot_polygon(ax, show_unstable=show_unstable, label_stable=label_stable)
        return ax

    def _plot_binary(self, ax: Any, *, show_unstable: bool, label_stable: bool) -> None:
        binary = len(self._elements) == 2
        xs = [row[1] if binary else 0.0 for row in self._compositions]
        ys = list(self._energies_per_atom)
        formation = False
        if binary:
            left = [index for index in self._hull_indices if abs(xs[index]) <= _COMPOSITION_TOLERANCE]
            right = [index for index in self._hull_indices if abs(xs[index] - 1.0) <= _COMPOSITION_TOLERANCE]
            if left and right:
                left_energy = min(self._energies_per_atom[index] for index in left)
                right_energy = min(self._energies_per_atom[index] for index in right)
                ys = [
                    energy - ((1.0 - x) * left_energy + x * right_energy)
                    for x, energy in zip(xs, self._energies_per_atom, strict=True)
                ]
                formation = True

        binary_lines = sorted(
            self._phase_lines,
            key=lambda pair: (
                min(xs[pair[0]], xs[pair[1]]),
                max(xs[pair[0]], xs[pair[1]]),
                pair,
            ),
        )
        for first, second in binary_lines:
            if xs[first] > xs[second]:
                first, second = second, first
            ax.plot(
                [xs[first], xs[second]],
                [ys[first], ys[second]],
                color="black",
                linewidth=1.5,
            )
        stable_set = set(self._hull_indices)
        stable = list(self._hull_indices)
        unstable = [index for index in range(len(self)) if index not in stable_set]
        if stable:
            ax.plot(
                [xs[index] for index in stable],
                [ys[index] for index in stable],
                linestyle="none",
                marker="o",
                color="black",
            )
        if show_unstable and unstable:
            ax.plot(
                [xs[index] for index in unstable],
                [ys[index] for index in unstable],
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                markeredgecolor="black",
            )
        if label_stable:
            for index in stable:
                ax.annotate(
                    self._ids[index],
                    (xs[index], ys[index]),
                    xytext=(4, 4),
                    textcoords="offset points",
                )
        if binary:
            ax.set_xlim(-0.03, 1.03)
            ax.set_xlabel(f"fraction of {self._elements[1]}")
        else:
            ax.set_xticks([0.0], labels=[self._elements[0]])
            ax.set_xlabel("composition")
        ax.set_ylabel("formation energy / atom" if formation else "energy / atom")

    def _plot_polygon(self, ax: Any, *, show_unstable: bool, label_stable: bool) -> None:
        corners = tuple(
            (
                math.cos(2.0 * math.pi * index / len(self._elements)),
                math.sin(2.0 * math.pi * index / len(self._elements)),
            )
            for index in range(len(self._elements))
        )
        positions = tuple(
            (
                sum(weight * corner[0] for weight, corner in zip(row, corners, strict=True)),
                sum(weight * corner[1] for weight, corner in zip(row, corners, strict=True)),
            )
            for row in self._compositions
        )
        boundary = (*corners, corners[0])
        ax.plot(
            [point[0] for point in boundary],
            [point[1] for point in boundary],
            color="black",
            linewidth=1.0,
        )
        for element, corner in zip(self._elements, corners, strict=True):
            ax.text(
                1.12 * corner[0],
                1.12 * corner[1],
                element,
                horizontalalignment="center",
                verticalalignment="center",
            )
        for first, second in self._phase_lines:
            ax.plot(
                [positions[first][0], positions[second][0]],
                [positions[first][1], positions[second][1]],
                color="black",
                linewidth=1.5,
            )
        stable_set = set(self._hull_indices)
        stable = list(self._hull_indices)
        unstable = [index for index in range(len(self)) if index not in stable_set]
        if stable:
            ax.plot(
                [positions[index][0] for index in stable],
                [positions[index][1] for index in stable],
                linestyle="none",
                marker="o",
                color="black",
            )
        if show_unstable and unstable:
            ax.plot(
                [positions[index][0] for index in unstable],
                [positions[index][1] for index in unstable],
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                markeredgecolor="black",
            )
        if label_stable:
            for index in stable:
                ax.annotate(
                    self._ids[index],
                    positions[index],
                    xytext=(4, 4),
                    textcoords="offset points",
                )
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()


def _rows_close(
    first: Sequence[float],
    second: Sequence[float],
    tolerance: float = _COMPOSITION_TOLERANCE,
) -> bool:
    return all(abs(left - right) <= tolerance for left, right in zip(first, second, strict=True))
