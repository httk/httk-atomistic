"""Finite closest-vector search for Cartesian periodic images."""

import math
from collections.abc import Sequence


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.fsum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


def _scaled_subtract(
    left: tuple[float, float, float], scale: float, right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[0] - scale * right[0],
        left[1] - scale * right[1],
        left[2] - scale * right[2],
    )


class _NearestImageMetric:
    """A factored periodic lattice metric for repeated nearest-image queries."""

    __slots__ = ("_coefficients", "_orthogonal", "_squared_norms")

    def __init__(self, basis: Sequence[Sequence[float]], periodicity: Sequence[bool]) -> None:
        if len(basis) != 3 or any(len(row) != 3 for row in basis) or len(periodicity) != 3:
            raise ValueError("nearest-image distance requires a three-dimensional basis and periodicity")
        rows = tuple((float(row[0]), float(row[1]), float(row[2])) for row in basis)
        if not all(math.isfinite(value) for row in rows for value in row):
            raise ValueError("nearest-image distance requires a finite cell basis")

        # Validate the coordinate frame itself, including when no direction is periodic.
        frame_orthogonal: list[tuple[float, float, float]] = []
        for vector in rows:
            remainder = vector
            for orthogonal_vector in frame_orthogonal:
                squared_norm = _dot(orthogonal_vector, orthogonal_vector)
                remainder = _scaled_subtract(
                    remainder, _dot(vector, orthogonal_vector) / squared_norm, orthogonal_vector
                )
            squared_norm = _dot(remainder, remainder)
            if not math.isfinite(squared_norm) or squared_norm <= 0.0:
                raise ValueError("nearest-image distance requires a non-singular cell basis")
            frame_orthogonal.append(remainder)

        lattice = tuple(row for row, periodic in zip(rows, periodicity, strict=True) if periodic)
        orthogonal: list[tuple[float, float, float]] = []
        squared_norms: list[float] = []
        coefficients = [[0.0] * len(lattice) for _ in lattice]
        for index, vector in enumerate(lattice):
            remainder = vector
            for previous_index, previous_vector in enumerate(orthogonal):
                coefficient = _dot(vector, previous_vector) / squared_norms[previous_index]
                coefficients[index][previous_index] = coefficient
                remainder = _scaled_subtract(remainder, coefficient, previous_vector)
            squared_norm = _dot(remainder, remainder)
            if not math.isfinite(squared_norm) or squared_norm <= 0.0:
                raise ValueError("nearest-image distance requires independent periodic basis vectors")
            orthogonal.append(remainder)
            squared_norms.append(squared_norm)
        self._orthogonal = tuple(orthogonal)
        self._squared_norms = tuple(squared_norms)
        self._coefficients = tuple(tuple(row) for row in coefficients)

    def distance(self, displacement: Sequence[float]) -> float:
        """Return the Cartesian distance to the nearest periodic image."""
        if len(displacement) != 3:
            raise ValueError("nearest-image distance requires a three-dimensional displacement")
        difference = (float(displacement[0]), float(displacement[1]), float(displacement[2]))
        if not all(math.isfinite(value) for value in difference):
            raise ValueError("nearest-image distance requires a finite displacement")
        squared = _dot(difference, difference)
        if not math.isfinite(squared):
            raise ValueError("nearest-image distance requires a finite Cartesian metric")
        target = tuple(
            _dot(difference, vector) / squared_norm
            for vector, squared_norm in zip(self._orthogonal, self._squared_norms, strict=True)
        )
        if not all(math.isfinite(value) for value in target):
            raise ValueError("nearest-image distance produced a non-finite lattice target")
        if not target:
            return math.sqrt(squared)
        # Subtract vectors, not squared norms: a tiny nonperiodic component can
        # disappear in |d|² - |projection|². A full-rank lattice has no residual.
        perpendicular = 0.0
        if len(target) < 3:
            residual = difference
            for coordinate, vector in zip(target, self._orthogonal, strict=True):
                residual = _scaled_subtract(residual, coordinate, vector)
            perpendicular = _dot(residual, residual)

        # Babai's nearest-plane result gives a finite initial radius for the exhaustive search.
        babai = [0] * len(target)
        for index in range(len(target) - 1, -1, -1):
            center = target[index] - math.fsum(
                babai[later] * self._coefficients[later][index] for later in range(index + 1, len(target))
            )
            babai[index] = round(center)

        def residual_squared(integers: Sequence[int]) -> float:
            return math.fsum(
                self._squared_norms[index]
                * (
                    target[index]
                    - integers[index]
                    - math.fsum(
                        integers[later] * self._coefficients[later][index] for later in range(index + 1, len(target))
                    )
                )
                ** 2
                for index in range(len(target))
            )

        best = residual_squared(babai)
        if not math.isfinite(best):
            raise ValueError("nearest-image distance produced a non-finite lattice distance")

        def search(index: int, chosen: list[int], accumulated: float) -> None:
            nonlocal best
            if index < 0:
                best = min(best, accumulated)
                return
            remaining = best - accumulated
            if remaining < 0.0:
                return
            center = target[index] - math.fsum(
                chosen[later] * self._coefficients[later][index] for later in range(index + 1, len(target))
            )
            radius = math.nextafter(math.sqrt(remaining / self._squared_norms[index]), math.inf)
            lower = math.ceil(center - radius)
            upper = math.floor(center + radius)
            for value in sorted(range(lower, upper + 1), key=lambda candidate: (abs(candidate - center), candidate)):
                contribution = self._squared_norms[index] * (center - value) ** 2
                if contribution <= remaining:
                    chosen[index] = value
                    search(index - 1, chosen, accumulated + contribution)

        search(len(target) - 1, [0] * len(target), 0.0)
        return math.sqrt(perpendicular + best)
