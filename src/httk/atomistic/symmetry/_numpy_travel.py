"""NumPy kernel for the approximate structure travel comparison.

The exact structure representation remains the source of truth.  This module is
loaded only by the opt-in approximate comparison path and converts the expanded
orbits to temporary ``float64`` Cartesian arrays for the numerical part of one
structure pair.
"""

from collections.abc import Callable
from typing import Any

from httk.atomistic.models._vector_guards import require_numpy
from httk.atomistic.models.structure.asu import ASUStructure


def prepare_travel(first: ASUStructure, second: ASUStructure) -> Callable[[int, int], float]:
    """Prepare vectorized orbit travel between two exact asymmetric units.

    The exact Wyckoff expansions and cell bases are rendered once as temporary
    ``float64`` Cartesian arrays.  The returned callable computes the minimum
    assignment cost for a pair of orbit indices, retaining only that orbit pair's
    cost matrix while it runs.  Periodic image selection and assignment use the
    existing exact-path implementations after the floating point distance kernel.

    :param first: The first structure, whose orbit indices are accepted by the result.
    :param second: The second structure, whose orbit indices are accepted by the result.
    :return: A callable returning the approximate travel cost for two orbit indices.
    :raises ImportError: If the optional NumPy dependency is unavailable.
    :raises ValueError: If the expanded structures or their metric are not finite or
        three-dimensional.
    """
    require_numpy()
    import numpy

    # Import lazily to avoid a module cycle: paths imports this kernel only when
    # its explicit approximate mode is selected.
    from httk.atomistic.symmetry.paths import _minimum_assignment_cost, _TravelMetric

    metric = _TravelMetric.from_cells(first.cell, second.cell)
    first_orbits = _cartesian_orbits(first, numpy)
    second_orbits = _cartesian_orbits(second, numpy)

    gram = numpy.asarray(metric.gram, dtype=numpy.float64)
    lower = numpy.asarray(metric.lower, dtype=numpy.float64)
    mean_rows = numpy.asarray(metric.mean_rows, dtype=numpy.float64)
    if gram.shape != (3, 3) or lower.shape != (3, 3) or mean_rows.shape != (3, 3):
        raise ValueError("structure_delta requires a three-dimensional cell metric")
    if not (
        numpy.all(numpy.isfinite(gram)) and numpy.all(numpy.isfinite(lower)) and numpy.all(numpy.isfinite(mean_rows))
    ):
        raise ValueError("structure_delta requires a finite cell metric")

    def orbit_travel(first_index: int, second_index: int) -> float:
        """Return the minimum approximate travel for two prepared orbit indices."""
        left = first_orbits[first_index]
        right = second_orbits[second_index]
        if len(left) != len(right):
            raise ValueError("structures have incompatible Wyckoff orbit multiplicities")
        if len(left) == 0:
            return 0.0

        # Keep the temporary arrays bounded by a single orbit pair.  A pair of
        # m/n-site orbits uses O(m*n) doubles; the prepared endpoint arrays are
        # only O(total expanded atoms) and are released with this closure.
        displacement = (left[:, numpy.newaxis, :] - right[numpy.newaxis, :, :]).reshape((-1, 3))
        if not numpy.all(numpy.isfinite(displacement)):
            raise ValueError("structure_delta produced non-finite Cartesian displacement")
        linear = displacement @ mean_rows.T
        if not numpy.all(numpy.isfinite(linear)):
            raise ValueError("structure_delta produced a non-finite travel metric")
        # L L.T = gram.  Solving the two triangular systems preserves the
        # factorization used by the exact-path metric and avoids forming an
        # inverse for every orbit pair.
        linear_rhs = -linear
        first_component = linear_rhs[:, 0] / lower[0, 0]
        second_component = (linear_rhs[:, 1] - lower[1, 0] * first_component) / lower[1, 1]
        third_component = (linear_rhs[:, 2] - lower[2, 0] * first_component - lower[2, 1] * second_component) / lower[
            2, 2
        ]
        result_third = third_component / lower[2, 2]
        result_second = (second_component - lower[2, 1] * result_third) / lower[1, 1]
        result_first = (first_component - lower[1, 0] * result_second - lower[2, 0] * result_third) / lower[0, 0]
        center = numpy.column_stack((result_first, result_second, result_third))
        if not numpy.all(numpy.isfinite(center)):
            raise ValueError("structure_delta produced a non-finite travel metric")
        nearest_displacement = center @ lower
        if not numpy.all(numpy.isfinite(nearest_displacement)):
            raise ValueError("structure_delta produced a non-finite nearest-image displacement")
        nearest_squared = numpy.asarray(
            [
                metric.nearest_image.distance((float(row[0]), float(row[1]), float(row[2]))) ** 2
                for row in nearest_displacement
            ],
            dtype=numpy.float64,
        )
        baseline = numpy.sum(displacement * displacement, axis=1) - numpy.sum((center @ gram) * center, axis=1)
        squared = baseline + nearest_squared
        if not numpy.all(numpy.isfinite(squared)):
            raise ValueError("structure_delta produced a non-finite travel squared")
        roundoff = 1e-12 * numpy.maximum(1.0, numpy.maximum(numpy.abs(baseline), nearest_squared))
        if numpy.any(squared < -roundoff):
            raise ValueError("structure_delta produced a negative travel squared")
        costs = numpy.sqrt(numpy.maximum(squared, 0.0)).reshape((len(left), len(right)))
        return _minimum_assignment_cost(tuple(tuple(float(value) for value in row) for row in costs))

    return orbit_travel


def _cartesian_orbits(structure: ASUStructure, numpy: Any) -> tuple[Any, ...]:
    """Expand one exact ASU into read-only temporary NumPy Cartesian arrays."""
    array = numpy.asarray(structure.cell.basis.to_floats(), dtype=numpy.float64)
    if array.shape != (3, 3) or not numpy.all(numpy.isfinite(array)):
        raise ValueError("structure_delta requires a finite three-dimensional cell basis")
    orbits: list[object] = []
    for site in structure.wyckoff_sites:
        points = structure.spacegroup.wyckoff_position(site.wyckoff).coordinates(site.free_params)
        fractional = numpy.asarray([point.to_floats() for point in points], dtype=numpy.float64)
        if fractional.ndim != 2 or fractional.shape[1] != 3 or not numpy.all(numpy.isfinite(fractional)):
            raise ValueError("structure_delta requires finite three-dimensional orbit coordinates")
        cartesian = fractional @ array
        if not numpy.all(numpy.isfinite(cartesian)):
            raise ValueError("structure_delta produced non-finite Cartesian orbit coordinates")
        cartesian.setflags(write=False)
        orbits.append(cartesian)
    return tuple(orbits)
