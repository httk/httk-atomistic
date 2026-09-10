"""Conservative sparse grids for approximate structure comparison candidates."""

import itertools
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from httk.atomistic.models._vector_guards import require_numpy
from httk.atomistic.models.protostructure.api import ProtostructureAPI
from httk.atomistic.models.prototype.api import PrototypeAPI
from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure
from httk.atomistic.symmetry.comparison_cache import StructureComparisonCache

__all__ = ["StructureComparisonGrid"]

type _Value = ASUStructure | FundamentalDomainStructure | PrototypeAPI | ProtostructureAPI
type _Class = tuple[str, str]


@dataclass
class _Reference:
    basis: Any
    inverse: Any
    anchors: list[tuple[_Class, Any]]


class StructureComparisonGrid:
    """Index necessary atom-neighborhood conditions for one same-group comparison batch.

    A false result from :meth:`might_match` excludes a pair; a true result still needs
    the ordinary similarity comparison. One to three Cartesian projections of expanded
    Wyckoff coordinates are gridded. Every candidate orbit member and normalizer image
    is indexed, allowing repeated sites and symmetry-equivalent descriptions. Queries
    include the reference cell's periodic images and both alignment directions.

    For atom travel c, the current endpoint metric implies
    ``norm(r1 - r2 + n B1) <= sqrt(2) c`` for some integer image n. Total travel within
    delta therefore requires every matched atom to satisfy this neighborhood condition.
    Selected projections only weaken that condition. Sparse buckets retain occupied
    cells only; caps and unsupported preparations fall back to ordinary comparison.

    :param values: Exact structures or geometry-carrying prototypes/protostructures.
    :param delta: Finite non-negative Cartesian travel threshold.
    :param dimensions: Number of Cartesian projections to index, from one to three.
    :param strategy: Axis selection: ``first``, ``variance``, or ``occupancy``.
    :param cache: Optional shared canonicalization and prototype-conversion cache.
    :param max_points: Maximum expanded points retained during index preparation.
    :param max_images: Maximum periodic images examined for any reference anchor.
    :raises ValueError: If a threshold, strategy, dimension, or capacity is invalid.
    :raises TypeError: If a value is not a supported structure or prototype.
    :raises ImportError: If NumPy is unavailable.
    """

    def __init__(
        self,
        values: Sequence[ASUStructure | FundamentalDomainStructure | PrototypeAPI | ProtostructureAPI],
        delta: float,
        *,
        dimensions: int = 2,
        strategy: str = "occupancy",
        cache: StructureComparisonCache | None = None,
        max_points: int = 100_000,
        max_images: int = 512,
    ) -> None:
        delta = _budget(delta)
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions not in (1, 2, 3):
            raise ValueError("dimensions must be 1, 2, or 3")
        if strategy not in ("first", "variance", "occupancy"):
            raise ValueError("strategy must be first, variance, or occupancy")
        for name, limit in (("max_points", max_points), ("max_images", max_images)):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError(f"{name} must be a positive integer")
        require_numpy()
        import numpy

        self._numpy = numpy
        self._count = len(values)
        self._max_images = max_images
        self._max_points = max_points
        self._references: list[_Reference] = []
        self._buckets: dict[_Class, dict[tuple[int, ...], set[int]]] = {}
        self._bounds: dict[_Class, tuple[Any, Any]] = {}
        self._queries: OrderedDict[int, frozenset[int] | None] = OrderedDict()
        self._query_entries = 0
        self._selected_axes: tuple[int, ...] = ()
        self._indexed_points = 0
        self._fallback_reason: str | None = None
        self._radius = 0.0
        if self._count < 2 or delta == 0:
            self._fallback_reason = "fewer than two values or a zero threshold"
            return
        if cache is None:
            cache = StructureComparisonCache(max_structures=max(1, 2 * self._count))
        try:
            self._build(values, float(delta), dimensions, strategy, cache)
        except (ValueError, ArithmeticError) as error:
            self._fallback_reason = str(error)
            self._references.clear()
            self._buckets.clear()
            self._bounds.clear()
            self._indexed_points = 0

    @property
    def fallback_reason(self) -> str | None:
        """Return why this index permits every pair, or ``None`` when indexing succeeded."""
        return self._fallback_reason

    @property
    def indexed_points(self) -> int:
        """Return the number of candidate orbit points processed by the index."""
        return self._indexed_points

    @property
    def selected_axes(self) -> tuple[int, ...]:
        """Return the selected Cartesian coordinate indices in grid-key order."""
        return self._selected_axes

    def might_match(self, first: int, second: int) -> bool:
        """Return whether a pair still requires the ordinary similarity comparison.

        :param first: Index of the first input value.
        :param second: Index of the second input value.
        :return: False only when both directed neighborhood tests exclude the pair.
        :raises IndexError: If an index is outside the supplied sequence.
        """
        if not 0 <= first < self._count or not 0 <= second < self._count:
            raise IndexError("comparison grid index out of range")
        if first == second or self._fallback_reason is not None:
            return True
        forward = self._query(first)
        if forward is None or second in forward:
            return True
        reverse = self._query(second)
        return reverse is None or first in reverse

    def _build(
        self, values: Sequence[_Value], delta: float, dimensions: int, strategy: str, cache: StructureComparisonCache
    ) -> None:
        from httk.atomistic.symmetry._numpy_travel import _cartesian_orbits
        from httk.atomistic.symmetry.lift import rerepresent
        from httk.atomistic.symmetry.paths import _normalizer_candidates
        from httk.atomistic.symmetry.subgroups import _standard_input

        numpy = self._numpy
        points: dict[_Class, list[Any]] = {}
        owners: dict[_Class, list[int]] = {}
        condition = 1.0
        scale = 1.0
        group: int | None = None
        retained = 0
        for index, value in enumerate(values):
            reference = _reference(value, cache)
            if group is None:
                group = reference.spacegroup.it_number
            if reference.spacegroup.it_number != group or not reference.transform_from_standard.is_identity():
                raise ValueError("grid requires one standard space group with identity transforms")
            basis = numpy.asarray(reference.cell.basis.to_floats(), dtype=float)
            condition = max(condition, _condition(basis, numpy))
            inverse = numpy.linalg.inv(basis)
            if retained + _point_count(reference) > self._max_points:
                raise ValueError("expanded point limit exceeded")
            orbits = _cartesian_orbits(reference, numpy)
            anchors: list[tuple[_Class, Any]] = []
            sizes: dict[_Class, int] = {}
            for site, orbit in zip(reference.wyckoff_sites, orbits, strict=True):
                key = (site.species, site.wyckoff)
                sizes[key] = sizes.get(key, 0) + len(orbit)
                if len(orbit):
                    anchors.append((key, orbit[0].copy()))
                    scale = max(scale, float(numpy.max(numpy.abs(orbit))))
                retained += len(orbit)
            if not anchors:
                raise ValueError("grid requires non-empty atom orbits")
            # Prefer positions with actual free parameters, then rare compatible
            # classes. Repeated sites remain safe: candidate points are never sorted
            # into a presumed one-to-one site correspondence.
            anchors.sort(
                key=lambda anchor: (
                    not reference.spacegroup.wyckoff_position(anchor[0][1]).free_count,
                    sizes[anchor[0]],
                    anchor[0],
                )
            )
            self._references.append(_Reference(basis, inverse, anchors[:4]))
            represented = _standard_input(rerepresent(reference, reference.spacegroup, tolerance=None))
            for candidate in _normalizer_candidates(represented):
                if retained + _point_count(candidate) > self._max_points:
                    raise ValueError("expanded point limit exceeded")
                candidate_basis = numpy.asarray(candidate.cell.basis.to_floats(), dtype=float)
                condition = max(condition, _condition(candidate_basis, numpy))
                scale = max(scale, float(numpy.max(numpy.abs(basis))), float(numpy.max(numpy.abs(candidate_basis))))
                for site, orbit in zip(candidate.wyckoff_sites, _cartesian_orbits(candidate, numpy), strict=True):
                    retained += len(orbit)
                    if retained > self._max_points:
                        raise ValueError("expanded point limit exceeded")
                    key = (site.species, site.wyckoff)
                    points.setdefault(key, []).extend(orbit)
                    owners.setdefault(key, []).extend([index] * len(orbit))
                    self._indexed_points += len(orbit)
                    if len(orbit):
                        scale = max(scale, float(numpy.max(numpy.abs(orbit))))
        # The comparison kernel subtracts nearly equal squared distances. Add a
        # deliberately generous scale/conditioning allowance before discretization;
        # ill-conditioned metrics use the unfiltered path instead.
        self._radius = math.sqrt(2.0) * delta + 1e-6 * condition**3 * scale
        arrays = {key: numpy.asarray(rows) for key, rows in points.items() if rows}
        if not arrays or not math.isfinite(self._radius):
            raise ValueError("non-finite or empty grid geometry")
        combined = numpy.concatenate(tuple(arrays.values()))
        if float(numpy.max(numpy.abs(combined))) / self._radius > 1e12:
            raise ValueError("grid coordinates exceed reliable integer resolution")
        if strategy == "first":
            order = [0, 1, 2]
        elif strategy == "variance":
            scores = numpy.var(combined, axis=0)
            order = sorted(range(3), key=lambda axis: (-float(scores[axis]), axis))
        else:
            scores = [
                sum(len(numpy.unique(numpy.floor(rows[:, axis] / self._radius))) for rows in arrays.values())
                for axis in range(3)
            ]
            order = sorted(range(3), key=lambda axis: (-scores[axis], axis))
        self._selected_axes = tuple(order[:dimensions])
        for key, rows in arrays.items():
            self._bounds[key] = (rows.min(axis=0), rows.max(axis=0))
            bins = numpy.floor(rows[:, self._selected_axes] / self._radius)
            buckets: dict[tuple[int, ...], set[int]] = {}
            for owner, row in zip(owners[key], bins, strict=True):
                buckets.setdefault(tuple(int(part) for part in row), set()).add(owner)
            self._buckets[key] = buckets

    def _query(self, index: int) -> frozenset[int] | None:
        if index in self._queries:
            self._queries.move_to_end(index)
            return self._queries[index]
        try:
            answer = self._neighbors(index)
        except (ValueError, ArithmeticError):
            # Numerical bounds that cannot be evaluated must never suppress a pair.
            answer = None
        self._queries[index] = answer
        self._query_entries += 0 if answer is None else len(answer)
        # Retain sparse rows across large leader scans; a fixed row cap repeatedly
        # evicts early leaders even when their candidate sets consume little memory.
        while len(self._queries) > self._max_points or self._query_entries > self._max_points:
            _, removed = self._queries.popitem(last=False)
            self._query_entries -= 0 if removed is None else len(removed)
        return answer

    def _neighbors(self, index: int) -> frozenset[int] | None:
        """Query periodic anchor neighborhoods without retaining a dense pair graph."""
        result: set[int] | None = None
        reference = self._references[index]
        numpy = self._numpy
        offsets = tuple(itertools.product((-1, 0, 1), repeat=len(self._selected_axes)))
        for key, point in reference.anchors:
            if key not in self._bounds:
                continue
            low, high = self._bounds[key]
            corners = numpy.asarray(
                tuple(itertools.product(*zip(low - self._radius, high + self._radius, strict=True)))
            )
            fractional = (corners - point) @ reference.inverse
            if not numpy.all(numpy.isfinite(fractional)):
                continue
            # Padding covers inverse/corner arithmetic and box-boundary rounding.
            padding = 1e-9 * max(1.0, float(numpy.max(numpy.abs(fractional))))
            lower = numpy.ceil(fractional.min(axis=0) - padding)
            upper = numpy.floor(fractional.max(axis=0) + padding)
            limits = tuple((int(start), int(stop)) for start, stop in zip(lower, upper, strict=True))
            if math.prod(max(0, stop - start + 1) for start, stop in limits) > self._max_images:
                continue
            ranges = tuple(range(start, stop + 1) for start, stop in limits)
            found: set[int] = set()
            buckets = self._buckets[key]
            for shift in itertools.product(*ranges):
                image = point + numpy.asarray(shift) @ reference.basis
                cell = tuple(math.floor(float(image[axis]) / self._radius) for axis in self._selected_axes)
                for offset in offsets:
                    found.update(buckets.get(tuple(a + b for a, b in zip(cell, offset, strict=True)), ()))
            result = found if result is None else result & found
            if not result:
                break
        return None if result is None or len(result) >= self._count else frozenset(result)


def _budget(value: object) -> float:
    """Validate a runtime real threshold without narrowing the annotated constructor."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("delta must be a finite non-negative real")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("delta must be a finite non-negative real")
    return result


def _point_count(structure: ASUStructure) -> int:
    """Count orbit points before allocating their expanded numerical arrays."""
    return sum(structure.spacegroup.wyckoff_position(site.wyckoff).multiplicity for site in structure.wyckoff_sites)


def _condition(basis: Any, numpy: Any) -> float:
    """Reject metrics whose numerical image bounds would be too uncertain."""
    if basis.shape != (3, 3) or not numpy.all(numpy.isfinite(basis)):
        raise ValueError("grid requires finite three-dimensional cells")
    condition = float(numpy.linalg.cond(basis))
    if not math.isfinite(condition) or condition > 100:
        raise ValueError("ill-conditioned cell metric")
    return condition


def _reference(value: _Value, cache: StructureComparisonCache) -> ASUStructure:
    """Prepare exactly the same source and standard reference as structure_delta."""
    from httk.atomistic.models.prototype.derived import _prototype_to_structure
    from httk.atomistic.symmetry.lift import rerepresent
    from httk.atomistic.symmetry.paths import _exact_asu, _validate, canonicalize_full
    from httk.atomistic.symmetry.spacegroup import Spacegroup
    from httk.atomistic.symmetry.subgroups import _standard_input

    if isinstance(value, (ASUStructure, FundamentalDomainStructure)):
        source = value
    elif isinstance(value, PrototypeAPI):
        template = value.representative
        if template is None:
            raise ValueError("a prototype has no geometry")
        source = cache._structure(template, lambda: _prototype_to_structure(template), kind="anonymous")
    elif isinstance(value, ProtostructureAPI):
        if value.representative is None:
            raise ValueError("a protostructure has no geometry")
        source = value.representative
    else:
        raise TypeError("grid requires structures, prototypes, or protostructures")
    structure = _exact_asu(source, "structure comparison grid")
    _validate(structure, "structure comparison grid")
    canonical = cache._structure(
        source, lambda: canonicalize_full(structure, structure.spacegroup, tolerance=None), kind="canonical"
    )
    if canonical.spacegroup.it_number != structure.spacegroup.it_number:
        raise ValueError("canonicalization changed the declared space group")
    return _standard_input(rerepresent(canonical, Spacegroup.standard(canonical.spacegroup.it_number), tolerance=None))
