"""Exact symmetry-preserving structure alignment and interpolation."""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import cache

from httk.core import FracVector, SurdVector, register_citation

from httk.atomistic import data
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure, WyckoffSite
from httk.atomistic.symmetry._periodicity_guard import require_full_periodicity
from httk.atomistic.symmetry._standardization_common import (
    _matrix_column_sum_factor,
    _matrix_row_sum_factor,
    _scaled_precision,
)
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.lift import (
    _apply_normalizer_operation,
    _demote_sites,
    _discrete_normalizer_translations,
    _translation_normal_form,
    _wrapped,
    rerepresent,
)
from httk.atomistic.symmetry.lift import (
    _canonical_sites as _orbit_canonical_sites,
)
from httk.atomistic.symmetry.lift import (
    _site_key as _orbit_site_key,
)
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup
from httk.atomistic.symmetry.subgroups import _standard_input, subgroup_closure

__all__ = [
    "CommonSubgroupResult",
    "StructurePath",
    "canonicalize_full",
    "common_subgroup_representation",
    "interpolate_structures",
    "list_representations",
    "represent_like",
    "structure_delta",
]

_MAX_PAIRING_PERMUTATIONS = 40_320


@cache
def _register_subgroup_matching_citation() -> None:
    """Register the subgroup-matching thesis citation, once per process."""
    register_citation(
        applies_to=(
            "The structure-matching and symmetry-path features (represent_like, "
            "common_subgroup_representation, structure_delta, interpolate_structures) build on Edvard "
            "Valentin's subgroup-matching work for httk v1"
        ),
        references={
            "authors": ({"name": "Edvard Valentin"},),
            "title": "Connecting Crystal Structures by Symmetry via Subgroup Matching",
            "school": "Linköping University",
            "year": "2024",
            "note": "Master's thesis, urn:nbn:se:liu:diva-207867",
            "url": "https://urn.kb.se/resolve?urn=urn:nbn:se:liu:diva-207867",
            "bib_type": "mastersthesis",
        },
    )


@dataclass(frozen=True, slots=True)
class CommonSubgroupResult:
    """Two aligned structures in their highest common subgroup.

    :param first: The first input represented in the common subgroup's standard setting.
    :param second: The second input represented and aligned to ``first``.
    :param spacegroup: The selected highest common subgroup in standard setting.
    """

    first: ASUStructure
    second: ASUStructure
    spacegroup: Spacegroup


@dataclass(frozen=True, slots=True)
class StructurePath:
    """A finite exact interpolation path between two aligned asymmetric units.

    :param frames: The endpoint-inclusive asymmetric-unit frames.
    :param spacegroup: The shared space group and setting of all frames.
    :param start: The first frame.
    :param end: The last frame.
    """

    frames: tuple[ASUStructure, ...]
    spacegroup: Spacegroup
    start: ASUStructure
    end: ASUStructure


@dataclass(frozen=True, slots=True)
class _Alignment:
    structure: ASUStructure
    pairs: tuple[tuple[int, int], ...]


def _validate(structure: ASUStructure, operation: str) -> None:
    require_full_periodicity(structure.cell, operation)
    if any(site.moment is not None for site in structure.wyckoff_sites):
        raise ValueError(f"{operation} does not support structures with site moments")
    if structure.assemblies is not None:
        raise ValueError(f"{operation} does not support structures with assemblies")
    if structure.molecular:
        raise ValueError(f"{operation} does not support molecular structures")


def _exact_asu(
    structure: ASUStructure | FundamentalDomainStructure,
    operation: str,
) -> ASUStructure:
    """Promote a clean fundamental domain to the exact ASU path representation."""
    if isinstance(structure, ASUStructure):
        return structure
    if not isinstance(structure, FundamentalDomainStructure):
        raise TypeError(
            f"{operation} requires ASUStructure or FundamentalDomainStructure, got {type(structure).__name__}"
        )
    # This forces the fundamental domain's exact expansion proof before copying its
    # declared representation. No symmetry recognition or coordinate snapping occurs.
    structure.expand_sites()
    return ASUStructure(
        structure.cell,
        structure.spacegroup,
        structure.wyckoff_sites,
        structure.species,
        transform=structure.transform,
        coordinate_precision=structure.coordinate_precision,
        molecular=structure.molecular,
        assemblies=structure.assemblies,
        charge=structure.charge,
    )


def _species_by_name(structure: ASUStructure) -> dict[str, Species]:
    return {species.name: species for species in structure.species}


def _species_signature(structure: ASUStructure) -> tuple[tuple[str, Species], ...]:
    return tuple(
        sorted(((species.name, species) for species in structure.species), key=lambda item: (item[0], repr(item[1])))
    )


def _signature(structure: ASUStructure) -> tuple[tuple[Species, str, int], ...]:
    species = _species_by_name(structure)
    entries = [
        (
            species[site.species],
            site.wyckoff,
            structure.spacegroup.wyckoff_position(site.wyckoff).multiplicity,
        )
        for site in structure.wyckoff_sites
    ]
    return tuple(sorted(entries, key=lambda item: (item[0].name, item[1], item[2], repr(item[0]))))


def _site_key(site: WyckoffSite) -> tuple[str, str, tuple[Fraction, ...]]:
    return site.species, site.wyckoff, tuple(Fraction(value) for value in site.free_params.to_fractions())


def _canonical_sites(sites: tuple[WyckoffSite, ...]) -> tuple[tuple[str, str, tuple[Fraction, ...]], ...]:
    return tuple(sorted(_site_key(site) for site in sites))


def _classes(structure: ASUStructure) -> dict[tuple[Species, str], tuple[int, ...]]:
    species = _species_by_name(structure)
    result: dict[tuple[Species, str], list[int]] = {}
    for index, site in enumerate(structure.wyckoff_sites):
        result.setdefault((species[site.species], site.wyckoff), []).append(index)
    return {key: tuple(value) for key, value in result.items()}


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    """Return a three-dimensional dot product at the float metric boundary."""
    return math.fsum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


def _scaled_subtract(
    left: tuple[float, float, float], scale: float, right: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Return ``left - scale * right`` for three-dimensional float vectors."""
    return (
        left[0] - scale * right[0],
        left[1] - scale * right[1],
        left[2] - scale * right[2],
    )


def _basis_rows(basis: SurdVector) -> tuple[tuple[float, float, float], ...]:
    """Convert an exact three-dimensional basis at the metric boundary."""
    values = tuple(tuple(float(value) for value in row) for row in basis.to_floats())
    if len(values) != 3 or any(len(row) != 3 for row in values):
        raise ValueError("structure_delta requires a three-dimensional cell")
    return tuple((row[0], row[1], row[2]) for row in values)


def _mean_endpoint_gram(
    first: tuple[tuple[float, float, float], ...], second: tuple[tuple[float, float, float], ...]
) -> tuple[tuple[float, float, float], ...]:
    """Return the symmetric arithmetic mean of two endpoint cell metrics."""
    gram = tuple(
        tuple((_dot(first[row], first[column]) + _dot(second[row], second[column])) / 2.0 for column in range(3))
        for row in range(3)
    )
    if not all(math.isfinite(value) for row in gram for value in row):
        raise ValueError("structure_delta requires a finite cell basis")
    return tuple((row[0], row[1], row[2]) for row in gram)


def _cholesky_basis(gram: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    """Return lower ``L`` with ``L L.T == gram`` for a positive 3D metric."""
    first_diagonal = gram[0][0]
    if first_diagonal <= 0.0:
        raise ValueError("structure_delta requires a non-singular cell basis")
    lower00 = math.sqrt(first_diagonal)
    lower10 = gram[1][0] / lower00
    lower20 = gram[2][0] / lower00
    second_diagonal = gram[1][1] - lower10**2
    if second_diagonal <= 0.0:
        raise ValueError("structure_delta requires a non-singular cell basis")
    lower11 = math.sqrt(second_diagonal)
    lower21 = (gram[2][1] - lower20 * lower10) / lower11
    third_diagonal = gram[2][2] - lower20**2 - lower21**2
    if third_diagonal <= 0.0:
        raise ValueError("structure_delta requires a non-singular cell basis")
    lower22 = math.sqrt(third_diagonal)
    return (
        (lower00, 0.0, 0.0),
        (lower10, lower11, 0.0),
        (lower20, lower21, lower22),
    )


def _row_matrix_product(
    row: tuple[float, float, float], matrix: tuple[tuple[float, float, float], ...]
) -> tuple[float, float, float]:
    """Return a three-dimensional row-vector matrix product."""
    return (
        math.fsum(row[index] * matrix[index][0] for index in range(3)),
        math.fsum(row[index] * matrix[index][1] for index in range(3)),
        math.fsum(row[index] * matrix[index][2] for index in range(3)),
    )


def _solve_cholesky(
    lower: tuple[tuple[float, float, float], ...], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Solve ``(lower lower.T) x = right`` for a three-dimensional vector."""
    first = right[0] / lower[0][0]
    second = (right[1] - lower[1][0] * first) / lower[1][1]
    third = (right[2] - lower[2][0] * first - lower[2][1] * second) / lower[2][2]
    result_third = third / lower[2][2]
    result_second = (second - lower[2][1] * result_third) / lower[1][1]
    return (
        (first - lower[1][0] * result_second - lower[2][0] * result_third) / lower[0][0],
        result_second,
        result_third,
    )


def _nearest_lattice_distance(
    displacement: tuple[float, float, float], basis: tuple[tuple[float, float, float], ...]
) -> float:
    """Return the exact-lattice minimum image of ``displacement`` in a 3D basis.

    The basis is converted to floats only here, at the final metric boundary.  A backwards
    Gram--Schmidt branch-and-bound search solves the closest-vector problem over every
    integer lattice translation.  Unlike a fixed ``[-1, 1]^3`` image box, its finite bounds
    follow from the current best Cartesian distance and remain correct for arbitrarily skew
    non-singular cells.
    """
    lattice = basis
    if not all(math.isfinite(value) for row in lattice for value in row):
        raise ValueError("structure_delta requires a finite cell basis")

    orthogonal: list[tuple[float, float, float]] = []
    squared_norms: list[float] = []
    coefficients = [[0.0] * 3 for _ in range(3)]
    for index, vector in enumerate(lattice):
        remainder = vector
        for previous, previous_vector in enumerate(orthogonal):
            coefficient = _dot(vector, previous_vector) / squared_norms[previous]
            coefficients[index][previous] = coefficient
            remainder = _scaled_subtract(remainder, coefficient, previous_vector)
        squared_norm = _dot(remainder, remainder)
        if not math.isfinite(squared_norm) or squared_norm <= 0.0:
            raise ValueError("structure_delta requires a non-singular cell basis")
        orthogonal.append(remainder)
        squared_norms.append(squared_norm)

    target = tuple(_dot(displacement, vector) / squared_norm for vector, squared_norm in zip(orthogonal, squared_norms))
    if not all(math.isfinite(value) for value in target):
        raise ValueError("structure_delta produced a non-finite lattice target")

    # Babai's nearest-plane result gives a finite initial radius for the exhaustive search.
    babai = [0, 0, 0]
    for index in range(2, -1, -1):
        center = target[index] - math.fsum(babai[later] * coefficients[later][index] for later in range(index + 1, 3))
        babai[index] = round(center)

    def residual_squared(integers: tuple[int, int, int]) -> float:
        return math.fsum(
            squared_norms[index]
            * (
                target[index]
                - integers[index]
                - math.fsum(integers[later] * coefficients[later][index] for later in range(index + 1, 3))
            )
            ** 2
            for index in range(3)
        )

    best = residual_squared((babai[0], babai[1], babai[2]))
    if not math.isfinite(best):
        raise ValueError("structure_delta produced a non-finite lattice distance")

    def search(index: int, chosen: list[int], accumulated: float) -> None:
        nonlocal best
        if index < 0:
            best = min(best, accumulated)
            return
        remaining = best - accumulated
        if remaining < 0.0:
            return
        center = target[index] - math.fsum(chosen[later] * coefficients[later][index] for later in range(index + 1, 3))
        radius = math.sqrt(remaining / squared_norms[index])
        # Widen the mathematical interval by one representable float so a boundary optimum
        # survives roundoff in the QR arithmetic.
        radius = math.nextafter(radius, math.inf)
        lower = math.ceil(center - radius)
        upper = math.floor(center + radius)
        values = range(lower, upper + 1)
        for value in sorted(values, key=lambda candidate: (abs(candidate - center), candidate)):
            contribution = squared_norms[index] * (center - value) ** 2
            if contribution <= remaining:
                chosen[index] = value
                search(index - 1, chosen, accumulated + contribution)

    search(2, [0, 0, 0], 0.0)
    return math.sqrt(best)


def _point_travel(first: FracVector, first_cell: Cell, second: FracVector, second_cell: Cell) -> float:
    """Return symmetric physical travel between two periodic endpoint positions.

    Endpoint coordinates are first converted with their *own* exact cells, retaining the
    physical displacement caused by a lattice deformation.  For one shared integer image
    ``n``, its squared travel is the arithmetic mean
    ``(||d + n B1||^2 + ||d + n B2||^2) / 2``, where ``d`` is the Cartesian endpoint
    displacement and ``B1``/``B2`` are the endpoint bases.  This symmetric endpoint-cell
    convention is invariant under reversal and simultaneous rigid rotation, and reduces to
    the ordinary minimum image for equal cells.  The finite closest-vector search completes
    this quadratic square in the mean endpoint metric, while the raw Cartesian displacement
    retains the contribution from lattice deformation.
    """
    first_rows = _basis_rows(SurdVector(first_cell.basis))
    second_rows = _basis_rows(SurdVector(second_cell.basis))
    first_cartesian = SurdVector(first) * first_cell.basis
    second_cartesian = SurdVector(second) * second_cell.basis
    displacement_values = tuple(float(value) for value in (first_cartesian - second_cartesian).to_floats())
    displacement = displacement_values[0], displacement_values[1], displacement_values[2]
    gram = _mean_endpoint_gram(first_rows, second_rows)
    lower = _cholesky_basis(gram)
    linear = tuple(
        _dot(
            displacement,
            (
                (first_rows[index][0] + second_rows[index][0]) / 2.0,
                (first_rows[index][1] + second_rows[index][1]) / 2.0,
                (first_rows[index][2] + second_rows[index][2]) / 2.0,
            ),
        )
        for index in range(3)
    )
    center = _solve_cholesky(lower, (-linear[0], -linear[1], -linear[2]))
    nearest_squared = _nearest_lattice_distance(_row_matrix_product(center, lower), lower) ** 2
    baseline = _dot(displacement, displacement) - _dot(center, _row_matrix_product(center, gram))
    squared = baseline + nearest_squared
    if squared < 0.0:
        roundoff = 1e-12 * max(1.0, abs(baseline), nearest_squared)
        if squared >= -roundoff:
            squared = 0.0
        else:
            raise ValueError("structure_delta produced a negative travel squared")
    return math.sqrt(squared)


def _orbit_travel(
    first: ASUStructure,
    first_site: WyckoffSite,
    second: ASUStructure,
    second_site: WyckoffSite,
) -> float:
    """Return the branch-wise Cartesian travel for one compatible pair of Wyckoff orbits."""
    first_coordinates = first.spacegroup.wyckoff_position(first_site.wyckoff).coordinates(first_site.free_params)
    second_coordinates = second.spacegroup.wyckoff_position(second_site.wyckoff).coordinates(second_site.free_params)
    if len(first_coordinates) != len(second_coordinates):
        raise ValueError("structures have incompatible Wyckoff orbit multiplicities")
    costs = tuple(
        tuple(
            _point_travel(FracVector(left), first.cell, FracVector(right), second.cell) for right in second_coordinates
        )
        for left in first_coordinates
    )
    return _minimum_assignment_cost(costs)


def _minimum_assignment(costs: tuple[tuple[float, ...], ...]) -> tuple[float, tuple[int, ...]]:
    """Return the deterministic minimum one-to-one cost for a square matrix.

    The complete atom sets of two Wyckoff orbits do not necessarily retain the same
    branch order after a cell re-expression. The Hungarian algorithm gives the required
    minimum physical pairing without the factorial site-orbit search used above.
    """
    count = len(costs)
    if count == 0:
        return 0.0, ()
    if any(len(row) != count for row in costs):
        raise ValueError("Wyckoff orbit cost matrix must be square")

    # 1-indexed implementation of the shortest-augmenting-path Hungarian algorithm.
    potential_left = [0.0] * (count + 1)
    potential_right = [0.0] * (count + 1)
    matched_left = [0] * (count + 1)
    predecessor = [0] * (count + 1)
    for left in range(1, count + 1):
        matched_left[0] = left
        right0 = 0
        minimum = [math.inf] * (count + 1)
        used = [False] * (count + 1)
        while True:
            used[right0] = True
            left0 = matched_left[right0]
            delta = math.inf
            right1 = 0
            for right in range(1, count + 1):
                if used[right]:
                    continue
                reduced = costs[left0 - 1][right - 1] - potential_left[left0] - potential_right[right]
                if reduced < minimum[right]:
                    minimum[right] = reduced
                    predecessor[right] = right0
                if minimum[right] < delta:
                    delta = minimum[right]
                    right1 = right
            for right in range(count + 1):
                if used[right]:
                    potential_left[matched_left[right]] += delta
                    potential_right[right] -= delta
                else:
                    minimum[right] -= delta
            right0 = right1
            if matched_left[right0] == 0:
                break
        while True:
            right1 = predecessor[right0]
            matched_left[right0] = matched_left[right1]
            right0 = right1
            if right0 == 0:
                break
    assignment = [0] * count
    for right in range(1, count + 1):
        assignment[matched_left[right] - 1] = right - 1
    return math.fsum(costs[left][right] for left, right in enumerate(assignment)), tuple(assignment)


def _minimum_assignment_cost(costs: tuple[tuple[float, ...], ...]) -> float:
    """Return the deterministic minimum one-to-one cost for a square matrix."""
    return _minimum_assignment(costs)[0]


def _pair_score(candidate: ASUStructure, reference: ASUStructure) -> tuple[Fraction, tuple[tuple[int, int], ...]]:
    """Pair compatible orbits by wrapped fractional free-parameter distance."""
    candidate_classes = _classes(candidate)
    reference_classes = _classes(reference)
    if candidate_classes.keys() != reference_classes.keys():
        raise ValueError("structures have incompatible site classes")

    score = Fraction(0)
    pairs: list[tuple[int, int]] = []
    for key in sorted(reference_classes, key=lambda item: (item[0].name, item[1], repr(item[0]))):
        reference_indices = reference_classes[key]
        candidate_indices = candidate_classes[key]
        if len(reference_indices) != len(candidate_indices):
            raise ValueError("structures have incompatible site classes")
        if len(reference_indices) > 1 and len(reference_indices) > _MAX_PAIRING_PERMUTATIONS:
            raise ValueError(f"pairing permutation bound exceeded for {key!r}; maximum is {_MAX_PAIRING_PERMUTATIONS}")
        best: tuple[Fraction, tuple[tuple[Fraction, ...], ...], tuple[int, ...]] | None = None
        for permutation in itertools.permutations(candidate_indices):
            distance = Fraction(0)
            parameter_key: list[tuple[Fraction, ...]] = []
            for reference_index, candidate_index in zip(reference_indices, permutation, strict=True):
                reference_params = reference.wyckoff_sites[reference_index].free_params.to_fractions()
                candidate_params = candidate.wyckoff_sites[candidate_index].free_params.to_fractions()
                distance += sum(
                    (
                        _wrapped(Fraction(right) - Fraction(left)) ** 2
                        for left, right in zip(reference_params, candidate_params)
                    ),
                    Fraction(0),
                )
                parameter_key.append(tuple(Fraction(value) for value in candidate_params))
            choice = (distance, tuple(parameter_key), tuple(permutation))
            if best is None or choice < best:
                best = choice
        assert best is not None
        score += best[0]
        pairs.extend(zip(reference_indices, best[2], strict=True))
    return score, tuple(pairs)


def _pair_travel_score(candidate: ASUStructure, reference: ASUStructure) -> tuple[float, tuple[tuple[int, int], ...]]:
    """Pair compatible Wyckoff orbits by their total physical Cartesian travel."""
    candidate_classes = _classes(candidate)
    reference_classes = _classes(reference)
    if candidate_classes.keys() != reference_classes.keys():
        raise ValueError("structures have incompatible site classes")

    score = 0.0
    pairs: list[tuple[int, int]] = []
    for key in sorted(reference_classes, key=lambda item: (item[0].name, item[1], repr(item[0]))):
        reference_indices = reference_classes[key]
        candidate_indices = candidate_classes[key]
        if len(reference_indices) != len(candidate_indices):
            raise ValueError("structures have incompatible site classes")
        costs = tuple(
            tuple(
                _orbit_travel(
                    reference,
                    reference.wyckoff_sites[reference_index],
                    candidate,
                    candidate.wyckoff_sites[candidate_index],
                )
                for candidate_index in candidate_indices
            )
            for reference_index in reference_indices
        )
        distance, assignment = _minimum_assignment(costs)
        score += distance
        pairs.extend(
            (reference_index, candidate_indices[candidate_offset])
            for reference_index, candidate_offset in zip(reference_indices, assignment, strict=True)
        )
    return score, tuple(pairs)


def _reference_setting(candidate: ASUStructure, reference: ASUStructure) -> ASUStructure:
    transform = reference.transform_from_standard
    basis_matrix = transform.matrix.T().inv()
    cell = Cell(
        transform.basis_to_setting(candidate.cell.basis),
        precision=_scaled_precision(candidate.cell.precision, _matrix_row_sum_factor(basis_matrix)),
        periodicity=candidate.cell.periodicity,
    )
    spacegroup = reference.spacegroup
    sites = candidate.wyckoff_sites
    residual = reference.transform
    if not spacegroup.is_standard_setting:
        mapped = []
        for site in sites:
            point = candidate.spacegroup.wyckoff_position(site.wyckoff).representative.coordinate(site.free_params)
            identified = spacegroup.identify_wyckoff(transform.to_setting(point).normalize())
            if identified is None:
                raise ValueError(f"cannot express Wyckoff site {site.wyckoff!r} in {spacegroup.setting}")
            position, parameters = identified
            mapped.append(WyckoffSite(position.letter, parameters, site.species))
        sites = tuple(mapped)
        residual = SettingTransform.identity()
    return ASUStructure(
        cell,
        spacegroup,
        sites,
        candidate.species,
        transform=residual,
        coordinate_precision=_scaled_precision(
            candidate.coordinate_precision,
            _matrix_column_sum_factor(transform.matrix.T()),
        ),
        charge=candidate.charge,
    )


def _normalizer_image(structure: ASUStructure, operation: AffineOperation) -> ASUStructure | None:
    return _apply_normalizer_operation(structure, operation)


def _aligned(
    end: ASUStructure,
    reference: ASUStructure,
    *,
    tolerance: float | None,
    pair_score: Callable[[ASUStructure, ASUStructure], tuple[object, tuple[tuple[int, int], ...]]] = _pair_score,
) -> _Alignment:
    _validate(end, "represent_like")
    _validate(reference, "represent_like")
    reference_standard = _standard_input(reference)
    represented = rerepresent(end, reference_standard.spacegroup, tolerance=tolerance)
    represented = _standard_input(represented)
    reference_signature = _signature(reference_standard)
    represented_signature = _signature(represented)
    if (
        _species_signature(represented) != _species_signature(reference_standard)
        or represented_signature != reference_signature
    ):
        raise ValueError(
            f"structures are not representable alike: signatures {represented_signature!r} and {reference_signature!r}"
        )

    candidates: dict[tuple[tuple[str, str, tuple[Fraction, ...]], ...], ASUStructure] = {
        _canonical_sites(represented.wyckoff_sites): represented
    }
    try:
        record = data.affine_normalizer_coset_record(represented.spacegroup.hall_entry)
    except KeyError:
        record = {}
    for coset in record.get("affine_normalizer_cosets", ()):
        if represented.spacegroup.crystal_system not in coset["compatible_systems"]:
            continue
        image = _normalizer_image(represented, AffineOperation.from_record(coset))
        if image is not None:
            candidates.setdefault(_canonical_sites(image.wyckoff_sites), image)

    best: (
        tuple[object, tuple[tuple[str, str, tuple[Fraction, ...]], ...], ASUStructure, tuple[tuple[int, int], ...]]
        | None
    ) = None
    for candidate in candidates.values():
        try:
            score, pairs = pair_score(candidate, reference_standard)
        except ValueError:
            continue
        choice = (score, _canonical_sites(candidate.wyckoff_sites), candidate, pairs)
        if best is None or choice[:2] < best[:2]:
            best = choice
    assert best is not None
    aligned = _reference_setting(best[2], reference)
    return _Alignment(aligned, best[3])


def represent_like(
    structure: ASUStructure,
    reference: ASUStructure,
    *,
    tolerance: float | None = None,
) -> ASUStructure:
    """Represent a structure in a reference's group and setting.

    The input is first sent through :func:`~httk.atomistic.symmetry.lift.rerepresent`, then equivalent affine-normalizer
    coset images of that one descent realization are scored against the reference. This is
    deliberately bounded: tabulated variants of alternate multi-hop descent paths are not
    enumerated because :func:`~httk.atomistic.symmetry.lift.rerepresent` exposes only its deterministic selected realization.
    Site pairing is brute force and capped at 40,320 permutations per class; larger classes
    require a future assignment solver.

    :param structure: The structure to represent.
    :param reference: The structure supplying the group, setting, and alignment target.
    :param tolerance: Cartesian tolerance passed to upward rerepresentation.
    :return: The input represented in the reference's group and setting.
    :raises ValueError: If the groups are unrelated, signatures differ, or the input is
        unsupported by the exact symmetry machinery.
    """
    _register_subgroup_matching_citation()
    return _aligned(structure, reference, tolerance=tolerance).structure


def _representation_gram(structure: ASUStructure) -> tuple[object, ...]:
    metric = structure.cell.metric()
    return tuple(metric._element((row, column)) for row in range(3) for column in range(3))


def _representation_orbit(structure: ASUStructure) -> tuple[ASUStructure, ...]:
    """Return every distinct representation reachable by the group's discrete affine normalizer.

    The images are the tabulated affine-normalizer cosets crossed with the runtime discrete
    Euclidean-normalizer translations -- exactly the crossing :func:`~httk.atomistic.symmetry.lift`'s
    normal form minimizes over, but enumerated instead of reduced to the least.  Each image is put in
    its continuous-normalizer translation-normal form, made right-handed where inversion re-describes
    the group (and dropped as the enantiomorph where it does not, for a Sohncke group), stored at its
    orbit-canonical Wyckoff representatives, then deduplicated by exact orbit-canonical site key and
    cell gram and sorted by that key.  This is the full set of representations modulo the continuous
    normalizer, for the discrete-normalizer freedom; representations differing by an untabulated
    conventional-cell re-choice (the A.5 recell-class freedom) are not generated.
    """
    structure = _demote_sites(structure)
    identity = FracVector.eye((3, 3))
    inversion = AffineOperation(FracVector(((-1, 0, 0), (0, -1, 0), (0, 0, -1))), (0, 0, 0))
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
    images: dict[tuple[object, ...], ASUStructure] = {}
    for operation in operations:
        image = _apply_normalizer_operation(structure, operation)
        if image is None:
            continue
        for translation in translations:
            shifted = (
                image
                if not any(translation)
                else _apply_normalizer_operation(image, AffineOperation(identity, FracVector(translation)))
            )
            if shifted is None:
                continue
            reduced = _translation_normal_form(shifted)
            if reduced.cell.basis.det().sign() < 0:
                flipped = _apply_normalizer_operation(reduced, inversion)
                if flipped is None:
                    # Inversion does not re-describe an enantiomorphic (Sohncke) group in its own
                    # setting, so a left-handed image is the enantiomorph -- a different crystal, not
                    # another representation of this one.  Drop it rather than emit a mirror twin.
                    continue
                reduced = _translation_normal_form(flipped)
            reduced = _orbit_canonical_sites(reduced)
            images.setdefault((_orbit_site_key(reduced), _representation_gram(reduced)), reduced)
    return tuple(images[key] for key in sorted(images))


def list_representations(
    structure: ASUStructure,
    target: Spacegroup | int,
    *,
    tolerance: float | None = None,
) -> tuple[ASUStructure, ...]:
    """Return every distinct representation of one crystal in a target group's standard setting.

    The crystal is first expressed once in ``target`` by
    :func:`~httk.atomistic.symmetry.lift.rerepresent` -- an exact descent for a subgroup target, a
    round-trip-gated lift for a supergroup target, itself for the same group -- and the full discrete
    affine-normalizer orbit of that one realization is then enumerated.  Every representation is
    returned in its continuous-translation normal form (otherwise a polar or triclinic target would
    have infinitely many), deduplicated by exact orbit-canonical site key and cell gram, and sorted by
    that key.

    **Scope.**  When ``target`` is the crystal's *own* full symmetry group, two representations differ
    only by an element of that group's affine normalizer, so this one orbit is the complete set --
    modulo the continuous normalizer and limited only by the bounded tabulated coset table.  For a
    PROPER-SUBGROUP (or supergroup) target only the normalizer orbit of the single
    :func:`~httk.atomistic.symmetry.lift.rerepresent` embedding is returned; inequivalent embeddings
    reachable by *other* descent chains -- the same crystal at the same cell size but a genuinely
    different site placement -- are deliberately out of scope and are NOT returned, because
    enumerating every chain is combinatorially explosive for deep targets (many tabulated chains), so
    a single canonical embedding is chosen.  Representations needing an untabulated conventional-cell
    re-choice (the A.5 recell-class freedom) are likewise not generated.  A supercell description is
    the same crystal in a larger cell; it too is not enumerated -- the exclusion there is "not a
    distinct representation at the same cell size", not "not the same crystal".

    This honors the explicit ``target`` exactly and never flips an enantiomorphic group; normalizing an
    enantiomorphic pair to its lower-numbered member is the closed-target canonicalizers'
    (:func:`~httk.atomistic.canonicalize`, :func:`~httk.atomistic.canonical_asu`) job.

    :param structure: The crystal, as an asymmetric-unit structure.
    :param target: The target space group or IT number.
    :param tolerance: Cartesian acceptance tolerance passed to any upward lift; ``None`` derives it.
    :return: The distinct representations in ``target``'s standard setting, sorted by canonical key.
    :raises ValueError: If ``target`` is unrelated to the crystal's group, or the input is
        unsupported by the exact symmetry machinery.
    """
    _validate(structure, "list_representations")
    standardized = _standard_input(structure)
    target_group = (target if isinstance(target, Spacegroup) else Spacegroup.standard(target)).standard_setting()
    base = _standard_input(rerepresent(standardized, target_group, tolerance=tolerance))
    return _representation_orbit(base)


def canonicalize_full(
    structure: ASUStructure,
    target: Spacegroup | int,
    *,
    tolerance: float | None = None,
) -> ASUStructure:
    """Return the canonically least representation of a crystal in a target group's standard setting.

    The least element, by exact orbit-canonical site key then cell gram, of
    :func:`list_representations`.  On the crystal's own group this is a normalizer-canonical form: it
    selects the same representative the upward search's normal form does, over the same discrete
    normalizer crossing and modulo the continuous quotient.  It is idempotent -- re-running it on its
    own result in the same target returns that result.

    This honors the explicit ``target`` exactly and never flips an enantiomorphic group; normalizing an
    enantiomorphic pair to its lower-numbered member is the closed-target canonicalizers'
    (:func:`~httk.atomistic.canonicalize`, :func:`~httk.atomistic.canonical_asu`) job.

    :param structure: The crystal, as an asymmetric-unit structure.
    :param target: The target space group or IT number.
    :param tolerance: Cartesian acceptance tolerance passed to any upward lift; ``None`` derives it.
    :return: The canonically least representation in ``target``'s standard setting.
    :raises ValueError: If ``target`` is unrelated, or the input is unsupported.
    """
    return list_representations(structure, target, tolerance=tolerance)[0]


def common_subgroup_representation(
    first: ASUStructure,
    second: ASUStructure,
    *,
    tolerance: float | None = None,
) -> CommonSubgroupResult:
    """Represent two structures in their highest common subgroup.

    Common subgroups are ordered by descending symmetry-operation count and then descending
    International Tables number. The first group for which both exact descents succeed is
    selected; the second structure is then aligned to the first by
    :func:`~httk.atomistic.symmetry.paths.represent_like`.

    :param first: The first structure.
    :param second: The second structure.
    :param tolerance: Cartesian tolerance passed to upward rerepresentation.
    :return: The two aligned structures and their selected common subgroup.
    :raises ValueError: If no common subgroup can represent both structures.
    """
    _register_subgroup_matching_citation()
    _validate(first, "common_subgroup_representation")
    _validate(second, "common_subgroup_representation")
    common = set(subgroup_closure(first.spacegroup, include_self=True)) & set(
        subgroup_closure(second.spacegroup, include_self=True)
    )
    ordered = sorted(
        common,
        key=lambda number: (-len(Spacegroup.standard(number).symmetry_operations), -number),
    )
    for number in ordered:
        target = Spacegroup.standard(number)
        try:
            first_child = _standard_input(rerepresent(first, target, tolerance=tolerance))
            second_child = _standard_input(rerepresent(second, target, tolerance=tolerance))
            second_aligned = represent_like(second_child, first_child, tolerance=tolerance)
        except ValueError:
            continue
        return CommonSubgroupResult(first_child, second_aligned, target)
    raise ValueError("no common subgroup representation succeeded")


def structure_delta(
    first: ASUStructure | FundamentalDomainStructure,
    second: ASUStructure | FundamentalDomainStructure,
    *,
    tolerance: float | None = None,
) -> float:
    """Return the total Cartesian atom travel between two compatible structures.

    Each exact asymmetric-unit or fundamental-domain input is first canonicalized in its
    declared space group, without symmetry recognition. The canonical structures are then expressed in every
    compatible common Bärnighausen subgroup, in the same descending-symmetry order used
    by :func:`~httk.atomistic.symmetry.paths.common_subgroup_representation`, and the second
    is aligned through the bounded affine-normalizer search used by
    :func:`~httk.atomistic.symmetry.paths.represent_like`. Compatible full
    :class:`~httk.atomistic.Species` and Wyckoff-letter classes are paired one-to-one,
    choosing the minimum total physical travel within each class.  Every member of each
    paired Wyckoff orbit contributes its shortest Cartesian distance to the total; the
    first and second positions are converted with their respective endpoint cells, so a
    lattice deformation contributes even when fractional coordinates do not change.

    Periodic endpoint images are selected by a finite closest-vector search in the
    arithmetic mean endpoint metric. This remains correct for skew cells and is symmetric
    when the endpoints are interchanged. The return value is a finite non-negative ``float``
    in the units of the cells' bases (ångström for ordinary crystallographic structures).

    Both directed bounded normalizer alignments are considered for each common subgroup,
    and their least travel is used.  This makes the metric symmetric without pretending to
    enumerate every possible Bärnighausen embedding. The subgroup and normalizer searches
    are deliberately bounded: only the deterministic subgroup embedding exposed by
    :func:`~httk.atomistic.symmetry.lift.rerepresent` and its tabulated normalizer images are
    considered. Atom and orbit
    assignment uses a deterministic Hungarian minimum-cost matching, so repeated
    Wyckoff classes do not require a factorial permutation search. Charges do not enter this
    geometrical metric.

    :param first: The first fully periodic, non-molecular asymmetric-unit or fundamental-domain structure.
    :param second: The second fully periodic, non-molecular asymmetric-unit or fundamental-domain structure.
    :param tolerance: Cartesian tolerance passed only to any required upward rerepresentation.
    :return: Total atom travel in the endpoint cells' length units.
    :raises ValueError: If the structures are unsupported, cannot be represented in a
        common subgroup, have incompatible species/Wyckoff classes, or yield a non-finite
        travel.
    """
    _register_subgroup_matching_citation()
    first = _exact_asu(first, "structure_delta")
    second = _exact_asu(second, "structure_delta")
    _validate(first, "structure_delta")
    _validate(second, "structure_delta")
    if first == second:
        return 0.0
    first_canonical = canonicalize_full(first, first.spacegroup, tolerance=tolerance)
    second_canonical = canonicalize_full(second, second.spacegroup, tolerance=tolerance)
    common = set(subgroup_closure(first_canonical.spacegroup, include_self=True)) & set(
        subgroup_closure(second_canonical.spacegroup, include_self=True)
    )
    ordered = sorted(
        common,
        key=lambda number: (-len(Spacegroup.standard(number).symmetry_operations), -number),
    )
    best: float | None = None
    for number in ordered:
        target = Spacegroup.standard(number)
        try:
            first_child = _standard_input(rerepresent(first_canonical, target, tolerance=tolerance))
            second_child = _standard_input(rerepresent(second_canonical, target, tolerance=tolerance))
        except ValueError:
            continue
        directed: list[float] = []
        for reference, candidate in ((first_child, second_child), (second_child, first_child)):
            try:
                alignment = _aligned(candidate, reference, tolerance=tolerance, pair_score=_pair_travel_score)
            except ValueError:
                continue
            delta = math.fsum(
                _orbit_travel(
                    reference,
                    reference.wyckoff_sites[reference_index],
                    alignment.structure,
                    alignment.structure.wyckoff_sites[candidate_index],
                )
                for reference_index, candidate_index in alignment.pairs
            )
            if not math.isfinite(delta):
                raise ValueError("structure_delta produced a non-finite travel")
            directed.append(delta)
        if not directed:
            continue
        delta = min(directed)
        if delta == 0.0:
            return 0.0
        if best is None or delta < best:
            best = delta
    if best is None:
        raise ValueError("no common subgroup representation succeeded")
    return best


def interpolate_structures(
    start: ASUStructure,
    end: ASUStructure,
    *,
    steps: int,
    tolerance: float | None = None,
) -> StructurePath:
    """Build an exact symmetry-preserving linear interpolation.

    Free parameters follow the wrapped shortest rational displacement and cell bases are
    linearly interpolated in the shared setting. Every intermediate frame is expanded so a
    collision with an already occupied orbit is reported with its step index. Frames carry
    the start structure's setting transform, while their Wyckoff parameters remain standard-
    setting values.

    :param start: The first endpoint.
    :param end: The second endpoint.
    :param steps: Number of endpoint-inclusive frames, at least two.
    :param tolerance: Cartesian tolerance passed to upward rerepresentation.
    :return: The exact interpolation path.
    :raises ValueError: If endpoints cannot be aligned, charges differ, or an intermediate
        frame is invalid.
    """
    _register_subgroup_matching_citation()
    if steps < 2:
        raise ValueError("interpolate_structures requires steps >= 2")
    _validate(start, "interpolate_structures")
    _validate(end, "interpolate_structures")
    start_standard = rerepresent(start, start.spacegroup, tolerance=tolerance)
    alignment = _aligned(end, start_standard, tolerance=tolerance)
    end_aligned = alignment.structure
    if set(start_standard.species) != set(end_aligned.species):
        raise ValueError("interpolation requires identical species definitions at both endpoints")
    if start_standard.charge != end_aligned.charge:
        raise ValueError("interpolation requires equal charges or both charges to be None")

    pairs = alignment.pairs
    frames: list[ASUStructure] = []
    last = steps - 1
    for index in range(steps):
        if index == 0:
            frames.append(start_standard)
            continue
        if index == last:
            frames.append(end_aligned)
            continue
        weight = Fraction(index, last)
        sites: list[WyckoffSite] = []
        for start_index, end_index in pairs:
            left = start_standard.wyckoff_sites[start_index]
            right = end_aligned.wyckoff_sites[end_index]
            parameters = [
                Fraction(left_value) + weight * _wrapped(Fraction(right_value) - Fraction(left_value))
                for left_value, right_value in zip(
                    left.free_params.to_fractions(), right.free_params.to_fractions(), strict=True
                )
            ]
            sites.append(WyckoffSite(left.wyckoff, FracVector(parameters), left.species))
        basis = (SurdVector(start_standard.cell.basis) * (1 - weight)) + (SurdVector(end_aligned.cell.basis) * weight)
        try:
            frame = ASUStructure(
                Cell(basis, periodicity=start_standard.cell.periodicity),
                start_standard.spacegroup,
                sites,
                start_standard.species,
                transform=start_standard.transform,
                coordinate_precision=start_standard.coordinate_precision,
                charge=start_standard.charge,
            )
            frame.expand_sites()
        except ValueError as error:
            raise ValueError(f"interpolation step {index}: {error}") from error
        frames.append(frame)
    return StructurePath(tuple(frames), start_standard.spacegroup, start_standard, end_aligned)
