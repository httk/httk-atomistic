"""Exact supercell construction and deterministic cell-shape search.

Construction itself is rational/surd-exact. An integer matrix ``A`` maps the old
row-vector basis ``B`` to ``A * B`` and multiplies the number of sites by
``abs(det(A))``. The corresponding finite rational translation group supplies
every periodic image exactly; no geometric tolerance or open-ended cell search
is involved.

Automatic shape selection fixes that multiplier up front. A 50-digit Decimal
approximation of the exact Gram matrix seeds a bounded set of nearby integer
matrices, but candidates are ranked by exact, dimensionless Gram-matrix scores.
The result is therefore the best member of the documented candidate set, not a
claim of unrestricted global optimality.
"""

import decimal
import fractions
import itertools
from dataclasses import dataclass
from typing import Any

from httk.core import FracVector, SurdScalar, SurdVector, VectorLike, unwrap

from ._lattice import finite_translation_cosets
from ._periodicity_guard import require_full_periodicity
from .asu_structure import FundamentalDomainStructure
from .cell import Cell
from .composition import Assembly, ChemicalComposition
from .sites import Sites
from .structure import Structure
from .structure_like import StructureLike
from .unitcell_structure_view import UnitcellStructureView

__all__ = [
    "SupercellResult",
    "build_supercell",
    "cubic_supercell",
    "orthogonal_supercell",
]

DEFAULT_MAX_SITES = 100_000
_SEARCH_DIGITS = 50
_MAX_SEARCH_RADIUS = 2


@dataclass(frozen=True, slots=True)
class SupercellResult:
    """A materialized supercell together with its exact construction metadata.

    ``orthogonality_score`` is the sum of the squared pairwise cosines between
    cell vectors. ``cubicity_score`` is the squared Frobenius distance between
    the trace-normalized Gram matrix and the identity. Both are exact
    :class:`~httk.core.SurdScalar` values; zero proves the ideal shape exactly.
    """

    structure: Structure
    transformation: FracVector
    multiplier: int
    orthogonality_score: SurdScalar
    cubicity_score: SurdScalar


def _integer_transformation(transformation: VectorLike) -> tuple[FracVector, int]:
    matrix = FracVector.create(transformation).simplify()
    if matrix.dim != (3, 3):
        raise ValueError(f"a supercell transformation must be 3x3, got {matrix.dim}")
    if matrix.denom != 1:
        raise ValueError("a supercell transformation must contain only integers")
    determinant = matrix.det().to_fraction()
    if determinant == 0:
        raise ValueError("a supercell transformation must be nonsingular")
    return matrix, abs(determinant.numerator)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _tolerance_fraction(
    value: fractions.Fraction | str | float,
) -> fractions.Fraction:
    if isinstance(value, bool):
        raise ValueError("tolerance must be a non-negative number")
    try:
        exact = fractions.Fraction(str(value) if isinstance(value, float) else value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("tolerance must be a non-negative number") from error
    if exact < 0:
        raise ValueError("tolerance must be non-negative")
    return exact


def _validate_max_sites(max_sites: int | None) -> None:
    if max_sites is not None:
        _positive_integer(max_sites, "max_sites")


def _validate_site_count(nsites: int, multiplier: int, max_sites: int | None) -> None:
    _validate_max_sites(max_sites)
    predicted = nsites * multiplier
    if max_sites is not None and predicted > max_sites:
        raise ValueError(
            f"supercell would contain {predicted} sites ({nsites} × {multiplier}), "
            f"exceeding max_sites={max_sites}; pass a larger value or None to allow it"
        )


def _scaled_precision(
    precision: fractions.Fraction | None,
    factor: fractions.Fraction,
) -> fractions.Fraction | None:
    return None if precision is None else precision * factor


def _basis_precision_factor(matrix: FracVector) -> fractions.Fraction:
    rows = matrix.to_fractions()
    return max(sum((abs(value) for value in row), start=fractions.Fraction(0)) for row in rows)


def _coordinate_precision_factor(inverse: FracVector) -> fractions.Fraction:
    rows = inverse.to_fractions()
    return max(sum((abs(rows[i][j]) for i in range(3)), start=fractions.Fraction(0)) for j in range(3))


def _metric_element(metric: SurdVector, i: int, j: int) -> SurdScalar:
    return metric._element((i, j))


def _shape_scores(metric: SurdVector) -> tuple[SurdScalar, SurdScalar]:
    diagonal = [_metric_element(metric, i, i) for i in range(3)]
    orthogonality = (
        _metric_element(metric, 0, 1) * _metric_element(metric, 0, 1) / (diagonal[0] * diagonal[1])
        + _metric_element(metric, 0, 2) * _metric_element(metric, 0, 2) / (diagonal[0] * diagonal[2])
        + _metric_element(metric, 1, 2) * _metric_element(metric, 1, 2) / (diagonal[1] * diagonal[2])
    )._as_scalar()

    trace = (diagonal[0] + diagonal[1] + diagonal[2])._as_scalar()
    cubicity: SurdScalar | None = None
    for i in range(3):
        for j in range(3):
            normalized = (_metric_element(metric, i, j) * 3 / trace)._as_scalar()
            difference = (normalized - (1 if i == j else 0))._as_scalar()
            term = (difference * difference)._as_scalar()
            cubicity = term if cubicity is None else (cubicity + term)._as_scalar()
    assert cubicity is not None
    return orthogonality, cubicity


def _semantic_source(view: UnitcellStructureView) -> Any:
    raw = unwrap(view)
    # A symmetry-reduced source's domain assemblies are not indexed against the expanded
    # view. FundamentalDomainStructure exposes the checked, remapped unit-cell semantics instead.
    return view if isinstance(raw, FundamentalDomainStructure) else raw


def _scaled_composition(composition: ChemicalComposition | None, multiplier: int) -> ChemicalComposition | None:
    if composition is None:
        return None
    amounts = {element: amount * multiplier for element, amount in composition.amounts}
    precision = {
        element: None if width is None else width * multiplier for element, width in composition.amounts_precision
    }
    return ChemicalComposition(amounts, mode=composition.mode, amounts_precision=precision)


def _replicated_assemblies(
    assemblies: tuple[Assembly, ...] | None, nsites: int, multiplier: int
) -> tuple[Assembly, ...] | None:
    if assemblies is None:
        return None
    return tuple(
        Assembly(
            tuple(tuple(index + copy * nsites for index in group) for group in assembly.sites_in_groups),
            assembly.group_probabilities,
            assembly.group_probabilities_precision,
        )
        for copy in range(multiplier)
        for assembly in assemblies
    )


def build_supercell(
    structure: StructureLike,
    transformation: VectorLike,
    *,
    max_sites: int | None = DEFAULT_MAX_SITES,
) -> SupercellResult:
    """Build the exact supercell selected by an integer transformation matrix.

    Lattice vectors are rows and the returned basis is ``transformation * basis``.
    Reduced coordinates are transformed by the inverse matrix and wrapped into
    ``[0, 1)``. Any input representation is first presented as a full
    :class:`~httk.atomistic.structure.Structure`.

    Requires a fully 3D-periodic structure. Repeating a slab within its own plane is a
    perfectly sensible operation, but it is not this one: the transformation matrix here
    mixes all three rows and the coordinates are wrapped in all three directions, so applied
    to a reduced-periodicity cell it would generate images along a direction that has no
    lattice translation. Refused rather than half-supported.
    """
    matrix, multiplier = _integer_transformation(transformation)
    view = UnitcellStructureView(structure)
    require_full_periodicity(view.cell, "supercell construction")
    if view.cell.volume.is_zero():
        raise ValueError("supercell construction requires a nonsingular cell basis")
    _validate_site_count(len(view.sites), multiplier, max_sites)

    inverse = matrix.inv().simplify()
    cosets = finite_translation_cosets(inverse[index] for index in range(3))
    if len(cosets) != multiplier:
        raise RuntimeError(
            "internal supercell error: translation-coset count "
            f"{len(cosets)} does not match determinant multiplier {multiplier}"
        )

    transformed_coords: list[FracVector] = []
    if len(view.sites):
        base_coords = view.sites.reduced_coords * inverse
        for coset in cosets:
            transformed_coords.extend((coordinate + coset).normalize() for coordinate in base_coords)

    new_basis_precision = _scaled_precision(
        view.cell.precision,
        _basis_precision_factor(matrix),
    )
    new_coordinate_precision = _scaled_precision(
        view.sites.precision,
        _coordinate_precision_factor(inverse),
    )
    new_unscaled_basis = SurdVector.create(matrix) * view.cell.unscaled_basis
    new_cell = Cell(new_unscaled_basis, view.cell.scale, new_basis_precision)
    new_sites = Sites(transformed_coords, new_coordinate_precision)
    semantics = _semantic_source(view)
    assemblies = _replicated_assemblies(getattr(semantics, "assemblies", None), len(view.sites), multiplier)
    composition = getattr(semantics, "chemical_composition", None)
    result = Structure(
        new_cell,
        new_sites,
        view.species,
        view.species_at_sites * multiplier,
        molecular=bool(getattr(semantics, "molecular", False)),
        assemblies=assemblies,
        chemical_composition=_scaled_composition(composition, multiplier),
        chemical_formula_descriptive=getattr(semantics, "chemical_formula_descriptive", None),
        chemical_formula_hill=getattr(semantics, "chemical_formula_hill", None),
        optimization_type=getattr(semantics, "optimization_type", None),
    )
    orthogonality, cubicity = _shape_scores(new_cell.metric())
    return SupercellResult(result, matrix, multiplier, orthogonality, cubicity)


def _decimal_metric(cell: Cell) -> list[list[decimal.Decimal]]:
    metric = cell.metric()
    return [[_metric_element(metric, i, j).to_decimal(digits=_SEARCH_DIGITS) for j in range(3)] for i in range(3)]


def _cholesky(matrix: list[list[decimal.Decimal]]) -> list[list[decimal.Decimal]]:
    zero = decimal.Decimal(0)
    result = [[zero for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(i + 1):
            value = matrix[i][j] - sum((result[i][k] * result[j][k] for k in range(j)), start=zero)
            if i == j:
                if value <= 0:
                    raise ValueError("a supercell search requires a positive-definite cell metric")
                result[i][j] = value.sqrt()
            else:
                result[i][j] = value / result[j][j]
    return result


def _inverse_lower(matrix: list[list[decimal.Decimal]]) -> list[list[decimal.Decimal]]:
    zero = decimal.Decimal(0)
    one = decimal.Decimal(1)
    inverse = [[zero for _ in range(3)] for _ in range(3)]
    for column in range(3):
        for row in range(3):
            rhs = one if row == column else zero
            rhs -= sum((matrix[row][k] * inverse[k][column] for k in range(row)), start=zero)
            inverse[row][column] = rhs / matrix[row][row]
    return inverse


def _ideal_transformation(cell: Cell, multiplier: int) -> list[list[decimal.Decimal]]:
    with decimal.localcontext() as context:
        context.prec = _SEARCH_DIGITS
        canonical_basis = _cholesky(_decimal_metric(cell))
        inverse = _inverse_lower(canonical_basis)
        volume = cell.volume.to_decimal(digits=_SEARCH_DIGITS)
        one_third = decimal.Decimal(1) / decimal.Decimal(3)
        target_length = context.power(decimal.Decimal(multiplier) * volume, one_third)
        return [[target_length * inverse[i][j] for j in range(3)] for i in range(3)]


def _determinant(values: tuple[int, ...]) -> int:
    return (
        values[0] * values[4] * values[8]
        + values[1] * values[5] * values[6]
        + values[2] * values[3] * values[7]
        - values[2] * values[4] * values[6]
        - values[1] * values[3] * values[8]
        - values[0] * values[5] * values[7]
    )


def _diagonal_fallbacks(multiplier: int) -> list[tuple[int, ...]]:
    candidates: list[tuple[int, ...]] = []
    for first in range(1, multiplier + 1):
        if multiplier % first:
            continue
        remaining = multiplier // first
        for second in range(1, remaining + 1):
            if remaining % second:
                continue
            third = remaining // second
            candidates.append((first, 0, 0, 0, second, 0, 0, 0, third))
    return candidates


def _candidate_values(
    ideal: list[list[decimal.Decimal]],
    multiplier: int,
    search_radius: int,
) -> itertools.chain[tuple[int, ...]]:
    center = tuple(int(value.to_integral_value(rounding=decimal.ROUND_HALF_EVEN)) for row in ideal for value in row)
    offsets = range(-search_radius, search_radius + 1)
    local = (
        tuple(center[index] + delta[index] for index in range(9)) for delta in itertools.product(offsets, repeat=9)
    )
    return itertools.chain(local, _diagonal_fallbacks(multiplier))


def _search_transformation(
    cell: Cell,
    multiplier: int,
    search_radius: int,
    *,
    target: str,
) -> FracVector:
    if isinstance(search_radius, bool) or not isinstance(search_radius, int):
        raise ValueError("search_radius must be an integer")
    if search_radius < 0 or search_radius > _MAX_SEARCH_RADIUS:
        raise ValueError(f"search_radius must be between 0 and {_MAX_SEARCH_RADIUS}")

    ideal = _ideal_transformation(cell, multiplier)
    ideal_flat = tuple(value for row in ideal for value in row)
    seen: set[tuple[int, ...]] = set()
    best_values: tuple[int, ...] | None = None
    best_key: tuple[Any, ...] | None = None

    for values in _candidate_values(ideal, multiplier, search_radius):
        if values in seen or _determinant(values) != multiplier:
            continue
        seen.add(values)
        matrix = FracVector.create((values[0:3], values[3:6], values[6:9]))
        transformed_metric = SurdVector.create(matrix) * cell.metric() * SurdVector.create(matrix.T())
        orthogonality, cubicity = _shape_scores(transformed_metric)
        ideal_distance = sum(
            (decimal.Decimal(value) - target_value) ** 2 for value, target_value in zip(values, ideal_flat, strict=True)
        )
        coefficient_norm = sum(value * value for value in values)
        if target == "orthogonal":
            key = (orthogonality, cubicity, ideal_distance, coefficient_norm, values)
        else:
            key = (cubicity, orthogonality, ideal_distance, coefficient_norm, values)
        if best_key is None or key < best_key:
            best_key = key
            best_values = values

    if best_values is None:
        raise RuntimeError("internal supercell search error: no determinant-matching candidate was generated")
    return FracVector.create((best_values[0:3], best_values[3:6], best_values[6:9]))


def _transformation_scores(cell: Cell, transformation: FracVector) -> tuple[SurdScalar, SurdScalar]:
    transformed_metric = SurdVector.create(transformation) * cell.metric() * SurdVector.create(transformation.T())
    return _shape_scores(transformed_metric)


def _tolerance_supercell(
    view: UnitcellStructureView,
    tolerance: fractions.Fraction,
    max_multiplier: int,
    search_radius: int,
    max_sites: int | None,
    *,
    target: str,
) -> SupercellResult:
    _validate_max_sites(max_sites)
    nsites = len(view.sites)
    if max_sites is not None and nsites > max_sites:
        _validate_site_count(nsites, 1, max_sites)

    score_name = "orthogonality_score" if target == "orthogonal" else "cubicity_score"
    best_multiplier: int | None = None
    best_score: SurdScalar | None = None
    last_multiplier = 0
    for multiplier in range(1, max_multiplier + 1):
        if max_sites is not None and nsites * multiplier > max_sites:
            break
        last_multiplier = multiplier
        transformation = _search_transformation(
            view.cell,
            multiplier,
            search_radius,
            target=target,
        )
        orthogonality, cubicity = _transformation_scores(view.cell, transformation)
        score = orthogonality if target == "orthogonal" else cubicity
        if best_score is None or score < best_score:
            best_multiplier = multiplier
            best_score = score
        if score <= tolerance:
            return build_supercell(view, transformation, max_sites=max_sites)

    if best_score is None or best_multiplier is None:
        raise ValueError(f"no {target} supercell multipliers could be tried within max_sites={max_sites}")
    raise ValueError(
        f"no {target} supercell met tolerance {tolerance} for multipliers 1..{last_multiplier} "
        f"(max_multiplier={max_multiplier}, max_sites={max_sites}); "
        f"best was (multiplier={best_multiplier}, {score_name}={float(best_score)})"
    )


def orthogonal_supercell(
    structure: StructureLike,
    multiplier: int | None = None,
    *,
    tolerance: fractions.Fraction | str | float | None = None,
    max_multiplier: int | None = None,
    search_radius: int = 1,
    max_sites: int | None = DEFAULT_MAX_SITES,
) -> SupercellResult:
    """Build the most orthogonal supercell in the bounded candidate set.

    Exactly one of ``multiplier`` and ``tolerance`` must be provided. With
    ``multiplier``, it is the exact number of source cells in the result.
    With ``tolerance``, the multiplier is increased from one until the exact
    orthogonality score is at most the given bound, up to ``max_multiplier``.
    Candidate matrices are centered on the ideal cubic real-valued transform
    and vary each integer entry by at most ``search_radius`` (0--2); diagonal
    factorizations provide guaranteed determinant-matching fallbacks. Exact
    orthogonality is ranked first and cubicity breaks equal-shape ties.
    """
    if (multiplier is None) == (tolerance is None):
        raise ValueError("provide exactly one of multiplier or tolerance")
    if tolerance is not None:
        view = UnitcellStructureView(structure)
        bound = _tolerance_fraction(tolerance)
        limit = 32 if max_multiplier is None else _positive_integer(max_multiplier, "max_multiplier")
        return _tolerance_supercell(view, bound, limit, search_radius, max_sites, target="orthogonal")
    assert multiplier is not None
    multiplier = _positive_integer(multiplier, "multiplier")
    view = UnitcellStructureView(structure)
    _validate_site_count(len(view.sites), multiplier, max_sites)
    transformation = _search_transformation(view.cell, multiplier, search_radius, target="orthogonal")
    return build_supercell(view, transformation, max_sites=max_sites)


def cubic_supercell(
    structure: StructureLike,
    multiplier: int | None = None,
    *,
    tolerance: fractions.Fraction | str | float | None = None,
    max_multiplier: int | None = None,
    search_radius: int = 1,
    max_sites: int | None = DEFAULT_MAX_SITES,
) -> SupercellResult:
    """Build the most cubic supercell in the bounded candidate set.

    Exactly one of ``multiplier`` and ``tolerance`` must be provided. With
    ``multiplier``, it is the exact number of source cells in the result.
    With ``tolerance``, the multiplier is increased from one until the exact
    cubicity score is at most the given bound, up to ``max_multiplier``.
    Candidate matrices are centered on the ideal cubic real-valued transform
    and vary each integer entry by at most ``search_radius`` (0--2); diagonal
    factorizations provide guaranteed determinant-matching fallbacks.
    """
    if (multiplier is None) == (tolerance is None):
        raise ValueError("provide exactly one of multiplier or tolerance")
    if tolerance is not None:
        view = UnitcellStructureView(structure)
        bound = _tolerance_fraction(tolerance)
        limit = 32 if max_multiplier is None else _positive_integer(max_multiplier, "max_multiplier")
        return _tolerance_supercell(view, bound, limit, search_radius, max_sites, target="cubic")
    assert multiplier is not None
    multiplier = _positive_integer(multiplier, "multiplier")
    view = UnitcellStructureView(structure)
    _validate_site_count(len(view.sites), multiplier, max_sites)
    transformation = _search_transformation(view.cell, multiplier, search_radius, target="cubic")
    return build_supercell(view, transformation, max_sites=max_sites)
