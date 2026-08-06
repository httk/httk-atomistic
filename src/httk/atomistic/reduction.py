"""Exact Niggli reduction of three-dimensional periodic cells and structures."""

import fractions
from dataclasses import dataclass
from typing import Any

from httk.core import FracVector, SurdVector

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.cell.like import CellLike
from httk.atomistic.models.cell.view import CellView
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.cartesian_view import CartesianSiteMomentsView
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.semantics import initialize_semantics
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

__all__ = [
    "NiggliReducedStructureResult",
    "NiggliReductionResult",
    "is_niggli_reduced",
    "niggli_reduce",
    "niggli_reduced",
]

_ZERO = fractions.Fraction(0)
_ONE = fractions.Fraction(1)
_MAX_STEPS = 10_000


@dataclass(frozen=True, slots=True)
class NiggliReductionResult:
    """A cell in exact Niggli-reduced form.

    ``transform`` uses the row-vector convention ``basis_reduced = transform * basis``.
    Its entries are integers and its determinant is +1. ``parameters`` contains the
    reduced ``(A, B, C, xi, eta, zeta)`` metric parameters as exact fractions.
    """

    cell: Cell
    transform: FracVector
    parameters: tuple[fractions.Fraction, ...]


@dataclass(frozen=True, slots=True)
class NiggliReducedStructureResult:
    """A structure whose cell and fractional coordinates are in Niggli form.

    ``transform`` uses the row-vector convention ``basis_reduced = transform * basis``;
    site coordinates are remapped by its exact inverse and wrapped into ``[0, 1)``.
    """

    structure: UnitcellStructure
    cell: Cell
    transform: FracVector


def _scaled_precision(
    precision: fractions.Fraction | None,
    factor: fractions.Fraction,
) -> fractions.Fraction | None:
    return None if precision is None else precision * factor


def _matrix_row_sum_factor(matrix: FracVector) -> fractions.Fraction:
    rows = matrix.to_fractions()
    return max(sum((abs(value) for value in row), start=_ZERO) for row in rows)


def _matrix_column_sum_factor(matrix: FracVector) -> fractions.Fraction:
    rows = matrix.to_fractions()
    return max(sum((abs(rows[row][column]) for row in range(3)), start=_ZERO) for column in range(3))


def _as_cell(cell: CellLike) -> Cell:
    value = cell if isinstance(cell, Cell) else CellView(cell)
    if value.periodicity != (True, True, True):
        raise ValueError(
            "Niggli reduction requires a fully 3D-periodic cell; this one is periodic in "
            f"{value.nperiodic_dimensions} of 3 directions ({value.periodicity})"
        )
    return value


def _rational_metric(cell: Cell) -> FracVector:
    metric = cell.metric()
    if not metric.is_rational:
        raise ValueError("Niggli reduction requires a rational-valued exact Gram matrix")
    return metric.coefficient(1)


def _parameters(metric: FracVector) -> tuple[fractions.Fraction, ...]:
    values = metric.to_fractions()
    return (
        values[0][0],
        values[1][1],
        values[2][2],
        2 * values[1][2],
        2 * values[0][2],
        2 * values[0][1],
    )


def _sign(value: fractions.Fraction) -> int:
    return 1 if value > _ZERO else -1


def _diagonal(values: tuple[int, int, int]) -> FracVector:
    return FracVector.create([[values[0], 0, 0], [0, values[1], 0], [0, 0, values[2]]])


def _is_identity(matrix: FracVector) -> bool:
    return matrix.to_fractions() == [[_ONE, _ZERO, _ZERO], [_ZERO, _ONE, _ZERO], [_ZERO, _ZERO, _ONE]]


def _step_matrix(parameters: tuple[fractions.Fraction, ...]) -> FracVector | None:
    a, b, c, xi, eta, zeta = parameters

    if a > b or (a == b and abs(xi) > abs(eta)):
        return FracVector.create([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    if b > c or (b == c and abs(eta) > abs(zeta)):
        return FracVector.create([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])

    product = xi * eta * zeta
    if product > _ZERO:
        return _diagonal((_sign(xi), _sign(eta), _sign(zeta)))

    i = j = k = 1
    zero_index: str | None = None
    if xi > _ZERO:
        i = -1
    elif xi == _ZERO:
        zero_index = "i"
    if eta > _ZERO:
        j = -1
    elif eta == _ZERO:
        zero_index = "j"
    if zeta > _ZERO:
        k = -1
    elif zeta == _ZERO:
        zero_index = "k"
    if i * j * k < 0:
        assert zero_index is not None
        if zero_index == "i":
            i = -1
        elif zero_index == "j":
            j = -1
        else:
            k = -1
    return _diagonal((i, j, k)) if (i, j, k) != (1, 1, 1) else None


def _niggli_step(parameters: tuple[fractions.Fraction, ...]) -> FracVector | None:
    a, b, _c, xi, eta, zeta = parameters
    matrix = _step_matrix(parameters)
    if matrix is not None and not _is_identity(matrix):
        return matrix

    if abs(xi) > b or (xi == b and 2 * eta < zeta) or (xi == -b and zeta < _ZERO):
        sign = _sign(xi)
        return FracVector.create([[1, 0, 0], [0, 1, 0], [0, -sign, 1]])
    if abs(eta) > a or (eta == a and 2 * xi < zeta) or (eta == -a and zeta < _ZERO):
        sign = _sign(eta)
        return FracVector.create([[1, 0, 0], [0, 1, 0], [-sign, 0, 1]])
    if abs(zeta) > a or (zeta == a and 2 * xi < eta) or (zeta == -a and eta < _ZERO):
        sign = _sign(zeta)
        return FracVector.create([[1, 0, 0], [-sign, 1, 0], [0, 0, 1]])
    total = xi + eta + zeta + a + b
    if total < _ZERO or (total == _ZERO and 2 * (a + eta) + zeta > _ZERO):
        return FracVector.create([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
    return None


def _is_niggli_parameters(parameters: tuple[fractions.Fraction, ...]) -> bool:
    a, b, c, xi, eta, zeta = parameters
    if not (a <= b <= c):
        return False
    if a == b and abs(xi) > abs(eta):
        return False
    if b == c and abs(eta) > abs(zeta):
        return False
    if not (abs(xi) <= b and abs(eta) <= a and abs(zeta) <= a):
        return False
    if not ((xi > _ZERO and eta > _ZERO and zeta > _ZERO) or (xi <= _ZERO and eta <= _ZERO and zeta <= _ZERO)):
        return False
    if xi == b and not 2 * eta >= zeta:
        return False
    if xi == -b and not zeta >= _ZERO:
        return False
    if eta == a and not 2 * xi >= zeta:
        return False
    if eta == -a and not zeta >= _ZERO:
        return False
    if zeta == a and not 2 * xi >= eta:
        return False
    if zeta == -a and not eta >= _ZERO:
        return False
    total = xi + eta + zeta + a + b
    return total >= _ZERO and (total != _ZERO or 2 * (a + eta) + zeta <= _ZERO)


def niggli_reduce(cell: CellLike) -> NiggliReductionResult:
    """Return the exact Niggli reduction of a fully periodic cell.

    The calculation uses the rational Gram matrix with no tolerance. The returned integer
    transform follows ``basis_reduced = transform * basis`` and has determinant +1.
    """
    source = _as_cell(cell)
    metric = _rational_metric(source)
    transform = FracVector.create([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for _ in range(_MAX_STEPS):
        matrix = _niggli_step(_parameters(metric))
        if matrix is None:
            break
        transform = matrix * transform
        metric = matrix * metric * matrix.T()
    else:
        raise RuntimeError("niggli_reduce did not converge after 10000 algorithm steps")

    assert transform.det().to_fraction() == _ONE
    parameters = _parameters(metric)
    assert _is_niggli_parameters(parameters)
    new_precision = _scaled_precision(source.precision, _matrix_row_sum_factor(transform))
    new_cell = Cell(
        SurdVector.create(transform) * source.unscaled_basis,
        scale=source.scale,
        precision=new_precision,
        periodicity=source.periodicity,
    )
    return NiggliReductionResult(new_cell, transform, parameters)


def is_niggli_reduced(cell: CellLike) -> bool:
    """Return whether a fully periodic cell satisfies the complete exact Niggli conditions."""
    source = _as_cell(cell)
    return _is_niggli_parameters(_parameters(_rational_metric(source)))


def _structure_moments(view: UnitcellStructureView) -> Any:
    moments = view.site_moments
    if moments is not None and getattr(moments, "kind", None) == "crystalaxis":
        cartesian = CartesianSiteMomentsView(moments)
        # Each unit-axis component has absolute value <= 1, so a column-sum bound is 3.
        precision = None if moments.precision is None else moments.precision * 3
        return CartesianSiteMoments(cartesian.cartesian_moments, precision=precision)
    return moments


def niggli_reduced(structure: StructureLike) -> NiggliReducedStructureResult:
    """Return a structure remapped into the exact Niggli-reduced cell.

    The site order and count are unchanged. With row-vector fractional coordinates, the
    exact remapping is ``f_reduced = f_original * transform.inv()`` followed by periodic
    normalization. Species, species-at-sites order, Cartesian site moments, molecular
    information, assemblies, chemical composition, formulas, optimization type, charge,
    and precision are carried unchanged or propagated. Symmetry is invalidated because
    its operations are basis-relative; immutable identifiers and last-modified metadata
    are invalidated because this operation creates a derived structure.
    """
    view = UnitcellStructureView(structure)
    reduction = niggli_reduce(view.cell)
    inverse = reduction.transform.inv().simplify()
    new_precision = _scaled_precision(
        view.sites.precision,
        _matrix_column_sum_factor(inverse),
    )
    new_sites = Sites((view.sites.reduced_coords * inverse).normalize(), precision=new_precision)
    result = UnitcellStructure(
        reduction.cell,
        new_sites,
        view.species,
        view.species_at_sites,
        site_moments=_structure_moments(view),
        molecular=view.molecular,
        assemblies=view.assemblies,
        chemical_composition=view.chemical_composition,
        chemical_formula_descriptive=view.chemical_formula_descriptive,
        chemical_formula_hill=view.chemical_formula_hill,
        optimization_type=view.optimization_type,
        charge=view.charge,
    )
    initialize_semantics(
        result,
        nsites=len(new_sites),
        molecular=result.molecular,
        assemblies=result.assemblies,
        symmetry=result.symmetry,
        chemical_composition=result.chemical_composition,
        chemical_formula_descriptive=result.chemical_formula_descriptive,
        chemical_formula_hill=result.chemical_formula_hill,
        optimization_type=result.optimization_type,
        immutable_id=result.immutable_id,
        last_modified=result.last_modified,
    )
    return NiggliReducedStructureResult(result, reduction.cell, reduction.transform)
