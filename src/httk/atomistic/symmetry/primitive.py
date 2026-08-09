"""Express a structure in the fixed primitive cell of its standard setting.

The primitive basis is obtained from the IT standard-setting conventional cell with the
centring-dependent matrices documented by spglib. The matrices are stated here in their
column-vector convention and transposed for httk's row-vector cell representation.
"""

import fractions
from dataclasses import dataclass

from httk.core import FracVector, SurdVector, unwrap

from httk.atomistic.composition import Assembly, ChemicalComposition
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.semantics import initialize_semantics
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.symmetry.spacegroup import Spacegroup
from httk.atomistic.symmetry.standardization import ConventionalCellResult, conventional_cell

from ._standardization_common import (
    _as_existing_asu,
    _matrix_column_sum_factor,
    _matrix_row_sum_factor,
    _scaled_composition,
    _scaled_precision,
    _semantic_value,
)

__all__ = ["PrimitiveCellResult", "primitive_cell"]


# These are the exact column-vector matrices from the spglib definition. For the source and
# citation, see https://spglib.readthedocs.io/en/latest/definition.html#transformation-to-the-primitive-cell.
P = FracVector([["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]])
A = FracVector([["1", "0", "0"], ["0", "1/2", "-1/2"], ["0", "1/2", "1/2"]])
C = FracVector([["1/2", "1/2", "0"], ["-1/2", "1/2", "0"], ["0", "0", "1"]])
R = FracVector([["2/3", "-1/3", "-1/3"], ["1/3", "1/3", "-2/3"], ["1/3", "1/3", "1/3"]])
I = FracVector([["-1/2", "1/2", "1/2"], ["1/2", "-1/2", "1/2"], ["1/2", "1/2", "-1/2"]])
F = FracVector([["0", "1/2", "1/2"], ["1/2", "0", "1/2"], ["1/2", "1/2", "0"]])

_COLUMN_CENTRING_MATRICES = {"P": P, "A": A, "C": C, "I": I, "F": F, "R": R}
_ROW_CENTRING_MATRICES = {letter: matrix.T() for letter, matrix in _COLUMN_CENTRING_MATRICES.items()}


@dataclass(frozen=True, slots=True)
class PrimitiveCellResult:
    """Store a structure in the fixed primitive cell of its conventional cell.

    ``transform`` is the row-convention matrix actually applied to the conventional basis:
    ``basis_primitive = transform * basis_conventional``. The matrices are the transposes of
    spglib's documented column-vector matrices, where ``B_p = B_s P_c``; fractional row
    coordinates therefore transform as ``f_p = f_s * transform.inv()`` and are wrapped into
    ``[0, 1)``. See the `spglib primitive-cell definition
    <https://spglib.readthedocs.io/en/latest/definition.html#transformation-to-the-primitive-cell>`_.

    ``multiplier`` is the exact ratio of primitive-cell site count to input site count.

    :param structure: The resulting primitive-cell structure.
    :param spacegroup: The space group of the standardized input.
    :param conventional: The conventional-cell result used as input.
    :param transform: The row-convention matrix applied to the conventional basis.
    :param multiplier: The exact ratio of result site count to input site count.
    """

    structure: UnitcellStructure
    spacegroup: Spacegroup
    conventional: ConventionalCellResult
    transform: FracVector
    multiplier: fractions.Fraction


def _input_semantics(
    original: UnitcellStructureView,
    asu: object,
) -> tuple[
    bool,
    ChemicalComposition | None,
    str | None,
    str | None,
    str | None,
    tuple[Assembly, ...] | None,
]:
    source = original if asu is not None else unwrap(original)
    return (
        bool(_semantic_value(source, original, "molecular", False)),
        _semantic_value(source, original, "chemical_composition"),
        _semantic_value(source, original, "chemical_formula_descriptive"),
        _semantic_value(source, original, "chemical_formula_hill"),
        _semantic_value(source, original, "optimization_type"),
        _semantic_value(source, original, "assemblies"),
    )


def primitive_cell(
    structure: StructureLike,
    *,
    tolerance: float | None = None,
    limit_denominator: int | None = None,
) -> PrimitiveCellResult:
    """Return ``structure`` in the spglib-convention primitive cell.

    The input is first recognized or, when it already contains an asymmetric unit, used exactly
    as stored. Recognition arguments are rejected for an existing ASU. The recognized structure
    is converted to its IT standard-setting conventional cell by
    :func:`~httk.atomistic.conventional_cell`, then the fixed matrix for its centring type is
    applied exactly. Site moments and assemblies are refused because this operation cannot yet
    transform their frame or preserve correlated site groups through the centring collapse.
    This operation does not perform Niggli reduction.

    :param structure: The structure to express in a primitive cell.
    :param tolerance: The Cartesian recognition tolerance, or ``None`` to derive it.
    :param limit_denominator: The maximum denominator for idealised free parameters, or
        ``None`` to retain their exact stated values.
    :return: The primitive-cell structure and transform metadata.
    :raises ImportError: If recognition is needed and the optional spglib dependency is
        unavailable.
    :raises ValueError: If recognition arguments are invalid for the input, the structure
        has unsupported moments or assemblies, is not fully periodic, or has an unsupported
        centring type.
    """
    original = UnitcellStructureView(structure)
    asu = _as_existing_asu(structure)
    had_existing_asu = asu is not None
    if had_existing_asu and (tolerance is not None or limit_denominator is not None):
        raise ValueError("primitive_cell() tolerance and limit_denominator cannot be used with an existing ASU")

    molecular, source_composition, source_descriptive, source_hill, source_optimization, source_assemblies = (
        _input_semantics(original, asu)
    )
    if source_assemblies is not None:
        raise ValueError("primitive_cell does not support collapsing correlated site groups into a primitive cell")
    if asu is not None and any(site.moment is not None for site in asu.wyckoff_sites):
        raise ValueError("primitive_cell does not yet support structures with site moments; keep the original setting")
    if original.site_moments is not None:
        raise ValueError("primitive_cell does not yet support structures with site moments; keep the original setting")

    conventional = conventional_cell(
        structure,
        tolerance=tolerance,
        limit_denominator=limit_denominator,
    )
    spacegroup = conventional.spacegroup
    try:
        transform = _ROW_CENTRING_MATRICES[spacegroup.centring_type]
    except KeyError as error:
        raise ValueError(
            f"primitive_cell does not support unknown centring type {spacegroup.centring_type!r}"
        ) from error

    coordinate_matrix = transform.inv().simplify()
    cell_precision = _scaled_precision(
        conventional.structure.cell.precision,
        _matrix_row_sum_factor(transform),
    )
    coordinate_precision = _scaled_precision(
        conventional.structure.coordinate_precision,
        _matrix_column_sum_factor(coordinate_matrix),
    )
    primitive_cell_value = Cell(
        SurdVector(transform) * conventional.structure.cell.unscaled_basis,
        scale=conventional.structure.cell.scale,
        precision=cell_precision,
        periodicity=conventional.structure.cell.periodicity,
    )

    primitive_coordinates: list[FracVector] = []
    primitive_species: list[str] = []
    seen: set[tuple[tuple[fractions.Fraction, ...], str]] = set()
    for coordinate, species in zip(
        conventional.structure.sites.reduced_coords,
        conventional.structure.species_at_sites,
        strict=True,
    ):
        mapped = (coordinate * coordinate_matrix).normalize()
        key = (tuple(mapped.to_fractions()), species)
        if key not in seen:
            seen.add(key)
            primitive_coordinates.append(mapped)
            primitive_species.append(species)

    translations = spacegroup.centering_translations
    assert len(conventional.structure.sites) == len(primitive_coordinates) * len(translations)
    original_count = len(original.sites)
    if original_count == 0:
        raise ValueError("primitive_cell() cannot determine a site-count multiplier for an empty structure")
    multiplier = fractions.Fraction(len(primitive_coordinates), original_count)
    scaled_composition = _scaled_composition(source_composition, multiplier) if source_composition is not None else None
    charge = None if original.charge is None else original.charge * multiplier
    result = UnitcellStructure(
        primitive_cell_value,
        Sites(primitive_coordinates, precision=coordinate_precision),
        conventional.structure.species,
        primitive_species,
        molecular=molecular,
        chemical_composition=scaled_composition,
        chemical_formula_descriptive=source_descriptive,
        chemical_formula_hill=source_hill,
        optimization_type=source_optimization,
        charge=charge,
    )
    initialize_semantics(
        result,
        nsites=len(result.sites),
        molecular=molecular,
        assemblies=None,
        symmetry=None,
        chemical_composition=scaled_composition,
        chemical_formula_descriptive=source_descriptive,
        chemical_formula_hill=source_hill,
        optimization_type=source_optimization,
    )
    return PrimitiveCellResult(result, spacegroup, conventional, transform, multiplier)
