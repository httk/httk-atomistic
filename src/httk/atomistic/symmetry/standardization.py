"""Express an asymmetric-unit structure in its IT standard-setting cell.

The operation is exact after any optional recognition step. An ASU already carries the
standard-setting Wyckoff data and the stored transform from that setting to its own cell,
so the conventional cell is obtained by transforming the cell basis back and expanding a
new identity-transform ASU.
"""

import fractions
from dataclasses import dataclass
from typing import Any

from httk.core import FracVector, unwrap

from httk.atomistic.composition import Assembly, ChemicalComposition
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.structure.asu import ASUStructure, WyckoffSite
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.semantics import initialize_semantics
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.symmetry.recognition import recognize_asu
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup

__all__ = ["ConventionalCellResult", "conventional_cell"]


@dataclass(frozen=True, slots=True)
class ConventionalCellResult:
    """A structure in its space group's IT standard-setting conventional cell.

    ``asu`` is the new standard-setting ASU that was expanded to make ``structure``.
    ``transform`` is the standard-to-own transform from the ASU that was supplied to, or
    recognized from, the operation; its orientation is :math:`f_own = f_std M^T + v`, so
    this operation undoes it for the cell basis. ``multiplier`` is the exact ratio of
    conventional-cell site count to input-cell site count. For the 527 vendored settings
    it is at least one; an untabulated, caller-supplied supercell transform may still
    produce a ratio below one.
    """

    structure: UnitcellStructure
    asu: ASUStructure
    spacegroup: Spacegroup
    transform: SettingTransform
    multiplier: fractions.Fraction


def _scaled_precision(
    precision: fractions.Fraction | None,
    factor: fractions.Fraction,
) -> fractions.Fraction | None:
    return None if precision is None else precision * factor


def _matrix_row_sum_factor(matrix: FracVector) -> fractions.Fraction:
    rows = matrix.to_fractions()
    return max(sum((abs(value) for value in row), start=fractions.Fraction(0)) for row in rows)


def _matrix_column_sum_factor(matrix: FracVector) -> fractions.Fraction:
    rows = matrix.to_fractions()
    return max(sum((abs(rows[row][column]) for row in range(3)), start=fractions.Fraction(0)) for column in range(3))


def _as_existing_asu(structure: StructureLike | ASUStructure) -> ASUStructure | None:
    """Return an already-held ASU through any backend/view unwrap chain."""
    candidate: object = structure
    visited: set[int] = set()
    while id(candidate) not in visited:
        visited.add(id(candidate))
        if isinstance(candidate, ASUStructure):
            return candidate
        direct = getattr(candidate, "asu", None)
        if isinstance(direct, ASUStructure):
            return direct
        unwrapped = unwrap(candidate)
        if unwrapped is candidate:
            break
        candidate = unwrapped
    return None


def _semantic_value(source: object, view: UnitcellStructureView, name: str, default: object = None) -> Any:
    return getattr(source, name, getattr(view, name, default))


def _scaled_composition(
    composition: ChemicalComposition | None, multiplier: fractions.Fraction
) -> ChemicalComposition | None:
    if composition is None:
        return None
    amounts = {element: amount * multiplier for element, amount in composition.amounts}
    precision = {
        element: None if width is None else width * multiplier for element, width in composition.amounts_precision
    }
    return ChemicalComposition(amounts, mode=composition.mode, amounts_precision=precision)


def _exact_site_bijection(
    original: UnitcellStructureView,
    standardized: UnitcellStructureView,
    transform: SettingTransform,
) -> tuple[int, ...] | None:
    """Map old site indices to new ones only when coordinates and species are unique and exact."""
    if len(original.sites) != len(standardized.sites):
        return None
    targets: dict[tuple[tuple[fractions.Fraction, ...], str], list[int]] = {}
    for index, (coordinate, species) in enumerate(
        zip(standardized.sites.reduced_coords, standardized.species_at_sites, strict=True)
    ):
        key = (tuple(coordinate.normalize().to_fractions()), species)
        targets.setdefault(key, []).append(index)
    mapping: list[int] = []
    used: set[int] = set()
    for coordinate, species in zip(original.sites.reduced_coords, original.species_at_sites, strict=True):
        mapped = transform.to_standard(coordinate).normalize()
        candidates = targets.get((tuple(mapped.to_fractions()), species), [])
        if len(candidates) != 1 or candidates[0] in used:
            return None
        mapping.append(candidates[0])
        used.add(candidates[0])
    return tuple(mapping) if len(used) == len(targets) else None


def _remap_assemblies(
    assemblies: tuple[Assembly, ...] | None,
    mapping: tuple[int, ...] | None,
) -> tuple[Assembly, ...] | None:
    if assemblies is None or not assemblies:
        return assemblies
    if mapping is None:
        raise ValueError("standardization cannot remap assemblies without an exact site bijection")
    return tuple(
        Assembly(
            tuple(tuple(mapping[index] for index in group) for group in assembly.sites_in_groups),
            assembly.group_probabilities,
            assembly.group_probabilities_precision,
        )
        for assembly in assemblies
    )


def conventional_cell(
    structure: StructureLike | ASUStructure,
    *,
    tolerance: float | None = None,
    limit_denominator: int | None = None,
) -> ConventionalCellResult:
    """Return ``structure`` in its space group's IT standard-setting conventional cell.

    An existing :class:`~httk.atomistic.ASUStructure` (including an
    :class:`~httk.atomistic.ASUStructureView`, an ASU backend, or a full-cell view backed
    by one) is used exactly as stored. Supplying ``tolerance`` or ``limit_denominator`` for
    that path raises :class:`ValueError`, because those arguments belong to recognition.
    Any other :class:`~httk.atomistic.StructureLike` is first passed to
    :func:`~httk.atomistic.recognize_asu`. That tolerant step may snap measured coordinates
    onto symmetry positions and chooses the transform recorded in the result; it does not
    preserve an unstated input transform or promise a :func:`~httk.atomistic.same_crystal`
    match to noisy input coordinates. The optional tolerance is a Cartesian matching
    distance and the optional denominator limit idealises free parameters.

    The returned ``transform`` is the existing ASU's transform, or the transform chosen by
    recognition for a plain input; the returned ``asu`` has an identity transform.
    Construction and expansion are exact, including the rhombohedral case where the
    standard hexagonal cell contains three primitive cells. Basis precision is multiplied by
    ``M.T()`` and coordinate precision by the maximum absolute column sum of
    ``inv(M.T())``; unknown precision remains unknown. Requires a fully 3D-periodic
    structure.
    """
    original = UnitcellStructureView(structure)
    asu = _as_existing_asu(structure)
    had_existing_asu = asu is not None
    source: object = original if asu is not None else unwrap(original)
    source_molecular = bool(_semantic_value(source, original, "molecular", False))
    source_assemblies = _semantic_value(source, original, "assemblies")
    if source_assemblies is not None:
        source_assemblies = tuple(source_assemblies)
    source_composition = _semantic_value(source, original, "chemical_composition")
    source_descriptive = _semantic_value(source, original, "chemical_formula_descriptive")
    source_hill = _semantic_value(source, original, "chemical_formula_hill")
    source_optimization = _semantic_value(source, original, "optimization_type")
    if asu is not None:
        if tolerance is not None or limit_denominator is not None:
            raise ValueError("conventional_cell() tolerance and limit_denominator cannot be used with an existing ASU")
    else:
        asu = recognize_asu(
            original,
            tolerance=tolerance,
            limit_denominator=limit_denominator,
        )

    assert asu is not None
    transform = asu.transform
    basis_matrix = transform.matrix.T()
    coordinate_matrix = basis_matrix.inv()
    new_cell_precision = _scaled_precision(
        asu.cell.precision,
        _matrix_row_sum_factor(basis_matrix),
    )
    new_coordinate_precision = _scaled_precision(
        asu.coordinate_precision,
        _matrix_column_sum_factor(coordinate_matrix),
    )
    new_cell = Cell(
        transform.basis_to_standard(asu.cell.basis),
        precision=new_cell_precision,
        periodicity=asu.cell.periodicity,
    )
    standard_sites = tuple(
        WyckoffSite(
            site.wyckoff,
            site.free_params,
            site.species,
            None if site.representative is None else transform.to_standard(site.representative).normalize(),
        )
        for site in asu.wyckoff_sites
    )
    standard_asu = ASUStructure(
        new_cell,
        asu.spacegroup,
        standard_sites,
        asu.species,
        transform=SettingTransform.identity(),
        coordinate_precision=new_coordinate_precision,
        molecular=source_molecular,
        # An existing reduced representation indexes assemblies against its domain.
        # Plain input assemblies index the full unit cell and are remapped below.
        assemblies=asu.assemblies if had_existing_asu else None,
    )
    result_structure = UnitcellStructureView(standard_asu)
    original_count = len(original.sites)
    if original_count == 0:
        raise ValueError("conventional_cell() cannot determine a site-count multiplier for an empty structure")
    multiplier = fractions.Fraction(len(result_structure.sites), original_count)

    # All tabulated transforms have determinant 1 or 3. Keep this invariant explicit while
    # allowing a caller-supplied untabulated transform to describe a larger own cell.
    if asu.setting() is not None:
        assert multiplier >= 1

    if had_existing_asu:
        result_assemblies = result_structure.assemblies
    else:
        mapping = _exact_site_bijection(original, result_structure, transform)
        result_assemblies = _remap_assemblies(source_assemblies, mapping)
    scaled_composition = _scaled_composition(source_composition, multiplier) if source_composition is not None else None
    # The returned ASU is part of the public transform result, so retain every annotation
    # it can express directly. Full-cell assemblies from a plain input remain on the
    # expanded structure because their indices do not name domain sites.
    initialize_semantics(
        standard_asu,
        nsites=len(standard_asu.wyckoff_sites),
        molecular=source_molecular,
        assemblies=asu.assemblies if had_existing_asu else None,
        symmetry=None,
        chemical_composition=scaled_composition,
        chemical_formula_descriptive=source_descriptive,
        chemical_formula_hill=source_hill,
        optimization_type=source_optimization,
    )
    initialize_semantics(
        result_structure,
        nsites=len(result_structure.sites),
        molecular=source_molecular,
        assemblies=result_assemblies,
        symmetry=None,
        chemical_composition=scaled_composition,
        chemical_formula_descriptive=source_descriptive,
        chemical_formula_hill=source_hill,
        optimization_type=source_optimization,
    )

    return ConventionalCellResult(
        result_structure,
        standard_asu,
        asu.spacegroup,
        transform,
        multiplier,
    )
