"""Private helpers shared by the standardization-family operations."""

import fractions
from typing import Any

from httk.core import FracVector, unwrap

from httk.atomistic.composition import Assembly, ChemicalComposition
from httk.atomistic.models.structure.asu import ASUStructure
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.symmetry.setting_transform import SettingTransform


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
        effective = getattr(candidate, "_effective_asu", None)
        if callable(effective):
            candidate = effective()
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
