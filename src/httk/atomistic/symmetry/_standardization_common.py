"""Private helpers shared by the standardization-family operations."""

import fractions
import math
from collections.abc import Sequence
from typing import Any

from httk.core import FracVector, SurdVector, unwrap

from httk.atomistic.composition import Assembly, ChemicalComposition
from httk.atomistic.models.moments.backend import SiteMomentsBackend
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.models.structure.asu import ASUStructure
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.symmetry.setting_transform import SettingTransform

#: Refusal shared wherever crystal-axis moments meet a cell change. A crystal-axis moment is
#: stated as fractions of the old cell's axes, so it does not survive the recombination into a
#: new cell unchanged the way a lab-frame Cartesian vector or a frame-free collinear scalar does.
CRYSTAL_AXIS_MOMENT_REFUSAL = (
    "crystal-axis site moments cannot be re-expressed through a cell change; convert them to Cartesian first"
)

#: Refusal shared wherever moments that must fold onto one site disagree: the magnetic cell is
#: genuinely larger than the nuclear cell, so no smaller cell can represent the order.
MAGNETIC_SUPERCELL_REFUSAL = "magnetic order incompatible with the primitive cell; keep the original setting"

#: Absolute and relative closeness for the float moments that a cell change folds onto one site.
#: Site moments arriving from a DFT run are floats; images sharing a site must agree within this.
_MOMENT_CLOSENESS = 1e-3


def _moments_close(moments: SiteMomentsBackend, first: int, second: int) -> bool:
    """Whether two sites carry the same moment within the float closeness tolerance.

    :param moments: The per-site moments to compare within.
    :param first: The first site index.
    :param second: The second site index.
    :return: Whether the two sites' moments agree within :data:`_MOMENT_CLOSENESS`.
    :raises ValueError: If the moments are crystal-axis moments, which a cell change alters.
    """
    if isinstance(moments, CartesianSiteMoments):
        grid = moments.cartesian_moments
        return all(
            math.isclose(
                float(grid._element((first, column)).to_float()),
                float(grid._element((second, column)).to_float()),
                rel_tol=_MOMENT_CLOSENESS,
                abs_tol=_MOMENT_CLOSENESS,
            )
            for column in range(3)
        )
    if isinstance(moments, CollinearSiteMoments):
        values = moments.collinear_moments.to_fractions()
        return math.isclose(
            float(values[first]), float(values[second]), rel_tol=_MOMENT_CLOSENESS, abs_tol=_MOMENT_CLOSENESS
        )
    raise ValueError(CRYSTAL_AXIS_MOMENT_REFUSAL)


def _reorder_site_moments(moments: SiteMomentsBackend, order: Sequence[int]) -> SiteMomentsBackend:
    """Pick site-moment rows in a new order, carrying them as per-site data.

    Cartesian and collinear moments are moved verbatim: a setting or centring change is a
    pure basis recombination plus an origin shift, which leaves a lab-frame Cartesian vector
    and a frame-free collinear scalar unchanged. Crystal-axis moments are refused.

    :param moments: The source per-site moments.
    :param order: The source site index taken for each output site, in output order.
    :return: The reordered moments in the same backend kind.
    :raises ValueError: If the moments are crystal-axis moments, which a cell change alters.
    """
    if isinstance(moments, CartesianSiteMoments):
        grid = moments.cartesian_moments
        rows = [[grid._element((source, column)) for column in range(3)] for source in order]
        return CartesianSiteMoments(SurdVector._from_scalar_grid(rows, (len(rows), 3)), precision=moments.precision)
    if isinstance(moments, CollinearSiteMoments):
        values = moments.collinear_moments.to_fractions()
        return CollinearSiteMoments([values[source] for source in order], precision=moments.precision)
    raise ValueError(CRYSTAL_AXIS_MOMENT_REFUSAL)


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
