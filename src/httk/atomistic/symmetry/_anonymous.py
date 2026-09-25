"""Exact anonymous P1 framing for protostructure canonicalization."""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from httk.core import FracVector, SurdVector

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import ASUStructure, WyckoffSite
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.reduction import _MAX_STEPS, _niggli_step, _parameters
from httk.atomistic.symmetry._standardization_common import (
    _matrix_column_sum_factor,
    _matrix_row_sum_factor,
    _scaled_precision,
)
from httk.atomistic.symmetry.lift import (
    _basis_key,
    _canonical_orientation,
    _metric_automorphism_operations,
    _primitive_reduced_entry,
)


@dataclass(frozen=True)
class _AnonymousFrame:
    """Hold one anonymous geometry and all tied maps back to source classes."""

    structure: ASUStructure
    assignments: tuple[tuple[Species, ...], ...]


def _class_names(count: int) -> tuple[str, ...]:
    width = max(4, len(str(max(0, count - 1))))
    return tuple(f"__httk_anonymous_{index:0{width}d}" for index in range(count))


def _dummy_species(names: tuple[str, ...]) -> tuple[Species, ...]:
    return tuple(Species(name, ("X",), (1,)) for name in names)


def _rational_basis_change(new: SurdVector, old: SurdVector) -> FracVector:
    change = new * old.inv()
    rows = []
    for row in range(3):
        values = []
        for column in range(3):
            value = change._element((row, column))
            if not value.is_rational:
                raise ValueError("anonymous P1 framing produced a non-rational lattice re-expression")
            values.append(value._rational_fraction())
        rows.append(values)
    return FracVector(rows)


def _rebase_p1(structure: ASUStructure, basis_change: FracVector) -> ASUStructure:
    inverse = basis_change.inv()
    return ASUStructure(
        Cell(
            SurdVector(basis_change) * structure.cell.basis,
            precision=structure.cell.precision,
            periodicity=structure.cell.periodicity,
        ),
        1,
        tuple(
            WyckoffSite("a", (site.free_params * inverse).normalize(), site.species) for site in structure.wyckoff_sites
        ),
        structure.species,
        coordinate_precision=structure.coordinate_precision,
        charge=structure.charge,
    )


def _proxy_niggli_transform(metric: SurdVector) -> FracVector:
    gram = metric.coefficient(1)
    transform = FracVector.eye((3, 3))
    for _ in range(_MAX_STEPS):
        step = _niggli_step(_parameters(gram))
        if step is None:
            return transform
        transform = step * transform
        gram = step * gram * step.T()
    raise RuntimeError("anonymous proxy Niggli reduction did not converge")


def _frame_key(structure: ASUStructure) -> tuple[Any, ...]:
    metric = structure.cell.metric()
    return (
        tuple(metric._element((row, column)) for row in range(3) for column in range(3)),
        tuple((site.species, tuple(site.free_params.to_fractions())) for site in structure.wyckoff_sites),
        _basis_key(structure.cell.basis),
    )


def _anonymous_p1_frame(structure: StructureLike) -> _AnonymousFrame:
    """Return one exact anonymous P1 frame and every tied source-class assignment."""
    view = UnitcellStructureView(structure)
    coordinates = tuple(view.sites.reduced_coords.to_fractions())
    species_at_sites = tuple(view.species_at_sites)
    if not coordinates:
        raise ValueError("anonymous canonicalization requires at least one occupied site")

    source_by_name = {species.name: species for species in view.species}
    initial_by_source: dict[str, str] = {}
    initial_names: list[str] = []
    for source_name in species_at_sites:
        if source_name not in initial_by_source:
            initial_name = _class_names(len(initial_by_source) + 1)[-1]
            initial_by_source[source_name] = initial_name
            initial_names.append(initial_name)
    initial_species = _dummy_species(tuple(initial_names))
    source_for_initial = {
        initial_by_source[source_name]: source_by_name[source_name] for source_name in initial_by_source
    }
    p1 = ASUStructure(
        view.cell,
        1,
        tuple(
            WyckoffSite("a", FracVector(point).normalize(), initial_by_source[source_name])
            for point, source_name in zip(coordinates, species_at_sites, strict=True)
        ),
        initial_species,
        coordinate_precision=view.sites.precision,
        charge=view.charge,
    )
    reduced = _primitive_reduced_entry(p1)
    if reduced.cell.basis.det().sign() < 0:
        reduced = _rebase_p1(reduced, -FracVector.eye((3, 3)))
    reduced = _rebase_p1(reduced, _proxy_niggli_transform(reduced.cell.metric()))

    proxy = reduced.cell.metric().coefficient(1)
    gram = tuple(tuple(value for value in row) for row in proxy.to_fractions())
    operations = _metric_automorphism_operations(gram)
    populations = Counter(site.species for site in reduced.wyckoff_sites)
    least_population = min(populations.values())
    final_names = _class_names(len(populations))
    final_species = _dummy_species(final_names)
    best_geometry: tuple[Any, ...] | None = None
    candidates: list[tuple[ASUStructure, tuple[Species, ...]]] = []

    for operation in operations:
        basis_change = operation.matrix.T().inv()
        basis = SurdVector(basis_change) * reduced.cell.basis
        if basis.det().sign() < 0:
            continue
        points = tuple(
            (site.species, tuple(operation.apply_wrapped(site.free_params).to_fractions()))
            for site in reduced.wyckoff_sites
        )
        for anchor_name, anchor in points:
            if populations[anchor_name] != least_population:
                continue
            grouped: dict[str, list[tuple[Fraction, ...]]] = {}
            for class_name, point in points:
                grouped.setdefault(class_name, []).append(
                    tuple((value - origin) % 1 for value, origin in zip(point, anchor, strict=True))
                )
            blocks = sorted((len(values), tuple(sorted(values)), class_name) for class_name, values in grouped.items())
            anonymous_blocks = tuple((count, values) for count, values, _class_name in blocks)
            if len(set(anonymous_blocks)) != len(anonymous_blocks):
                raise ValueError("anonymous canonicalization cannot distinguish coincident species classes")
            metric = basis * basis.T()
            geometry = (
                tuple(metric._element((row, column)) for row in range(3) for column in range(3)),
                anonymous_blocks,
            )
            if best_geometry is not None and geometry > best_geometry:
                continue

            total_change = _rational_basis_change(basis, view.cell.basis)
            cell_precision = _scaled_precision(view.cell.precision, _matrix_row_sum_factor(total_change))
            coordinate_precision = _scaled_precision(
                view.sites.precision, _matrix_column_sum_factor(total_change.inv())
            )
            anonymous = ASUStructure(
                Cell(basis, precision=cell_precision, periodicity=view.cell.periodicity),
                1,
                tuple(
                    WyckoffSite("a", FracVector(point), final_names[index])
                    for index, (_count, values, _class_name) in enumerate(blocks)
                    for point in values
                ),
                final_species,
                coordinate_precision=coordinate_precision,
                charge=reduced.charge,
            )
            assignment = tuple(source_for_initial[class_name] for _count, _values, class_name in blocks)
            anonymous = _canonical_orientation(anonymous)
            if best_geometry is None or geometry < best_geometry:
                best_geometry = geometry
                candidates = [(anonymous, assignment)]
            else:
                candidates.append((anonymous, assignment))

    if not candidates:
        raise ValueError("anonymous P1 framing found no right-handed reduced-Gram representative")
    best_frame_key = min(_frame_key(candidate) for candidate, _assignment in candidates)
    winners = [
        (candidate, assignment) for candidate, assignment in candidates if _frame_key(candidate) == best_frame_key
    ]
    result = winners[0][0]
    assignments = tuple(sorted({assignment for _candidate, assignment in winners}, key=repr))
    return _AnonymousFrame(result, assignments)
