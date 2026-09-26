"""Anonymous protostructure-first canonicalization of measured crystal structures.

This convention chooses the anonymous Wyckoff occupation pattern, then its species assignment,
and only then the exact geometry. Its finite affine search uses the same tabulated normalizer scope
as the legacy canonicalizer and does not enumerate an infinite affine normalizer. Public measured
structure entry points optionally perform the established upward pseudosymmetry search first.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from httk.core import FracVector, SurdVector

from httk.atomistic import data
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import ASUStructure, WyckoffSite
from httk.atomistic.models.structure.comparison import same_crystal
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.symmetry._anonymous import (
    _anonymous_p1_frame,
    _proxy_niggli_transform,
    _rational_basis_change,
)
from httk.atomistic.symmetry._periodicity_guard import require_full_periodicity
from httk.atomistic.symmetry._standardization_common import (
    _matrix_column_sum_factor,
    _matrix_row_sum_factor,
    _scaled_composition,
    _scaled_precision,
)
from httk.atomistic.symmetry._wyckoff_actions import compile_wyckoff_action
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.canonical import (
    _recognition_sweep,
    _reversed_p1_frame,
)
from httk.atomistic.symmetry.lift import (
    _basis_key,
    _canonical_orientation,
    _canonical_sites,
    _demote_sites,
    _discrete_normalizer_translations,
    _enantiomorph,
    _isomorphic_reduced_entry,
    _metric_automorphism_operations,
    _niggli_reduced_entry,
    _p1_metric_automorphism_operations,
    _primitive_reduced_entry,
    _resetting_preserves_group,
    _site_key,
    _translation_normal_form,
)
from httk.atomistic.symmetry.limits import _checkpoint
from httk.atomistic.symmetry.recognition import structure_tolerance
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.subgroups import _standard_input

__all__ = ["canonical_asu_protostructure", "canonical_asu_protostructure_assignments"]


@dataclass(frozen=True)
class _Candidate:
    operation: AffineOperation
    basis_change: FracVector
    metric_key: tuple[Any, ...]
    identity: bool


def _restore_source_metadata(result: ASUStructure, source: Any) -> ASUStructure:
    metadata = {
        "chemical_composition": source.chemical_composition,
        "chemical_formula_descriptive": source.chemical_formula_descriptive,
        "chemical_formula_hill": source.chemical_formula_hill,
        "optimization_type": source.optimization_type,
        "immutable_id": source.immutable_id,
        "last_modified": source.last_modified,
    }
    if not any(value is not None for value in metadata.values()):
        return result
    if metadata["chemical_composition"] is not None:
        source_count = len(UnitcellStructureView(source).sites)
        if source_count == 0:
            raise ValueError("cannot scale source composition for an empty structure")
        multiplier = Fraction(len(UnitcellStructureView(result).sites), source_count)
        metadata["chemical_composition"] = _scaled_composition(metadata["chemical_composition"], multiplier)
    return ASUStructure(
        result.cell,
        result.spacegroup,
        result.wyckoff_sites,
        result.species,
        transform=result.transform,
        coordinate_precision=result.coordinate_precision,
        charge=result.charge,
        **metadata,
    )


def _restore_assignment(result: ASUStructure, assignment: tuple[Species, ...], source: Any) -> ASUStructure:
    by_anonymous = {anonymous.name: original for anonymous, original in zip(result.species, assignment, strict=True)}
    restored = ASUStructure(
        result.cell,
        result.spacegroup,
        tuple(
            WyckoffSite(site.wyckoff, site.free_params, by_anonymous[site.species].name)
            for site in result.wyckoff_sites
        ),
        source.species,
        transform=result.transform,
        coordinate_precision=result.coordinate_precision,
        charge=result.charge,
    )
    return _restore_source_metadata(restored, source)


def _restore_unchanged_precision(result: ASUStructure, source: StructureLike) -> ASUStructure:
    """Restore source bounds when canonicalization leaves the exact unit-cell data unchanged."""
    source_view = UnitcellStructureView(source)
    if not same_crystal(result, source_view):
        return result
    return ASUStructure(
        Cell(result.cell, precision=source_view.cell.precision),
        result.spacegroup,
        result.wyckoff_sites,
        result.species,
        transform=result.transform,
        coordinate_precision=source_view.sites.precision,
        charge=result.charge,
        molecular=result.molecular,
        assemblies=result.assemblies,
        chemical_composition=result.chemical_composition,
        chemical_formula_descriptive=result.chemical_formula_descriptive,
        chemical_formula_hill=result.chemical_formula_hill,
        optimization_type=result.optimization_type,
        immutable_id=result.immutable_id,
        last_modified=result.last_modified,
    )


def _anonymize_asu(
    structure: ASUStructure,
    anonymous_species: tuple[Species, ...],
    assignment: tuple[Species, ...],
) -> ASUStructure:
    anonymous_by_original = {
        original.name: anonymous.name for anonymous, original in zip(anonymous_species, assignment, strict=True)
    }
    return ASUStructure(
        structure.cell,
        structure.spacegroup,
        tuple(
            WyckoffSite(site.wyckoff, site.free_params, anonymous_by_original[site.species])
            for site in structure.wyckoff_sites
        ),
        anonymous_species,
        transform=structure.transform,
        coordinate_precision=structure.coordinate_precision,
        charge=structure.charge,
    )


def _anonymous_geometry_key(structure: ASUStructure) -> tuple[Any, ...]:
    classes: dict[str, list[tuple[str, tuple[Fraction, ...]]]] = {}
    populations: dict[str, int] = {}
    for site in structure.wyckoff_sites:
        _checkpoint()
        classes.setdefault(site.species, []).append((site.wyckoff, tuple(site.free_params.to_fractions())))
        populations[site.species] = (
            populations.get(site.species, 0) + structure.spacegroup.wyckoff_position(site.wyckoff).multiplicity
        )
    metric = structure.cell.metric()
    return (
        structure.spacegroup.it_number,
        structure.spacegroup.hall_entry,
        tuple(metric._element((row, column)) for row in range(3) for column in range(3)),
        tuple(sorted((populations[name], tuple(sorted(values))) for name, values in classes.items())),
        0 if structure.cell.basis.det().sign() > 0 else 1,
        _basis_key(structure.cell.basis),
    )


def _chemical_key(structure: ASUStructure) -> tuple[Any, ...]:
    return _site_key(structure), tuple(repr(species) for species in structure.species)


def _deduplicate_results(values: list[ASUStructure]) -> tuple[ASUStructure, ...]:
    unique: dict[tuple[Any, ...], ASUStructure] = {}
    for value in values:
        _checkpoint()
        key = (
            value.spacegroup,
            value.cell.basis,
            _site_key(value),
            value.species,
            value.coordinate_precision,
            value.charge,
        )
        unique.setdefault(key, value)
    return tuple(sorted(unique.values(), key=_chemical_key))


def _apply_action(structure: ASUStructure, operation: AffineOperation) -> ASUStructure:
    sites = []
    for site in structure.wyckoff_sites:
        _checkpoint()
        action = compile_wyckoff_action(structure.spacegroup, operation, site.wyckoff)
        sites.append(WyckoffSite(action.target_letter, action.apply(site.free_params), site.species))
    basis_change = operation.matrix.T().inv()
    basis = SurdVector(basis_change) * structure.cell.basis
    return ASUStructure(
        Cell(
            basis,
            precision=_scaled_precision(
                structure.cell.precision,
                _matrix_row_sum_factor(basis_change),
            ),
            periodicity=structure.cell.periodicity,
        ),
        structure.spacegroup,
        sites,
        structure.species,
        transform=SettingTransform.identity(),
        coordinate_precision=_scaled_precision(
            structure.coordinate_precision,
            _matrix_column_sum_factor(basis_change.inv()),
        ),
        charge=structure.charge,
    )


def _with_basis(structure: ASUStructure, basis: SurdVector) -> ASUStructure:
    basis_change = _rational_basis_change(basis, structure.cell.basis)
    return ASUStructure(
        Cell(
            basis,
            precision=_scaled_precision(
                structure.cell.precision,
                _matrix_row_sum_factor(basis_change),
            ),
            periodicity=structure.cell.periodicity,
        ),
        structure.spacegroup,
        structure.wyckoff_sites,
        structure.species,
        transform=SettingTransform.identity(),
        coordinate_precision=structure.coordinate_precision,
        charge=structure.charge,
    )


def _discrete_key(
    structure: ASUStructure, operation: AffineOperation
) -> tuple[tuple[tuple[str, ...], ...], tuple[Any, ...]]:
    letters: dict[str, list[str]] = {}
    for site in structure.wyckoff_sites:
        _checkpoint()
        target = compile_wyckoff_action(structure.spacegroup, operation, site.wyckoff).target_letter
        letters.setdefault(site.species, []).append(target)
    classes = {species: tuple(sorted(values)) for species, values in letters.items()}
    anonymous = tuple(sorted(classes.values()))
    assignment = tuple(sorted((values, species) for species, values in classes.items()))
    return anonymous, assignment


def _representatives(structure: ASUStructure) -> tuple[AffineOperation, ...]:
    operations = [AffineOperation.identity()]
    if structure.spacegroup.it_number == 1:
        operations.extend(_p1_metric_automorphism_operations(structure))
    else:
        record = data.affine_normalizer_coset_record(structure.spacegroup.hall_entry)
        operations.extend(AffineOperation.from_record(value) for value in record.get("affine_normalizer_cosets", ()))
    return tuple(dict.fromkeys(operations))


def _point_operations(structure: ASUStructure) -> tuple[AffineOperation, ...]:
    by_matrix: dict[FracVector, AffineOperation] = {}
    for operation in structure.spacegroup.symmetry_operations:
        _checkpoint()
        if operation.matrix not in by_matrix or operation.is_identity():
            by_matrix[operation.matrix] = operation
    return tuple(by_matrix.values())


def _discrete_candidates(structure: ASUStructure) -> tuple[_Candidate, ...]:
    identity_matrix = FracVector.eye((3, 3))
    inversion_matrix = FracVector(((-1, 0, 0), (0, -1, 0), (0, 0, -1)))
    inversion = AffineOperation(inversion_matrix, (0, 0, 0))
    can_invert = _resetting_preserves_group(structure.spacegroup, inversion_matrix)
    candidates: list[tuple[tuple[Any, ...], _Candidate]] = []
    best_key: tuple[Any, ...] | None = None
    point_operations = tuple((operation, operation.matrix.T().inv()) for operation in _point_operations(structure))
    discrete_keys: dict[AffineOperation, tuple[tuple[tuple[str, ...], ...], tuple[Any, ...]]] = {}
    metric_keys: dict[FracVector, tuple[Any, ...]] = {}
    source_metric = structure.cell.metric()
    source_handedness = structure.cell.basis.det().sign()
    for representative in _representatives(structure):
        _checkpoint()
        for translation in _discrete_normalizer_translations(structure.spacegroup):
            _checkpoint()
            operation = (
                representative
                if not any(translation)
                else AffineOperation(identity_matrix, translation) * representative
            )
            operation_basis_change = operation.matrix.T().inv()
            for point_operation, point_basis_change in point_operations:
                _checkpoint()
                basis_change = point_basis_change * operation_basis_change
                site_operation = operation
                if basis_change.det().sign() * source_handedness < 0 and can_invert:
                    basis_change = -basis_change
                    site_operation = inversion * operation
                key = discrete_keys.get(site_operation)
                if key is None:
                    key = _discrete_key(structure, site_operation)
                    discrete_keys[site_operation] = key
                if best_key is not None and key > best_key:
                    continue
                metric_key = metric_keys.get(basis_change)
                if metric_key is None:
                    basis_map = SurdVector(basis_change)
                    metric = basis_map * source_metric * basis_map.T()
                    metric_key = tuple(metric._element((row, column)) for row in range(3) for column in range(3))
                    metric_keys[basis_change] = metric_key
                candidate = _Candidate(
                    site_operation,
                    basis_change,
                    metric_key,
                    representative.is_identity() and point_operation.is_identity() and not any(translation),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    candidates = [(key, candidate)]
                elif key == best_key:
                    candidates.append((key, candidate))
    if not candidates:
        raise ValueError(f"no exact affine-normalizer action applies in {structure.spacegroup.setting}")
    return tuple(candidate for _key, candidate in candidates)


def _geometric_choice(
    candidate: _Candidate,
    reduced: ASUStructure,
    source_basis: SurdVector,
) -> tuple[tuple[Any, ...], ASUStructure]:
    represented = _with_basis(reduced, SurdVector(candidate.basis_change) * source_basis)
    metric = represented.cell.metric()
    metric_key = tuple(metric._element((row, column)) for row in range(3) for column in range(3))
    key = (
        metric_key,
        _site_key(represented),
        0 if represented.cell.basis.det().sign() > 0 else 1,
        _basis_key(represented.cell.basis),
        0 if candidate.identity else 1,
    )
    return key, represented


def _validate_exact_input(structure: ASUStructure) -> None:
    require_full_periodicity(structure.cell, "canonical_asu_protostructure")
    if any(site.moment is not None for site in structure.wyckoff_sites):
        raise ValueError("canonical_asu_protostructure does not support structures with site moments")
    if structure.assemblies is not None:
        raise ValueError("canonical_asu_protostructure does not support structures with assemblies")
    if structure.molecular:
        raise ValueError("canonical_asu_protostructure does not support molecular structures")


def _triclinic_proxy_entries(structure: ASUStructure) -> tuple[ASUStructure, ...]:
    """Return every tied proper proxy-reduced entry for an exact P-1 structure."""
    transform = _proxy_niggli_transform(structure.cell.metric())
    if transform.det() < 0:
        transform = -transform
    reduced = _apply_action(
        structure,
        AffineOperation(transform.T().inv(), (0, 0, 0)),
    )
    proxy = reduced.cell.metric().coefficient(1)
    gram = tuple(tuple(value for value in row) for row in proxy.to_fractions())
    candidates = [
        _apply_action(reduced, operation)
        for operation in _metric_automorphism_operations(gram)
        if operation.determinant() == 1
    ]
    if not candidates:
        raise ValueError("P-1 proxy reduction found no proper metric automorphism")
    keyed = [
        (
            candidate,
            tuple(candidate.cell.metric()._element((row, column)) for row in range(3) for column in range(3)),
        )
        for candidate in candidates
    ]
    least = min(key for _candidate, key in keyed)
    winners: list[ASUStructure] = []
    winner_keys: set[tuple[Any, ...]] = set()
    for candidate, key in keyed:
        _checkpoint()
        if key != least:
            continue
        candidate_key = (_site_key(candidate), _basis_key(candidate.cell.basis))
        if candidate_key not in winner_keys:
            winner_keys.add(candidate_key)
            winners.append(candidate)
    return tuple(winners)


def _canonical_protostructure_geometry_entry(
    current: ASUStructure,
    *,
    preserve_chirality: bool,
) -> ASUStructure:
    """Run the finite exact terminal from one already normalized lattice entry."""
    if current.spacegroup.it_number in (1, 2):
        current = _niggli_reduced_entry(current)
    current = _demote_sites(current)
    if not preserve_chirality:
        current = _enantiomorph(current) or current
        current = _demote_sites(current)

    candidates = _discrete_candidates(current)
    least_metric = min(candidate.metric_key for candidate in candidates)
    candidates = tuple(candidate for candidate in candidates if candidate.metric_key == least_metric)
    reduced_by_action: dict[AffineOperation, ASUStructure] = {}
    for candidate in candidates:
        _checkpoint()
        if candidate.operation not in reduced_by_action:
            reduced_by_action[candidate.operation] = _canonical_sites(
                _translation_normal_form(_apply_action(current, candidate.operation))
            )
    choices = (
        _geometric_choice(candidate, reduced_by_action[candidate.operation], current.cell.basis)
        for candidate in candidates
    )
    _key, best = min(choices, key=lambda value: value[0])
    return _canonical_orientation(best)


def _terminal_result_key(structure: ASUStructure) -> tuple[Any, ...]:
    """Order tied lattice entries through the terminal's discrete and geometric tiers."""
    metric = structure.cell.metric()
    return (
        _discrete_key(structure, AffineOperation.identity()),
        tuple(metric._element((row, column)) for row in range(3) for column in range(3)),
        _site_key(structure),
        0 if structure.cell.basis.det().sign() > 0 else 1,
        _basis_key(structure.cell.basis),
    )


def _canonical_protostructure_geometry(
    structure: ASUStructure,
    *,
    preserve_chirality: bool = True,
) -> ASUStructure:
    """Canonicalize one anonymously labelled ASU by discrete occupation before geometry."""
    if not isinstance(structure, ASUStructure):
        raise TypeError(f"expected ASUStructure, got {type(structure).__name__}")
    _validate_exact_input(structure)
    current = _standard_input(structure)
    if current.spacegroup.it_number == 1:
        current = _primitive_reduced_entry(current)
    else:
        current = _isomorphic_reduced_entry(current)
    entries = _triclinic_proxy_entries(current) if current.spacegroup.it_number == 2 else (current,)
    return min(
        (
            _canonical_protostructure_geometry_entry(
                entry,
                preserve_chirality=preserve_chirality,
            )
            for entry in entries
        ),
        key=_terminal_result_key,
    )


def _canonical_protostructure_assignments_asu(
    structure: ASUStructure,
    *,
    preserve_chirality: bool = True,
) -> tuple[ASUStructure, ...]:
    """Return all tied chemical assignments of one recognized exact ASU."""
    if not isinstance(structure, ASUStructure):
        raise TypeError(f"expected ASUStructure, got {type(structure).__name__}")
    _validate_exact_input(structure)
    frame = _anonymous_p1_frame(UnitcellStructureView(structure))
    anonymous_species = frame.structure.species
    candidates = []
    for assignment in frame.assignments:
        # For P1 the anonymous frame is itself the complete exact ASU.  Feeding that frame into
        # the terminal preserves its proxy-Niggli basis choice when the full Gram matrix contains
        # irrational coefficients (the legacy rational-only Niggli helper intentionally skips
        # such a metric).  Higher groups must retain their declared Wyckoff representation here;
        # their P1 expansion is used only to choose anonymous class identities.
        _checkpoint()
        anonymous = (
            frame.structure
            if structure.spacegroup.it_number == 1
            else _anonymize_asu(structure, anonymous_species, assignment)
        )
        canonical = _canonical_protostructure_geometry(anonymous, preserve_chirality=preserve_chirality)
        candidates.append((canonical, assignment, _anonymous_geometry_key(canonical)))
    best_geometry = min(key for _canonical, _assignment, key in candidates)
    restored = []
    for canonical, assignment, key in candidates:
        _checkpoint()
        if key != best_geometry:
            continue
        result = _restore_assignment(canonical, assignment, structure)
        restored.append(_restore_unchanged_precision(result, structure))
    return _deduplicate_results(restored)


def _canonical_protostructure_asu(
    structure: ASUStructure,
    *,
    preserve_chirality: bool = True,
) -> ASUStructure:
    """Canonicalize one recognized ASU, selecting chemistry only after geometry."""
    return min(
        _canonical_protostructure_assignments_asu(structure, preserve_chirality=preserve_chirality),
        key=_chemical_key,
    )


def canonical_asu_protostructure_assignments(
    structure: StructureLike,
    *,
    tolerance: float | None = None,
    factors: tuple[Fraction | float | int, ...] = (Fraction(1, 5), 1, 5),
    lift: bool = False,
    preserve_chirality: bool = True,
) -> tuple[ASUStructure, ...]:
    """Recognize and return every tied assignment of the anonymous canonical geometry.

    Recognition uses the established tolerance sweep and deterministic P1 frames. The accepted ASU
    is then handled exactly. When ``lift`` is true, the anonymous winner is first searched for
    upward pseudosymmetry.

    :param structure: The measured structure to recognize.
    :param tolerance: Base Cartesian recognition tolerance, or ``None`` to derive it.
    :param factors: Multipliers for the recognition tolerance sweep.
    :param lift: Whether to search upward for pseudosymmetry after recognition.
    :param preserve_chirality: Whether to keep the recognized enantiomorphic group.
    :return: The tied protostructure-first canonical asymmetric units.
    :raises ValueError: If the structure is unsupported or no tolerance member fits.
    """
    source_view = UnitcellStructureView(structure)
    if source_view.site_moments is not None:
        raise ValueError("canonical_asu_protostructure does not support structures with site moments")
    if source_view.assemblies is not None:
        raise ValueError("canonical_asu_protostructure does not support structures with assemblies")
    if source_view.molecular:
        raise ValueError("canonical_asu_protostructure does not support molecular structures")
    require_full_periodicity(source_view.cell, "canonical_asu_protostructure")
    base = structure_tolerance(source_view) if tolerance is None else float(tolerance)
    outer_frame = _anonymous_p1_frame(source_view)
    normalized_p1 = outer_frame.structure
    winner, failures = _recognition_sweep(UnitcellStructureView(normalized_p1), base, factors)
    if winner is None or winner.spacegroup.it_number == 1:
        alternate, alternate_failures = _recognition_sweep(
            UnitcellStructureView(_reversed_p1_frame(normalized_p1)), base, factors
        )
        failures.extend(f"reversed canonical frame {failure}" for failure in alternate_failures)
        if alternate is not None and (
            winner is None or len(alternate.spacegroup.symmetry_operations) > len(winner.spacegroup.symmetry_operations)
        ):
            winner = alternate
    if winner is None:
        raise ValueError(f"no symmetry fit the structure within tolerance {base:g}; tried [{', '.join(failures)}]")
    if lift:
        from httk.atomistic.symmetry.lift import canonicalize_legacy

        winner = canonicalize_legacy(winner, tolerance=base, preserve_chirality=preserve_chirality).asu
        preserve_chirality = True
    inner = _canonical_protostructure_assignments_asu(winner, preserve_chirality=preserve_chirality)
    outer_by_name = {anonymous.name: index for index, anonymous in enumerate(outer_frame.structure.species)}
    restored = []
    for candidate in inner:
        _checkpoint()
        for outer_assignment in outer_frame.assignments:
            _checkpoint()
            assignment = tuple(outer_assignment[outer_by_name[species.name]] for species in candidate.species)
            result = _restore_assignment(candidate, assignment, source_view)
            restored.append(_restore_unchanged_precision(result, source_view))
    return _deduplicate_results(restored)


def canonical_asu_protostructure(
    structure: StructureLike,
    *,
    tolerance: float | None = None,
    factors: tuple[Fraction | float | int, ...] = (Fraction(1, 5), 1, 5),
    lift: bool = False,
    preserve_chirality: bool = True,
) -> ASUStructure:
    """Return one chemically ordered member of the anonymous canonical assignment family.

    :param structure: The measured structure to recognize.
    :param tolerance: Base Cartesian recognition tolerance, or ``None`` to derive it.
    :param factors: Multipliers for the recognition tolerance sweep.
    :param lift: Whether to search upward for pseudosymmetry after recognition.
    :param preserve_chirality: Whether to keep the recognized enantiomorphic group.
    :return: One protostructure-first canonical asymmetric unit.
    :raises ValueError: If the structure is unsupported or no tolerance member fits.
    """
    return min(
        canonical_asu_protostructure_assignments(
            structure,
            tolerance=tolerance,
            factors=factors,
            lift=lift,
            preserve_chirality=preserve_chirality,
        ),
        key=_chemical_key,
    )
