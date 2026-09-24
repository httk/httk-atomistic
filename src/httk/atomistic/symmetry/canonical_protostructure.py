"""Protostructure-first canonicalization of a recognized crystal structure.

This opt-in convention chooses the anonymous Wyckoff occupation pattern, then its species
assignment, and only then the exact geometry. Its finite affine search is the same tabulated
normalizer scope used by the established canonicalizer; it does not enumerate an infinite affine
normalizer and performs no upward pseudosymmetry search.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from httk.core import FracVector, SurdVector

from httk.atomistic import data
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.structure.asu import ASUStructure, WyckoffSite
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.symmetry._periodicity_guard import require_full_periodicity
from httk.atomistic.symmetry._standardization_common import _scaled_composition
from httk.atomistic.symmetry._wyckoff_actions import compile_wyckoff_action
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.canonical import _preconditioned_p1, _recognition_sweep, _reversed_p1_frame
from httk.atomistic.symmetry.lift import (
    _basis_key,
    _canonical_orientation,
    _canonical_sites,
    _demote_sites,
    _discrete_normalizer_translations,
    _enantiomorph,
    _isomorphic_reduced_entry,
    _niggli_reduced_entry,
    _p1_metric_automorphism_operations,
    _primitive_reduced_entry,
    _resetting_preserves_group,
    _site_key,
    _translation_normal_form,
)
from httk.atomistic.symmetry.recognition import structure_tolerance
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.subgroups import _standard_input

__all__ = ["canonical_asu_protostructure"]


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


def _apply_action(structure: ASUStructure, operation: AffineOperation) -> ASUStructure:
    sites = []
    for site in structure.wyckoff_sites:
        action = compile_wyckoff_action(structure.spacegroup, operation, site.wyckoff)
        sites.append(WyckoffSite(action.target_letter, action.apply(site.free_params), site.species))
    basis = SurdVector(operation.matrix.T().inv()) * structure.cell.basis
    return ASUStructure(
        Cell(basis, precision=structure.cell.precision, periodicity=structure.cell.periodicity),
        structure.spacegroup,
        sites,
        structure.species,
        transform=SettingTransform.identity(),
        coordinate_precision=structure.coordinate_precision,
        charge=structure.charge,
    )


def _with_basis(structure: ASUStructure, basis: SurdVector) -> ASUStructure:
    return ASUStructure(
        Cell(basis, precision=structure.cell.precision, periodicity=structure.cell.periodicity),
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
        for translation in _discrete_normalizer_translations(structure.spacegroup):
            operation = (
                representative
                if not any(translation)
                else AffineOperation(identity_matrix, translation) * representative
            )
            operation_basis_change = operation.matrix.T().inv()
            for point_operation, point_basis_change in point_operations:
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


def _canonical_protostructure_asu(
    structure: ASUStructure,
    *,
    preserve_chirality: bool = True,
) -> ASUStructure:
    """Canonicalize one recognized ASU by discrete occupation before exact geometry."""
    if not isinstance(structure, ASUStructure):
        raise TypeError(f"expected ASUStructure, got {type(structure).__name__}")
    _validate_exact_input(structure)
    current = _standard_input(structure)
    if current.spacegroup.it_number == 1:
        current = _primitive_reduced_entry(current)
    else:
        current = _isomorphic_reduced_entry(current)
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
        if candidate.operation not in reduced_by_action:
            reduced_by_action[candidate.operation] = _canonical_sites(
                _translation_normal_form(_apply_action(current, candidate.operation))
            )
    choices = (
        _geometric_choice(candidate, reduced_by_action[candidate.operation], current.cell.basis)
        for candidate in candidates
    )
    _key, best = min(choices, key=lambda value: value[0])
    return _restore_source_metadata(_canonical_orientation(best), structure)


def canonical_asu_protostructure(
    structure: StructureLike,
    *,
    tolerance: float | None = None,
    factors: tuple[Fraction | float | int, ...] = (Fraction(1, 5), 1, 5),
    preserve_chirality: bool = True,
) -> ASUStructure:
    """Recognize and canonicalize a structure using the protostructure-first convention.

    Recognition uses the established tolerance sweep and deterministic P1 frames. The accepted ASU
    is then handled exactly, without an upward pseudosymmetry search.

    :param structure: The measured structure to recognize.
    :param tolerance: Base Cartesian recognition tolerance, or ``None`` to derive it.
    :param factors: Multipliers for the recognition tolerance sweep.
    :param preserve_chirality: Whether to keep the recognized enantiomorphic group.
    :return: The protostructure-first canonical asymmetric unit.
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
    normalized_p1 = _preconditioned_p1(source_view)
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
    result = _canonical_protostructure_asu(winner, preserve_chirality=preserve_chirality)
    return _restore_source_metadata(result, source_view)
