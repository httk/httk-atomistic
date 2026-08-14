"""Exact symmetry-preserving structure alignment and interpolation."""

import itertools
from dataclasses import dataclass
from fractions import Fraction
from functools import cache

from httk.core import FracVector, SurdVector, register_citation

from httk.atomistic import data
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import ASUStructure, WyckoffSite
from httk.atomistic.symmetry._periodicity_guard import require_full_periodicity
from httk.atomistic.symmetry._standardization_common import (
    _matrix_column_sum_factor,
    _matrix_row_sum_factor,
    _scaled_precision,
)
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.lift import _apply_normalizer_operation, _wrapped, rerepresent
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup
from httk.atomistic.symmetry.subgroups import _standard_input, subgroup_closure

__all__ = [
    "CommonSubgroupResult",
    "StructurePath",
    "common_subgroup_representation",
    "interpolate_structures",
    "represent_like",
]

_MAX_PAIRING_PERMUTATIONS = 40_320


@cache
def _register_subgroup_matching_citation() -> None:
    """Register the subgroup-matching thesis citation, once per process."""
    register_citation(
        applies_to=(
            "The structure-matching and symmetry-path features (represent_like, "
            "common_subgroup_representation, interpolate_structures) build on Edvard "
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


def _pair_score(candidate: ASUStructure, reference: ASUStructure) -> tuple[Fraction, tuple[tuple[int, int], ...]]:
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


def _aligned(end: ASUStructure, reference: ASUStructure, *, tolerance: float | None) -> _Alignment:
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
        tuple[Fraction, tuple[tuple[str, str, tuple[Fraction, ...]], ...], ASUStructure, tuple[tuple[int, int], ...]]
        | None
    ) = None
    for candidate in candidates.values():
        try:
            score, pairs = _pair_score(candidate, reference_standard)
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
