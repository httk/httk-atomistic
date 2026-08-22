"""Build magnetic structures from neutral mCIF mappings."""

import logging
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any

from httk.core import FracVector

from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.crystalaxis import CrystalAxisSiteMoments
from httk.atomistic.models.sites.sites import Sites
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.modulated import ModulatedStructure
from httk.atomistic.models.structure.symops import SymopsStructure
from httk.atomistic.models.structure.unitcell import UnitcellStructure
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.xyz import operation_from_xyz, operation_from_xyzt

from .cif_structures import (
    _cell_from_cif,
    _combine_cif_species,
    _decode_type_symbol,
    _exact_positions,
    _repair_cif_occupancy,
    _species_name,
)

__all__ = ["symops_structures_from_mcif"]

_MAX_IMPLICIT_SPATIAL_CLUSTER_TOLERANCE = 0.05
_MIN_IMPLICIT_SPATIAL_CLUSTER_TOLERANCE = 0.002

type _PositionKey = tuple[Fraction, ...]


def _perfect_orbit_matching(
    first: frozenset[_PositionKey],
    second: frozenset[_PositionKey],
    close: Callable[[_PositionKey, _PositionKey], bool],
) -> dict[_PositionKey, _PositionKey] | None:
    """Return a deterministic one-to-one proximity matching between two orbits."""
    if len(first) != len(second):
        return None
    candidates = {left: tuple(right for right in sorted(second) if close(left, right)) for left in first}
    if any(not choices for choices in candidates.values()):
        return None
    matched_right: dict[_PositionKey, _PositionKey] = {}

    def augment(left: _PositionKey, visited: set[_PositionKey]) -> bool:
        for right in candidates[left]:
            if right in visited:
                continue
            visited.add(right)
            previous = matched_right.get(right)
            if previous is None or augment(previous, visited):
                matched_right[right] = left
                return True
        return False

    for left in sorted(first, key=lambda key: (len(candidates[key]), key)):
        if not augment(left, set()):
            return None
    return {left: right for right, left in matched_right.items()}


def _mcif_spatial_tolerance(structure: SymopsStructure) -> float:
    """Return a bounded tolerance for collapsing rounded mCIF symmetry images."""
    from httk.atomistic.symmetry.recognition import DEFAULT_TOLERANCE

    fractional_precision = structure.coordinate_precision
    if fractional_precision is None:
        return DEFAULT_TOLERANCE
    longest = max(length.to_float() for length in structure.cell.lengths)
    cartesian_precision = fractional_precision * Fraction(str(longest))
    basis_precision = structure.basis_precision
    if basis_precision is not None and basis_precision > cartesian_precision:
        cartesian_precision = basis_precision
    return min(
        max(float(cartesian_precision) * 2, DEFAULT_TOLERANCE, _MIN_IMPLICIT_SPATIAL_CLUSTER_TOLERANCE),
        _MAX_IMPLICIT_SPATIAL_CLUSTER_TOLERANCE,
    )


def symops_structures_from_mcif(payload: Mapping[str, Any]) -> list[SymopsStructure | ModulatedStructure]:
    """Build one magnetic structure per block in a neutral mCIF payload.

    :param payload: The loaded whole-mCIF payload or one loaded mCIF block.
    :return: One symmetry-operations or modulated structure for each data block.
    :raises ValueError: If a block has invalid format, moments, or magnetic symmetry operations.
    """
    blocks = payload.get("blocks")
    if blocks is None:
        return [_structure_from_mcif_block(payload, "mcif block")]
    if not blocks:
        unparsed = payload.get("unparsed") or []
        if unparsed:
            detail = "; ".join(f"block {item['block']!r}: {item['reason']}" for item in unparsed)
            raise ValueError(f"this mCIF holds no structure blocks ({detail})")
        raise ValueError("this mCIF holds no structure blocks")
    repair = bool(payload.get("repair", False))
    return [
        _structure_from_mcif_block(
            {**block, "repair": bool(block.get("repair", repair))},
            f"mcif block {index}",
        )
        for index, block in enumerate(blocks)
    ]


def _structure_from_mcif_block(data: Mapping[str, Any], block_name: str) -> SymopsStructure | ModulatedStructure:
    if data.get("format") != "mcif":
        raise ValueError(f"{block_name} is not an 'mcif' mapping")
    if data.get("incomm") is not None:
        return ModulatedStructure(data)

    cell = _cell_from_cif(data)
    positions = _exact_positions(data)
    symbols = list(data["symbols"])
    labels = list(data.get("labels") or symbols)
    species, species_at_sites = _species(data, symbols, labels)
    site_moments = _moments(data, cell)
    return SymopsStructure(
        cell,
        Sites(positions, data.get("coordinate_precision")),
        species,
        species_at_sites,
        _symops(data, block_name),
        site_moments=site_moments,
        moment_component_resolutions=data.get("magmom_component_resolutions"),
        moment_component_esds=data.get("magmom_component_esds"),
        moment_symmforms=data.get("magmom_symmforms"),
        bns_number=data.get("bns_nbr"),
        bns_label=data.get("bns_name"),
    )


def _species(data: Mapping[str, Any], symbols: list[str], labels: list[str]) -> tuple[list[Species], list[str]]:
    occupancies = data.get("occupancies")
    occupancies_exact = data.get("occupancies_exact")
    occupancy_precisions = data.get("occupancy_precisions")
    masses = data.get("masses")
    by_name: dict[str, Species] = {}
    species_at_sites: list[str] = []
    warned_type_symbols: set[str] = set()
    for index, (symbol, label) in enumerate(zip(symbols, labels)):
        if occupancies_exact is not None and occupancies_exact[index] is not None:
            occupancy = occupancies_exact[index]
        elif occupancies is None:
            occupancy = 1
        elif occupancies[index] is None:
            raise ValueError(f"mCIF occupancy is missing for site {label!r}")
        else:
            occupancy = occupancies[index]
        occupancy = _repair_cif_occupancy(
            occupancy,
            label=label,
            block_name="mCIF",
            repair=bool(data.get("repair", False)),
        )
        raw_symbol = symbol
        stated_mass = None if masses is None else masses[index]
        decoded = _decode_type_symbol(raw_symbol, stated_mass)
        if not decoded.recognized and raw_symbol not in warned_type_symbols:
            logging.getLogger(__name__).warning(
                f"unrecognized CIF atom-type symbol {raw_symbol!r}; represented as chemical symbol 'X' "
                f"with species label {decoded.species_label!r}",
                extra={"context": "cif"},
            )
            warned_type_symbols.add(raw_symbol)
        name = _species_name(raw_symbol, label, occupancy)
        if name not in by_name:
            precision = None if occupancy_precisions is None else occupancy_precisions[index]
            by_name[name] = Species(
                name=name,
                chemical_symbols=(decoded.chemical_symbol,),
                concentration=(occupancy,),
                mass=(decoded.mass,) if decoded.mass is not None else None,
                original_name=None if label == raw_symbol else label,
                concentration_precision=(precision,) if occupancy_precisions is not None else None,
                charges=(decoded.charge,) if decoded.charge is not None else None,
                labels=(decoded.species_label,) if decoded.species_label is not None else None,
            )
        species_at_sites.append(name)
    return list(by_name.values()), species_at_sites


def _moments(data: Mapping[str, Any], cell: Cell) -> Any:
    basis = data.get("moment_basis")
    if basis is None:
        return None
    rows = data.get("magmoms_exact")
    if rows is None:
        raise ValueError("mCIF declares a moment basis but has no exact magnetic moments")
    exact_rows = tuple(tuple(Fraction(value) for value in row) for row in rows)
    precision = data.get("magmom_precision")
    if basis == "crystalaxis":
        return CrystalAxisSiteMoments(exact_rows, cell, precision=precision)
    if basis == "cartesian":
        return CartesianSiteMoments(exact_rows, precision=precision)
    raise ValueError(f"unsupported mCIF moment basis {basis!r}")


def _spatial_structure_from_mcif(
    structure: SymopsStructure, *, tolerance: float | None = None, repair: bool = False
) -> UnitcellStructure:
    """Project an mCIF structure onto spatial orbits with combined disorder."""
    from httk.atomistic.symmetry.recognition import _cartesian_distance_squared

    if tolerance is None:
        tolerance = _mcif_spatial_tolerance(structure)
    tolerance_squared = tolerance * tolerance
    species_by_name = {species.name: species for species in structure.species}
    groups: list[tuple[list[frozenset[_PositionKey]], list[int]]] = []
    vectors: dict[_PositionKey, FracVector] = {}
    float_vectors: dict[_PositionKey, tuple[float, float, float]] = {}
    basis = structure.cell.basis.to_floats()
    float_slack = 1e-12 * max(1.0, *(abs(value) for row in basis for value in row))
    float_limit_squared = (tolerance + float_slack) ** 2

    def close(first: _PositionKey, second: _PositionKey) -> bool:
        left_float = float_vectors.setdefault(first, (float(first[0]), float(first[1]), float(first[2])))
        right_float = float_vectors.setdefault(second, (float(second[0]), float(second[1]), float(second[2])))
        difference = tuple((left - right + 0.5) % 1 - 0.5 for left, right in zip(left_float, right_float))
        cartesian = tuple(sum(difference[axis] * basis[axis][component] for axis in range(3)) for component in range(3))
        if sum(value * value for value in cartesian) > float_limit_squared:
            return False
        left = vectors.setdefault(first, FracVector(first))
        right = vectors.setdefault(second, FracVector(second))
        return _cartesian_distance_squared(left - right, structure.cell) <= tolerance_squared

    def overlaps(first: frozenset[_PositionKey], second: frozenset[_PositionKey]) -> bool:
        return any(close(left, right) for left in first for right in second)

    def circular_mean(cluster: list[_PositionKey]) -> _PositionKey:
        anchor = FracVector(cluster[0])
        anchor_values = anchor.to_fractions()
        offsets = [(FracVector(key) - anchor).normalize_half().to_fractions() for key in cluster]
        return tuple(
            (anchor_values[axis] + sum((offset[axis] for offset in offsets), start=Fraction(0)) / len(cluster)) % 1
            for axis in range(3)
        )

    for index, site in enumerate(structure.listed_sites.reduced_coords):
        exact_keys = frozenset(
            tuple(operation.apply_wrapped(site).to_fractions()) for operation, _time_reversal in structure.symops
        )
        clusters: list[list[tuple[Fraction, ...]]] = []
        for key in sorted(exact_keys):
            for cluster in clusters:
                if any(close(key, previous) for previous in cluster):
                    cluster.append(key)
                    break
            else:
                clusters.append([key])
        keys = frozenset(circular_mean(cluster) for cluster in clusters)
        overlapping = [
            group_index
            for group_index, (previous_orbits, _members) in enumerate(groups)
            if any(overlaps(keys, previous) for previous in previous_orbits)
        ]
        if not overlapping:
            groups.append(([keys], [index]))
            continue
        if len(overlapping) != 1:
            raise ValueError("mCIF spatial projection has one listed site partially overlapping several earlier orbits")
        group_index = overlapping[0]
        previous_orbits, members = groups[group_index]
        if any(_perfect_orbit_matching(keys, previous, close) is None for previous in previous_orbits):
            raise ValueError("mCIF spatial projection has partially overlapping listed-site orbits")
        previous_orbits.append(keys)
        members.append(index)

    positions: list[tuple[Fraction, ...]] = []
    species_at_sites: list[str] = []
    combined_by_name: dict[str, Species] = {}
    for orbit_sets, indices in groups:
        sources = []
        for index in indices:
            name = structure.listed_species_at_sites[index]
            species = species_by_name[name]
            sources.append((species, species.original_name or name))
        sources.sort(key=lambda item: (item[0].chemical_symbols, item[1], item[0].name))
        combined = _combine_cif_species(
            sources,
            block_name="mCIF spatial projection",
            repair_overoccupancy=repair,
            drop_partial_masses=repair,
        )
        previous = combined_by_name.get(combined.name)
        if previous is not None and previous != combined:
            raise ValueError(
                f"mCIF spatial projection forms species name {combined.name!r} with conflicting definitions"
            )
        combined_by_name.setdefault(combined.name, combined)
        anchor_orbit = min(orbit_sets, key=lambda orbit: tuple(sorted(orbit)))
        aligned: dict[_PositionKey, list[_PositionKey]] = {key: [] for key in anchor_orbit}
        for orbit in orbit_sets:
            matching = _perfect_orbit_matching(anchor_orbit, orbit, close)
            if matching is None:
                raise RuntimeError("mCIF spatial orbit matching changed after grouping")
            for anchor_key, matched_key in matching.items():
                aligned[anchor_key].append(matched_key)
        ordered = sorted(circular_mean(cluster) for cluster in aligned.values())
        positions.extend(ordered)
        species_at_sites.extend((combined.name,) * len(ordered))

    return UnitcellStructure(
        structure.cell,
        Sites(positions, structure.coordinate_precision),
        tuple(combined_by_name.values()),
        species_at_sites,
        charge=structure.charge,
    )


def _parse_operation(value: str) -> tuple[AffineOperation, int]:
    try:
        if len(value.split(",")) == 3:
            return operation_from_xyz(value), 1
        return operation_from_xyzt(value)
    except ValueError as error:
        raise ValueError(f"cannot parse mCIF symmetry operation {value!r}: {error}") from error


def _symops(data: Mapping[str, Any], block_name: str) -> tuple[tuple[AffineOperation, int], ...]:
    raw_base = data.get("symops_xyz")
    if not raw_base:
        raise ValueError(f"{block_name} has no magnetic symmetry operations (symops_xyz is empty)")
    raw_centerings = data.get("centerings_xyz") or ("x,y,z,+1",)
    base = tuple(_parse_operation(value) for value in raw_base)
    centerings = tuple(_parse_operation(value) for value in raw_centerings)
    identity = AffineOperation.identity().matrix
    for raw, (operation, _) in zip(raw_centerings, centerings):
        if operation.matrix != identity:
            raise ValueError(f"mCIF centering must be a pure translation: {raw!r}")
    return tuple(
        (centering_operation * base_operation, base_time * centering_time)
        for base_operation, base_time in base
        for centering_operation, centering_time in centerings
    )
