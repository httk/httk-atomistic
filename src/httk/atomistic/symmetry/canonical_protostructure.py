"""Anonymous protostructure-first canonicalization of measured crystal structures.

This convention chooses the anonymous Wyckoff occupation pattern, then its species assignment,
and only then the exact geometry. Its finite affine search uses the same tabulated normalizer scope
as the legacy canonicalizer, taken in groups 2 to 15 also after the bare canonicalizers' kept cell
changes, and does not enumerate an infinite affine normalizer. In groups 3 to 15 the cell is then
reduced within the cell changes that keep every letter, so it does not depend on the input cell;
in P-1 the cell is the one with the least ``(a·a, b·b, c·c)`` among the cells that keep the
canonical letters, found exactly from a Selling-reduced superbase.
Public measured structure entry points optionally perform the established upward pseudosymmetry
search first.
"""

import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import combinations, product
from typing import Any

from httk.core import FracVector, SurdVector

from httk.atomistic import data
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.prototype.notation import class_order_key
from httk.atomistic.models.species.classes import _class_structure, _species_class
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure, WyckoffSite
from httk.atomistic.models.structure.bonds import (
    _AmbiguousMatch,
    _bonds_key,
    _carried,
    _cell_domain,
    _CoincidentAtoms,
    _dropped,
    _matched,
    _orbit_index,
    _ShapeConflict,
)
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.models.structuretype.fundamental import FundamentalDomainTemplate
from httk.atomistic.reduction import _is_niggli_parameters
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
    _exact_translations,
    _ext_gcd,
    _isomorphic_reduced_entry,
    _niggli_reduced_entry,
    _p1_metric_automorphism_operations,
    _primitive_reduced_entry,
    _pseudo_translations,
    _site_key,
    _translation_normal_form,
)
from httk.atomistic.symmetry.limits import _checkpoint
from httk.atomistic.symmetry.recognition import structure_tolerance
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup
from httk.atomistic.symmetry.subgroups import _standard_input

__all__ = ["canonical_asu_protostructure", "canonical_asu_protostructure_assignments"]


@dataclass(frozen=True)
class _Candidate:
    operation: AffineOperation
    basis_change: FracVector
    metric_key: tuple[Any, ...]
    reduction_key: tuple[Any, ...]


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
    restored = ASUStructure(
        result.cell,
        result.spacegroup,
        result.wyckoff_sites,
        result.species,
        transform=result.transform,
        coordinate_precision=result.coordinate_precision,
        charge=result.charge,
        **metadata,
    )
    return _carried(result, restored)


def _restore_assignment(result: ASUStructure, assignment: tuple[Species, ...], source: Any) -> ASUStructure:
    """Name the sites by the assignment, sorted by (species name, letter, parameters).

    The sort makes the site order a function of the crystal: the anonymous order follows the
    input's anonymous frame where species assignments tie.
    """
    by_anonymous = {anonymous.name: original for anonymous, original in zip(result.species, assignment, strict=True)}
    restored = ASUStructure(
        result.cell,
        result.spacegroup,
        sorted(
            (
                WyckoffSite(site.wyckoff, site.free_params, by_anonymous[site.species].name)
                for site in result.wyckoff_sites
            ),
            key=lambda site: (site.species, site.wyckoff, site.free_params.to_fractions()),
        ),
        source.species,
        transform=result.transform,
        coordinate_precision=result.coordinate_precision,
        charge=result.charge,
    )
    return _restore_source_metadata(_carried(result, restored), source)


def _with_precisions(result: ASUStructure, cell_precision: Any, coordinate_precision: Any) -> ASUStructure:
    restored = ASUStructure(
        Cell(result.cell, precision=cell_precision),
        result.spacegroup,
        result.wyckoff_sites,
        result.species,
        transform=result.transform,
        coordinate_precision=coordinate_precision,
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
    return _carried(result, restored)


def _exact_gram(cell: Cell) -> tuple[tuple[Any, ...], ...]:
    """Return the exact Gram of a cell: Fractions when it is rational, else SurdScalars."""
    metric = cell.metric()
    if metric.is_rational:
        return tuple(tuple(row) for row in metric.coefficient(1).to_fractions())
    return tuple(tuple(metric._element((row, column)) for column in range(3)) for row in range(3))


def _wrapped_atoms(view: UnitcellStructureView) -> list[tuple[str, tuple[Fraction, ...]]]:
    """Return the unit-cell sites as ``(species, coordinate wrapped into [0, 1))``."""
    coordinates = view.sites.reduced_coords.to_fractions()
    return [
        (species, tuple(value % 1 for value in row))
        for species, row in zip(view.species_at_sites, coordinates, strict=True)
    ]


def _lattice_vectors_of_norm(
    gram: tuple[tuple[Any, ...], ...], norm: Any, shifts: list[tuple[Fraction, ...]]
) -> list[tuple[Fraction, ...]]:
    """Return every vector ``z + s`` (``z`` integer, ``s`` in ``shifts``) of exact norm ``norm`` under ``gram``.

    Coordinate ``j`` obeys the Cauchy bound ``t_j² ≤ norm (G⁻¹)_jj``, with ``(G⁻¹)_jj`` an exact
    principal minor over the determinant (rational or surd); every comparison is exact.
    """
    determinant = (
        gram[0][0] * (gram[1][1] * gram[2][2] - gram[1][2] * gram[2][1])
        - gram[0][1] * (gram[1][0] * gram[2][2] - gram[1][2] * gram[2][0])
        + gram[0][2] * (gram[1][0] * gram[2][1] - gram[1][1] * gram[2][0])
    )
    limits = []
    bounds = []
    for j in range(3):
        k, m = (index for index in range(3) if index != j)
        limit = norm * (gram[k][k] * gram[m][m] - gram[k][m] * gram[m][k]) / determinant
        # seeded from a rational value near the limit, then corrected exactly
        seed = limit if isinstance(limit, Fraction) else Fraction(float(limit))
        bound = math.isqrt(max(math.floor(seed), 0))
        while bound > 0 and bound**2 > limit:
            _checkpoint()
            bound -= 1
        while (bound + 1) ** 2 <= limit:
            _checkpoint()
            bound += 1
        limits.append(limit)
        bounds.append(bound)
    found = []
    for z in product(*(range(-bound - 1, bound + 2) for bound in bounds)):
        _checkpoint()
        for shift in shifts:
            vector = tuple(z[j] + shift[j] for j in range(3))
            if (
                any(vector)
                and all(vector[j] ** 2 <= limits[j] for j in range(3))
                and _gram_dot(gram, vector, vector) == norm
            ):
                found.append(vector)
    return found


def _row_times(row: tuple[Fraction, ...], matrix: list[list[Fraction]]) -> tuple[Fraction, ...]:
    return tuple(sum((row[i] * matrix[i][j] for i in range(3)), start=Fraction(0)) for j in range(3))


def _translation_images(
    inverse_rows: list[list[Fraction]], shifts: list[tuple[Fraction, ...]]
) -> list[tuple[Fraction, ...]]:
    """Return the source translations modulo the result lattice, in result coordinates ``x T⁻¹``."""
    generators = [tuple(value % 1 for value in row) for row in inverse_rows]
    generators += [tuple(value % 1 for value in _row_times(shift, inverse_rows)) for shift in shifts]
    images: set[tuple[Fraction, ...]] = {(Fraction(0), Fraction(0), Fraction(0))}
    frontier = list(images)
    while frontier:
        _checkpoint()
        point = frontier.pop()
        for generator in generators:
            image = tuple((point[j] + generator[j]) % 1 for j in range(3))
            if image not in images:
                images.add(image)
                frontier.append(image)
    return sorted(images)


def _derived_precision(result: ASUStructure, source: StructureLike) -> ASUStructure:
    """Record on ``result`` the precisions the source's stated ones give along its best derivation.

    A valid derivation is an exact ``T`` (either determinant sign) whose rows lie in the source
    translation lattice (``Z³`` plus the exact self-translations of its unit cell, centrings
    included), with ``T G_src Tᵀ = G_res``, and under which the source unit-cell sites, mapped by
    ``x' = x T⁻¹`` and wrapped, are the result's up to one common translation (species-aware).
    Of these the ``T`` with the least ``(row sum of T, column sum of T⁻¹)`` gives
    ``cell.precision = ε_cell × rowsum(T)`` and ``coordinate_precision = ε_coord × colsum(T⁻¹)``,
    so the recorded precision depends on the source and the result, not on the path between
    them (a mirror pair records the same). A result cell ``|det T|`` times larger takes every
    image of each source site under the source lattice (:func:`_translation_images`). Rows are
    enumerated exactly in a proxy-reduced source basis (:func:`_lattice_vectors_of_norm`). Where no valid derivation exists (a tolerance-snapped
    recognition, a lifted crystal) the path value already on ``result`` is kept.
    """
    source_view = UnitcellStructureView(source)
    source_atoms = _wrapped_atoms(source_view)
    target = Counter(_wrapped_atoms(UnitcellStructureView(result)))
    if not source_atoms or not target:
        return result
    gram, result_gram = _exact_gram(source_view.cell), _exact_gram(result.cell)
    shifts = [(Fraction(0), Fraction(0), Fraction(0)), *_exact_translations(source_atoms)]
    # enumerate in a (proxy-)reduced basis B' = U B, where the Cauchy box is tight, then map back
    reduction = _proxy_niggli_transform(source_view.cell.metric()).to_fractions()
    unreduction = FracVector(reduction).inv().to_fractions()
    reduced_gram = tuple(tuple(_gram_dot(gram, row, column) for column in reduction) for row in reduction)
    reduced_shifts = [_row_times(shift, unreduction) for shift in shifts]
    rows = [
        [
            _row_times(vector, reduction)
            for vector in _lattice_vectors_of_norm(reduced_gram, result_gram[k][k], reduced_shifts)
        ]
        for k in range(3)
    ]
    population = Counter(species for species, _point in target.elements())
    anchor_species = min(population, key=lambda species: (population[species], species))
    anchor = next(point for species, point in target if species == anchor_species)
    best: tuple[Fraction, Fraction] | None = None
    for a in rows[0]:
        for b in rows[1]:
            if _gram_dot(gram, a, b) != result_gram[0][1]:
                continue
            for c in rows[2]:
                _checkpoint()
                if _gram_dot(gram, a, c) != result_gram[0][2] or _gram_dot(gram, b, c) != result_gram[1][2]:
                    continue
                change = FracVector((a, b, c))
                inverse = change.inv()
                norms = (_matrix_row_sum_factor(change), _matrix_column_sum_factor(inverse))
                if best is not None and norms >= best:
                    continue
                inverse_rows = inverse.to_fractions()
                images = _translation_images(inverse_rows, shifts)
                # a smaller cell sees each site once per image; a larger one (|det T| = k > 1) needs
                # the k images of every source site under the source lattice
                mapped: list[tuple[str, tuple[Fraction, ...]]] = []
                for species, x in source_atoms:
                    y = _row_times(x, inverse_rows)
                    mapped.extend((species, tuple((y[j] + image[j]) % 1 for j in range(3))) for image in images)
                # deduplicated, so a duplicated (species, coordinate) source site matches nothing and
                # falls back to the path value
                mapped = list(dict.fromkeys(mapped))
                for species, y in mapped:
                    if species != anchor_species:
                        continue
                    shift = tuple(anchor[j] - y[j] for j in range(3))
                    if Counter((name, tuple((p[j] + shift[j]) % 1 for j in range(3))) for name, p in mapped) == target:
                        best = norms
                        break
    if best is None:
        return result
    return _with_precisions(
        result,
        _scaled_precision(source_view.cell.precision, best[0]),
        _scaled_precision(source_view.sites.precision, best[1]),
    )


def _anonymize_asu(
    structure: ASUStructure,
    anonymous_species: tuple[Species, ...],
    assignment: tuple[Species, ...],
) -> ASUStructure:
    anonymous_by_original = {
        original.name: anonymous.name for anonymous, original in zip(anonymous_species, assignment, strict=True)
    }
    anonymous = ASUStructure(
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
    return _carried(structure, anonymous)


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
        _basis_key(structure.cell.basis),
    )


def _chemical_key(structure: ASUStructure) -> tuple[Any, ...]:
    return _site_key(structure), tuple(repr(species) for species in structure.species), _bonds_key(structure)


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
            _bonds_key(value),
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
    acted = ASUStructure(
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
    return _carried(structure, acted, operation)


def _with_basis(structure: ASUStructure, basis: SurdVector) -> ASUStructure:
    basis_change = _rational_basis_change(basis, structure.cell.basis)
    rebased = ASUStructure(
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
    return _carried(structure, rebased)


type _DiscreteKey = tuple[tuple[tuple[int, tuple[str, ...]], ...], tuple[Any, ...]]


def _site_letters(structure: ASUStructure, operation: AffineOperation) -> tuple[str, ...]:
    """Return the target Wyckoff letter of every site under one proven action."""
    return tuple(
        compile_wyckoff_action(structure.spacegroup, operation, site.wyckoff).target_letter
        for site in structure.wyckoff_sites
    )


def _letters_key(
    structure: ASUStructure, targets: tuple[str, ...], names: dict[str, str] | None = None
) -> _DiscreteKey:
    """Return the (anonymous, assigned) discrete key of the sites moved onto ``targets``.

    ``names`` renames the species of anonymized sites to the class names of their species, so
    the assigned part is the bare canonicalizer's key.
    """
    letters: dict[str, list[str]] = {}
    for site, target in zip(structure.wyckoff_sites, targets, strict=True):
        _checkpoint()
        letters.setdefault(names[site.species] if names else site.species, []).append(target)
    classes = {species: tuple(sorted(values)) for species, values in letters.items()}
    counts = {
        species: sum(structure.spacegroup.wyckoff_position(letter).multiplicity for letter in values)
        for species, values in classes.items()
    }
    anonymous = tuple(sorted(class_order_key(values, counts[species])[:2] for species, values in classes.items()))
    assignment = tuple(sorted(class_order_key(values, counts[species], species) for species, values in classes.items()))
    return anonymous, assignment


def _discrete_key(
    structure: ASUStructure, operation: AffineOperation, names: dict[str, str] | None = None
) -> _DiscreteKey:
    return _letters_key(structure, _site_letters(structure, operation), names)


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


type _AcChange = tuple[tuple[int, int], tuple[int, int]]


@cache
def _kernel_residues(spacegroup: Spacegroup) -> frozenset[_AcChange]:
    """Return, modulo 4, the ac-plane basis changes ``Y = X^-T`` of the cell changes ``X`` in ``K``.

    ``K`` is the kernel of the signature: the cell changes that keep every Wyckoff letter and
    operation. It contains every cell change congruent to the identity modulo 4 (design note
    §2.2), so one integer lift per unimodular residue decides membership. The 96 signatures
    cost about a second, once per group.
    """
    from httk.atomistic.symmetry.canonical_classification import _signature  # imports this module

    identity = _signature(spacegroup, AffineOperation.identity())
    lifts: dict[tuple[int, ...], tuple[int, ...]] = {}
    for a, b, c, d in product(range(-3, 4), repeat=4):
        if a * d - b * c in (1, -1):
            lifts.setdefault((a % 4, b % 4, c % 4, d % 4), (a, b, c, d))
    residues = set()
    for a, b, c, d in lifts.values():
        _checkpoint()
        if _signature(spacegroup, AffineOperation(((a, 0, b), (0, 1, 0), (c, 0, d)))) == identity:
            det = a * d - b * c
            residues.add(((det * d % 4, -det * c % 4), (-det * b % 4, det * a % 4)))
    return frozenset(residues)


def _ac_reduced(ac: tuple[Any, Any, Any], residues: frozenset[_AcChange]) -> tuple[tuple[Any, ...], list[_AcChange]]:
    """Return the least ``(a·a, c·c, a·c)`` over the ac basis changes with residues in ``residues``, and each one.

    ``ac`` is ``(a·a, a·c, c·c)``. This is the level-4 enumeration lemma of the design note §7: in a
    Lagrange-reduced basis every competing first row has ``|y| <= 4`` and is the first primitive
    vector of its residue class on either side of ``-B y / A``; each second row is the nearest
    ``±w + j·u`` with ``j`` in an admissible class modulo 4. Floats only bound the windows; every
    comparison is exact.
    """
    aa, ac_, cc = ac

    def form(u: tuple[int, int], v: tuple[int, int]) -> Any:
        return aa * (u[0] * v[0]) + ac_ * (u[0] * v[1] + u[1] * v[0]) + cc * (u[1] * v[1])

    r1, r2 = (1, 0), (0, 1)
    while True:
        _checkpoint()
        if form(r2, r2) < form(r1, r1):
            r1, r2 = r2, r1
        a, b = form(r1, r1), form(r1, r2)
        if not (2 * b > a or -2 * b > a):
            break
        # the float step size only guides; progress is checked exactly, else step by one
        k = round(float(b) / float(a))
        step = (r2[0] - k * r1[0], r2[1] - k * r1[1])
        smaller = form(r1, step)
        shrinks = (smaller < b and -smaller < b) if b > 0 else (smaller > b and -smaller > b)
        if k == 0 or not shrinks:
            k = 1 if b > 0 else -1
            step = (r2[0] - k * r1[0], r2[1] - k * r1[1])
        r2 = step
    fa, fb = float(a), float(b)
    first_rows = {first for first, _second in residues}
    rows = []
    for y in range(-4, 5):
        centre = -fb * y / fa
        for x in range(math.floor(centre) - 13, math.ceil(centre) + 14):
            _checkpoint()
            if math.gcd(x, y) == 1:
                u = (x * r1[0] + y * r2[0], x * r1[1] + y * r2[1])
                if (u[0] % 4, u[1] % 4) in first_rows:
                    rows.append((float(form(u, u)), u))
    bound = min(value for value, _u in rows) * (1 + 1e-6)
    lengths = [(form(u, u), u) for value, u in rows if value <= bound]
    least = min(length for length, _u in lengths)
    found = []
    for length, u in lengths:
        _checkpoint()
        if length != least:
            continue
        _gcd, s, t = _ext_gcd(*u)
        for sign in (1, -1):
            w = (-sign * t, sign * s)
            centre = -float(form(u, w)) / float(length)
            for rho in range(4):
                if ((u[0] % 4, u[1] % 4), ((w[0] + rho * u[0]) % 4, (w[1] + rho * u[1]) % 4)) not in residues:
                    continue
                start = math.floor((centre - rho) / 4)
                for j in range(rho + 4 * (start - 1), rho + 4 * (start + 3), 4):
                    v = (w[0] + j * u[0], w[1] + j * u[1])
                    found.append(((length, form(v, v), form(u, v)), (u, v)))
    key = min(value for value, _pair in found)
    return key, list(dict.fromkeys(pair for value, pair in found if value == key))


type _F2Matrix = tuple[tuple[int, ...], ...]
type _IntVector = tuple[int, ...]


@cache
def _f2_inverse_transpose(matrix: _F2Matrix) -> _F2Matrix:
    """Return ``matrix^-T`` over F₂: the cofactor matrix modulo 2, the determinant being 1."""
    return tuple(
        tuple(
            (
                matrix[(row + 1) % 3][(column + 1) % 3] * matrix[(row + 2) % 3][(column + 2) % 3]
                - matrix[(row + 1) % 3][(column + 2) % 3] * matrix[(row + 2) % 3][(column + 1) % 3]
            )
            % 2
            for column in range(3)
        )
        for row in range(3)
    )


def _gram_dot(gram: tuple[tuple[Any, ...], ...], u: tuple[Fraction | int, ...], v: tuple[Fraction | int, ...]) -> Any:
    """Return the exact inner product of two rational coordinate vectors under ``gram``."""
    total = gram[0][0] * 0
    for row in range(3):
        for column in range(3):
            if u[row] and v[column]:
                total += gram[row][column] * (u[row] * v[column])
    return total


def _selling_superbase(gram: tuple[tuple[Any, ...], ...]) -> tuple[_IntVector, ...]:
    """Return an obtuse superbase ``v₀..v₃`` (integer coordinates) of the lattice with the exact ``gram``.

    Selling reduction from ``(-b₁-b₂-b₃, b₁, b₂, b₃)``: while some ``v_i·v_j > 0``, add ``v_i``
    to the other two and negate it. Each step lowers ``Σ|v|²`` by ``2 v_i·v_j``, so it ends for
    every exact positive-definite Gram, surds included; only ring operations and exact signs are used.
    """
    vectors: list[_IntVector] = [(-1, -1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    while True:
        _checkpoint()
        pair = next(((i, j) for i, j in combinations(range(4), 2) if _gram_dot(gram, vectors[i], vectors[j]) > 0), None)
        if pair is None:
            return tuple(vectors)
        i, _j = pair
        k, m = (index for index in range(4) if index not in pair)
        vectors[k] = tuple(x + y for x, y in zip(vectors[k], vectors[i], strict=True))
        vectors[m] = tuple(x + y for x, y in zip(vectors[m], vectors[i], strict=True))
        vectors[i] = tuple(-x for x in vectors[i])


def _p1_class_minima(gram: tuple[tuple[Any, ...], ...]) -> dict[_IntVector, tuple[Any, tuple[_IntVector, ...]]]:
    """Return each nonzero class of ``L/2L`` with its exact least norm and every vector attaining it.

    With an obtuse superbase (:func:`_selling_superbase`) and conorms ``p_ij = -v_i·v_j``,
    ``|Σ n_i v_i|² = Σ p_ij (n_i - n_j)²``. A class is a split of ``{0, 1, 2, 3}`` by the parity
    of ``n``; its minimum is the sum of the conorms across the split, attained exactly where
    ``|n_i - n_j| = 1`` across and ``n_i = n_j`` within the split for every nonzero conorm. The
    graph of nonzero conorms is connected, so with ``n₀ = 0`` every minimizer has ``|n_i| ≤ 3``.
    A degenerate lattice (some zero conorm) can have up to four pairs of minimizers in a class.
    Vectors and classes are integer coordinates in the basis of ``gram``; no norm is evaluated.
    """
    vectors = _selling_superbase(gram)
    conorms = {(i, j): -_gram_dot(gram, vectors[i], vectors[j]) for i, j in combinations(range(4), 2)}
    edges = [(pair, value) for pair, value in conorms.items() if value != 0]
    minima: dict[_IntVector, tuple[Any, list[_IntVector]]] = {}
    for tail in product(range(-3, 4), repeat=3):
        _checkpoint()
        n = (0, *tail)
        odd = tuple(value % 2 for value in n)
        if not any(odd):
            continue
        if any(abs(n[i] - n[j]) != 1 if odd[i] != odd[j] else n[i] != n[j] for (i, j), _value in edges):
            continue
        vector = tuple(sum(n[index] * vectors[index][axis] for index in range(4)) for axis in range(3))
        residue = tuple(value % 2 for value in vector)
        if residue not in minima:
            norm = sum((value for (i, j), value in edges if odd[i] != odd[j]), start=gram[0][0] * 0)
            minima[residue] = (norm, [])
        minima[residue][1].append(vector)
    if len(minima) != 7:
        raise AssertionError("P-1 class minima: a Selling-reduced superbase must give all seven classes")
    return {residue: (norm, tuple(found)) for residue, (norm, found) in minima.items()}


def _is_niggli_gram(metric_key: tuple[Any, ...]) -> bool:
    """Return whether a row-major exact Gram (rational or surd) is the Niggli-reduced one."""
    g = metric_key
    return _is_niggli_parameters((g[0], g[4], g[8], 2 * g[5], 2 * g[2], 2 * g[1]))


def _p1_candidates(structure: ASUStructure, names: dict[str, str] | None = None) -> tuple[_Candidate, ...]:
    """Return every proper P-1 description with the least discrete key and the least ``(a·a, b·b, c·c)``.

    The letters of an action ``(M, t)`` depend only on ``(M mod 2, 2t mod 2)``, so a cell
    ``B' = P B`` keeps the least key exactly when ``P mod 2`` is the inverse transpose of a
    reaching ``M mod 2``, with the matching half translations: each row of ``P`` must lie in a
    prescribed class of ``L/2L``, and nothing else is constrained. Each row takes the least
    vectors of its class (:func:`_p1_class_minima`); every class triple with the least norms,
    and every signed minimizer triple with determinant +1, is kept. The reduction key is those
    norms, then whether the Gram is not the Niggli one. The candidates depend only on the
    lattice and the letters, not on the input cell; with every atom on the general position
    the cell is the Niggli cell.
    """
    from httk.atomistic.symmetry.canonical_classification import _letter_maps  # imports this module

    letters = tuple(site.wyckoff for site in structure.wyckoff_sites)
    keys: dict[tuple[str, ...], _DiscreteKey] = {}
    reached: list[tuple[_DiscreteKey, AffineOperation]] = []
    for operation, letter_map in _letter_maps(structure.spacegroup).items():
        targets = tuple(letter_map[letter] for letter in letters)
        if targets not in keys:
            keys[targets] = _letters_key(structure, targets, names)
        reached.append((keys[targets], operation))
    best_key = min(keys.values())
    # the row classes P mod 2 of the admissible cells, each with its half translations 2t mod 2
    shifts: dict[_F2Matrix, set[_IntVector]] = {}
    for key, operation in reached:
        if key == best_key:
            matrix = tuple(tuple(int(value) % 2 for value in row) for row in operation.matrix.to_fractions())
            shift = tuple(int(2 * value) % 2 for value in operation.vector.to_fractions())
            shifts.setdefault(_f2_inverse_transpose(matrix), set()).add(shift)
    source_metric = structure.cell.metric()
    gram = tuple(tuple(source_metric._element((row, column)) for column in range(3)) for row in range(3))
    minima = _p1_class_minima(gram)
    norms = {rows: tuple(minima[row][0] for row in rows) for rows in shifts}
    least = min(norms.values())
    candidates: list[_Candidate] = []
    for rows, row_shifts in shifts.items():
        if norms[rows] != least:
            continue
        for vectors in product(*(minima[row][1] for row in rows)):
            _checkpoint()
            basis_change = FracVector(vectors)
            # a triple of minimizers of independent classes can have determinant 3 (degenerate lattices)
            if basis_change.det() != 1:
                continue
            basis_map = SurdVector(basis_change)
            metric = basis_map * source_metric * basis_map.T()
            metric_key = tuple(metric._element((row, column)) for row in range(3) for column in range(3))
            reduction_key = (least, not _is_niggli_gram(metric_key))
            point = basis_change.T().inv()
            for shift in sorted(row_shifts):
                operation = AffineOperation(point, tuple(Fraction(value, 2) for value in shift))
                candidates.append(_Candidate(operation, basis_change, metric_key, reduction_key))
    return tuple(candidates)


def _discrete_candidates(
    structure: ASUStructure, names: dict[str, str] | None = None, *, proper: bool = False
) -> tuple[_Candidate, ...]:
    """Return every right-handed description with the least discrete key, its cell reduced within its class.

    P-1 (standard setting) is :func:`_p1_candidates`. In groups 3 to 15 the tabulated
    descriptions are also taken after each kept cell change ``sigma`` of the bare
    canonicalizers, so the least discrete key is the canonical bare one. Every ``sigma``
    reaching it contributes, and each description is moved by the cell changes ``K`` that keep
    every letter (:func:`_kernel_residues`) to its least ``(a·a, c·c, a·c)``
    (:func:`_ac_reduced`), the reduction key; the cell does not depend on the input cell. The structure must be
    right-handed (:func:`_right_handed`); left-handed descriptions are skipped, since an improper
    group operation gives each a right-handed twin with the same keys. With ``proper`` (a Sohncke
    group) only proper descriptions compete, so the key is the hand-preserving bare one.
    """
    group = structure.spacegroup
    if group.it_number == 2 and group.is_standard_setting:
        return _p1_candidates(structure, names)
    from httk.atomistic.symmetry.canonical_classification import _cell_changes, _letter_maps  # imports this module

    identity_matrix = FracVector.eye((3, 3))
    point_operations = tuple((operation, operation.matrix.T().inv()) for operation in _point_operations(structure))
    metric_keys: dict[FracVector, tuple[Any, ...]] = {}
    source_metric = structure.cell.metric()
    sigmas = _cell_changes(group) if group.is_standard_setting else (AffineOperation.identity(),)
    letter_maps = _letter_maps(group) if group.is_standard_setting else {}
    residues = _kernel_residues(group) if group.is_standard_setting and 3 <= group.it_number <= 15 else None
    if residues is not None and proper:
        # only the K changes of determinant +1 keep a proper description proper; residues modulo 4
        # tell the determinants +1 and -1 apart
        residues = frozenset(r for r in residues if (r[0][0] * r[1][1] - r[0][1] * r[1][0] - 1) % 4 == 0)
    bases: list[tuple[FracVector, AffineOperation]] = []
    for representative in _representatives(structure):
        _checkpoint()
        for translation in _discrete_normalizer_translations(group):
            _checkpoint()
            operation = (
                representative
                if not any(translation)
                else AffineOperation(identity_matrix, translation) * representative
            )
            bases.extend((point_basis_change, operation) for _point_operation, point_basis_change in point_operations)
    site_letters: dict[AffineOperation, tuple[str, ...]] = {}
    keys: dict[tuple[str, ...], _DiscreteKey] = {}

    def targets(operation: AffineOperation, sigma_map: dict[str, str]) -> tuple[str, ...]:
        if operation not in site_letters:
            site_letters[operation] = _site_letters(structure, operation)
        return tuple(sigma_map.get(letter, letter) for letter in site_letters[operation])

    def key_of(letters: tuple[str, ...]) -> _DiscreteKey:
        if letters not in keys:
            keys[letters] = _letters_key(structure, letters, names)
        return keys[letters]

    def metric_key_of(basis_change: FracVector) -> tuple[Any, ...]:
        if basis_change not in metric_keys:
            basis_map = SurdVector(basis_change)
            metric = basis_map * source_metric * basis_map.T()
            metric_keys[basis_change] = tuple(metric._element((row, column)) for row in range(3) for column in range(3))
        return metric_keys[basis_change]

    # per determinant of the base action, so that with ``proper`` sigma only meets bases of its own sign
    distinct: dict[bool, set[tuple[str, ...]]] = {True: set(), False: set()}
    for _point, operation in bases:
        distinct[operation.determinant() > 0].add(targets(operation, {}))
    sigma_keys = [
        min(
            (
                key_of(tuple(letter_maps.get(sigma, {}).get(letter, letter) for letter in letters))
                for letters in (distinct[sigma.determinant() > 0] if proper else distinct[True] | distinct[False])
            ),
            default=None,
        )
        for sigma in sigmas
    ]
    best_key = min(key for key in sigma_keys if key is not None)
    reductions: dict[tuple[Any, ...], tuple[tuple[Any, ...], list[_AcChange]]] = {}
    candidates: list[_Candidate] = []

    def add(site_operation: AffineOperation, basis_change: FracVector, reduction_key: tuple[Any, ...]) -> None:
        if basis_change.det().sign() > 0:
            candidates.append(_Candidate(site_operation, basis_change, metric_key_of(basis_change), reduction_key))

    for sigma in sigmas:
        _checkpoint()
        sigma_map = letter_maps.get(sigma, {})
        for point_basis_change, base in bases:
            _checkpoint()
            if proper and sigma.determinant() != base.determinant():
                continue
            if key_of(targets(base, sigma_map)) != best_key:
                continue
            operation = sigma * base
            basis_change = point_basis_change * operation.matrix.T().inv()
            if residues is None:
                add(operation, basis_change, ())
                continue
            metric_key = metric_key_of(basis_change)
            ac = (metric_key[0], metric_key[2], metric_key[8])
            if ac not in reductions:
                reductions[ac] = _ac_reduced(ac, residues)
            reduction_key, changes = reductions[ac]
            for (y00, y01), (y10, y11) in changes:
                change = FracVector(((y00, 0, y01), (0, 1, 0), (y10, 0, y11)))
                add(
                    AffineOperation(change.T().inv(), (0, 0, 0)) * operation,
                    change * basis_change,
                    reduction_key,
                )
    return tuple(candidates)


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
        # the network before the raw Cartesian basis, which the orientation rebuild replaces
        _bonds_key(represented),
        _basis_key(represented.cell.basis),
    )
    return key, represented


def _p1bar_primitive_entry(structure: ASUStructure) -> ASUStructure:
    """Collapse an exact P-1 supercell to its primitive cell, whatever its basis.

    The P1 expansion is collapsed by :func:`_primitive_reduced_entry` (``Z³`` and every exact
    self-translation); the inversion through the origin keeps that finer lattice, so the result
    is P-1 again, one site per inversion orbit. The tabulated isomorphic self-lifts reach only
    some sublattice orientations, so the collapse would otherwise depend on the input cell. A
    bond network that the self-translations do not keep makes the declared cell the crystal's
    translation lattice: the structure is then returned unchanged, with its network.
    """
    if not _pseudo_translations(structure):
        return structure
    group = structure.spacegroup
    points = [
        WyckoffSite("a", FracVector(point).normalize(), site.species)
        for site in structure.wyckoff_sites
        for point in group.wyckoff_position(site.wyckoff).coordinates(site.free_params)
    ]
    expanded = _carried(
        structure,
        ASUStructure(
            structure.cell,
            1,
            points,
            structure.species,
            coordinate_precision=structure.coordinate_precision,
            charge=structure.charge,
        ),
    )
    primitive = _primitive_reduced_entry(expanded)
    sites = []
    seen: set[tuple[str, tuple[Fraction, ...]]] = set()
    for site in primitive.wyckoff_sites:
        _checkpoint()
        point = tuple(site.free_params.to_fractions())
        if (site.species, point) in seen:
            continue
        seen.update({(site.species, point), (site.species, tuple(-value % 1 for value in point))})
        match = group.identify_wyckoff(site.free_params)
        if match is None:
            raise ValueError("P-1 supercell collapse: a point on no Wyckoff position; this is a logic error")
        sites.append(WyckoffSite(match[0].letter, match[1], site.species))
    collapsed = ASUStructure(
        primitive.cell,
        group,
        sites,
        structure.species,
        coordinate_precision=primitive.coordinate_precision,
        charge=primitive.charge,
    )
    collapsed = _carried(primitive, collapsed)
    return structure if structure.bonds is not None and collapsed.bonds is None else collapsed


def _validate_exact_input(structure: ASUStructure) -> None:
    require_full_periodicity(structure.cell, "canonical_asu_protostructure")
    if any(site.moment is not None for site in structure.wyckoff_sites):
        raise ValueError("canonical_asu_protostructure does not support structures with site moments")
    if structure.assemblies is not None:
        raise ValueError("canonical_asu_protostructure does not support structures with assemblies")
    if structure.molecular:
        raise ValueError("canonical_asu_protostructure does not support molecular structures")


@cache
def _sohncke(spacegroup: Spacegroup) -> bool:
    """Return whether every operation of the group is proper (the 65 Sohncke groups)."""
    return all(operation.determinant() > 0 for operation in spacegroup.symmetry_operations)


def _left_handed(domain: FundamentalDomainStructure) -> bool:
    """Return whether the standard-setting basis a structure stands for, after its setting transform, is left-handed."""
    return domain.cell.basis.det().sign() * domain.transform_from_standard.determinant() < 0


def _right_handed(domain: FundamentalDomainStructure) -> ASUStructure:
    """Return the same crystal as an ASU in a right-handed basis of the standard setting, or of its partner group.

    A left-handed basis is re-described improperly: by an improper group operation, else (a
    Sohncke group) an improper normalizer action, or, where none keeps the group, the inversion
    into the enantiomorphic partner. Its letters then read the crystal's own hand. A re-described
    structure keeps only its cell, sites, species and charge, with its precisions recorded from
    ``domain`` (:func:`_derived_precision`); sites lose retained representatives.
    """
    if isinstance(domain, ASUStructure) and not _left_handed(domain):
        return domain
    return _derived_precision(_right_handed_description(domain), domain)


def _right_handed_description(domain: FundamentalDomainStructure) -> ASUStructure:
    """Return :func:`_right_handed`'s re-description with its path precisions."""
    structure = _standard_input(
        _carried(
            domain,
            ASUStructure(
                domain.cell,
                domain.spacegroup,
                domain.wyckoff_sites,
                domain.species,
                transform=domain.transform,
                coordinate_precision=domain.coordinate_precision,
                charge=domain.charge,
            ),
        )
    )
    if structure.cell.basis.det().sign() > 0:
        return structure
    group = structure.spacegroup
    partner = group.record["it_number_enantiomorphic"]
    if partner is not None:
        inversion = AffineOperation(FracVector(((-1, 0, 0), (0, -1, 0), (0, 0, -1))), (0, 0, 0))
        target = Spacegroup.standard(partner)
        sites = []
        for site in structure.wyckoff_sites:
            _checkpoint()
            action = compile_wyckoff_action(group, inversion, site.wyckoff, target_spacegroup=target)
            sites.append(WyckoffSite(action.target_letter, action.apply(site.free_params), site.species))
        inverted = ASUStructure(
            Cell(-structure.cell.basis, precision=structure.cell.precision, periodicity=structure.cell.periodicity),
            target,
            sites,
            structure.species,
            transform=SettingTransform.identity(),
            coordinate_precision=structure.coordinate_precision,
            charge=structure.charge,
        )
        return _carried(structure, inverted, inversion)
    from httk.atomistic.symmetry.canonical_classification import _actions  # imports this module

    improper = next((op for op in group.symmetry_operations if op.determinant() < 0), None)
    if improper is None:
        # each of the 43 Sohncke groups without a partner has an improper normalizer action
        improper = next((action for action in _actions(group) if action.determinant() < 0), None)
    if improper is None:
        raise ValueError(f"no improper same-group action re-describes a left-handed cell of {group.setting}")
    return _apply_action(structure, improper)


def _right_handed_template(template: FundamentalDomainTemplate) -> FundamentalDomainTemplate:
    """Return a template in a right-handed basis, re-described as by :func:`_right_handed`."""
    if template.cell.basis.det().sign() > 0:
        return template
    turned = _right_handed(template._domain)
    return FundamentalDomainTemplate(
        turned.cell, turned.spacegroup, turned.wyckoff_sites, turned.species, turned.coordinate_precision
    )


def _canonical_protostructure_geometry_entry(
    current: ASUStructure,
    *,
    preserve_chirality: bool,
    names: dict[str, str] | None = None,
) -> ASUStructure:
    """Run the finite exact terminal from one already normalized lattice entry."""
    if current.spacegroup.it_number in (1, 2):
        current = _niggli_reduced_entry(current)
    current = _right_handed(_demote_sites(current))
    if not preserve_chirality:
        current = _enantiomorph(current) or current
        current = _demote_sites(current)

    candidates = _discrete_candidates(current, names, proper=_sohncke(current.spacegroup))

    def tiers(candidate: _Candidate) -> tuple[Any, ...]:
        return candidate.reduction_key, candidate.metric_key

    least = min(tiers(candidate) for candidate in candidates)
    # descriptions differing by a lattice translation give the same result: evaluate one of them
    unique: dict[tuple[FracVector, AffineOperation], _Candidate] = {}
    for candidate in candidates:
        if tiers(candidate) == least:
            unique.setdefault((candidate.basis_change, candidate.operation.wrapped()), candidate)
    candidates = tuple(unique.values())
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


def _own_enantiomorph_sohncke(spacegroup: Spacegroup) -> bool:
    """Return whether a group has only proper operations and is its own enantiomorphic partner.

    The enantiomer of :func:`_mirrored` keeps the group, so it applies whether or not the
    inversion through the origin keeps the group (I4₁, I4₁22 and F4₁32 need an origin shift).
    """
    return spacegroup.record["it_number_enantiomorphic"] is None and _sohncke(spacegroup)


def _mirrored(structure: ASUStructure) -> ASUStructure:
    """Return the enantiomer: the same fractional coordinates and group in the negated (left-handed) basis."""
    mirrored = ASUStructure(
        Cell(-structure.cell.basis, precision=structure.cell.precision, periodicity=structure.cell.periodicity),
        structure.spacegroup,
        structure.wyckoff_sites,
        structure.species,
        transform=structure.transform,
        coordinate_precision=structure.coordinate_precision,
        charge=structure.charge,
    )
    return _carried(structure, mirrored)


def _canonical_protostructure_geometry(
    structure: ASUStructure,
    *,
    preserve_chirality: bool = True,
    names: dict[str, str] | None = None,
) -> ASUStructure:
    """Canonicalize one anonymously labelled ASU by discrete occupation before geometry.

    With ``names`` the discrete occupation includes the species class names (:func:`_letters_key`).
    The input is first re-described in a right-handed basis (:func:`_right_handed`), and the
    representative is that crystal in a right-handed cell: in a Sohncke group only proper
    descriptions are used, so the discrete key keeps the hand. Without ``preserve_chirality`` an
    enantiomorphic pair maps to its lower member here; the two hands of a group that is its own partner are identified by the caller,
    which also runs the enantiomer (:func:`_canonical_protostructure_assignments_asu`).
    """
    if not isinstance(structure, ASUStructure):
        raise TypeError(f"expected ASUStructure, got {type(structure).__name__}")
    _validate_exact_input(structure)
    current = _standard_input(structure)
    if current.spacegroup.it_number == 1:
        current = _primitive_reduced_entry(current)
    elif current.spacegroup.it_number == 2:
        current = _p1bar_primitive_entry(current)
    else:
        current = _isomorphic_reduced_entry(current)
    if current.spacegroup.it_number == 2:
        # one entry: the P-1 candidates are intrinsic to the lattice (_p1_candidates)
        transform = _proxy_niggli_transform(current.cell.metric())
        if transform.det() < 0:
            transform = -transform
        current = _apply_action(current, AffineOperation(transform.T().inv(), (0, 0, 0)))
    return _canonical_protostructure_geometry_entry(current, preserve_chirality=preserve_chirality, names=names)


def _metric_keeps_group(structure: ASUStructure) -> bool:
    """Return whether every operation of the group is an exact isometry of the cell metric."""
    metric = structure.cell.metric()
    for operation in structure.spacegroup.symmetry_operations:
        matrix = SurdVector(structure.transform.symop_to_setting(operation).matrix)
        if matrix.T() * metric * matrix != metric:
            return False
    return True


def _canonical_protostructure_assignments_asu(
    structure: ASUStructure,
    *,
    preserve_chirality: bool = True,
    named: bool = False,
) -> tuple[ASUStructure, ...]:
    """Return all tied chemical assignments of one recognized exact ASU.

    With ``named`` the species class names enter the discrete key, before any geometry, as in the bare
    canonicalizer; otherwise chemistry is chosen only after the anonymous geometry. Without
    ``preserve_chirality``, in a Sohncke group that is its own enantiomorphic partner, the
    enantiomer is canonicalized too, with its own frame (whose proper-only automorphisms cannot
    relate the hands), and both hands' candidates compete. So do both hands of any other group
    whose improper operations are not exact isometries of the cell (:func:`_metric_keeps_group`):
    a recognized ASU keeps its measured cell, whose metric then cannot identify the hands. The
    test is exact. A cell built from parameters keeps the right angles and 120° angles its group
    fixes exactly, and an approximated rhombohedral-axes cell is built in its symmetric orientation
    with exactly equal edges and angles, so among the repository fixtures it fires only for
    cod/1001844 (P1 as stored, recognized in a higher group), where the mirrored hand's
    representative wins.
    """
    if not isinstance(structure, ASUStructure):
        raise TypeError(f"expected ASUStructure, got {type(structure).__name__}")
    _validate_exact_input(structure)
    hands = [structure]
    group = structure.spacegroup
    if not preserve_chirality and (
        _own_enantiomorph_sohncke(group) or (not _sohncke(group) and not _metric_keeps_group(structure))
    ):
        hands.append(_mirrored(structure))
    candidates = []
    for hand in hands:
        candidates.extend(_frame_candidates(hand, preserve_chirality=preserve_chirality, named=named))
    best_geometry = min(key for _canonical, _assignment, key in candidates)
    restored = []
    for canonical, assignment, key in candidates:
        _checkpoint()
        if key != best_geometry:
            continue
        result = _restore_assignment(canonical, assignment, structure)
        restored.append(_derived_precision(result, structure))
    return _deduplicate_results(restored)


def _class_name_or_own(species: Species) -> str:
    """Return a species' class name, or its own name when it has no class (only ``vacancy``)."""
    try:
        return _species_class(species).name
    except ValueError:
        return species.name


def _frame_candidates(
    structure: ASUStructure, *, preserve_chirality: bool, named: bool
) -> list[tuple[ASUStructure, tuple[Species, ...], tuple[Any, ...]]]:
    """Canonicalize every tied frame assignment of one hand, keyed for the choice between them."""
    bonded = structure if structure.spacegroup.it_number == 1 and structure.bonds is not None else None
    frame = _anonymous_p1_frame(UnitcellStructureView(structure), bonded)
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
        # the discrete key names species classes, so it is the bare canonicalizer's assigned key
        names = (
            {
                species.name: _class_name_or_own(original)
                for species, original in zip(anonymous_species, assignment, strict=True)
            }
            if named
            else None
        )
        canonical = _canonical_protostructure_geometry(anonymous, preserve_chirality=preserve_chirality, names=names)
        discrete = _discrete_key(canonical, AffineOperation.identity(), names) if named else ()
        candidates.append((canonical, assignment, (discrete, _anonymous_geometry_key(canonical))))
    return candidates


def _canonical_protostructure_asu(
    structure: ASUStructure,
    *,
    preserve_chirality: bool = True,
    named: bool = True,
) -> ASUStructure:
    """Canonicalize one recognized ASU by its bare assigned key, species class names included, then geometry.

    The discrete tier is then the bare canonicalizer's, so the refined and bare labels agree as
    strings, except where the refined terminal collapses an exact supercell (P-1 included) or
    demotes a site declared on a more general letter than its point. Without ``named`` (template class names) chemistry is chosen only after geometry.
    """
    return min(
        _canonical_protostructure_assignments_asu(structure, preserve_chirality=preserve_chirality, named=named),
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
    upward pseudosymmetry. The family keeps the anonymous convention: the geometry is fixed with
    synthetic class names and the species are assigned after it. :func:`canonical_asu_protostructure`
    chooses the species by the bare assigned key instead, so it need not be a member. As for the
    labels, the structure of species classes is recognized: every used species is replaced by its
    class first, and the results carry the class species.

    :param structure: The measured structure to recognize.
    :param tolerance: Base Cartesian recognition tolerance, or ``None`` to derive it.
    :param factors: Multipliers for the recognition tolerance sweep.
    :param lift: Whether to search upward for pseudosymmetry after recognition.
    :param preserve_chirality: Whether to keep the recognized enantiomorphic group.
    :return: The tied protostructure-first canonical asymmetric units.
    :raises ValueError: If the structure is unsupported or no tolerance member fits.
    """
    return _recognized_canonical_asus(
        _class_structure(structure),
        tolerance=tolerance,
        factors=factors,
        lift=lift,
        preserve_chirality=preserve_chirality,
        named=False,
        bonded=_bonded_domain(structure),
    )


def _bonded_domain(structure: Any) -> FundamentalDomainStructure | None:
    """Return the fundamental domain carrying a structure's bond network, else ``None``.

    A fundamental domain with a network is itself; a three-dimensional unit-cell presentation with
    a network is the P1 domain of its sites. The canonical frames rename species, so a unit cell
    with coincident sites cannot carry its network through them: it is dropped (and logged).
    """
    if isinstance(structure, FundamentalDomainStructure):
        return structure if structure.bonds is not None else None
    view = UnitcellStructureView(structure)
    if view.bonds is None or view.cell.periodicity != (True, True, True):
        return None
    domain = _cell_domain(view)
    if any(len(entries) > 1 for entries in _orbit_index(domain).values()):
        _dropped("coincident atoms: the canonical frames rename species")
        return None
    return domain


def _recognized_canonical_asus(
    structure: StructureLike,
    *,
    tolerance: float | None,
    factors: tuple[Fraction | float | int, ...],
    lift: bool,
    preserve_chirality: bool,
    named: bool,
    bonded: FundamentalDomainStructure | None = None,
) -> tuple[ASUStructure, ...]:
    """Recognize ``structure``; return the anonymous assignment family, or with ``named`` the one named result.

    ``bonded`` is a fundamental domain with a bond network whose own-setting coordinates are those of
    ``structure``. Its network is carried through the anonymous frame and recognition (atoms matched to
    the nearest recognized atoms) to the results; it is dropped if the recognized symmetry does not keep it.
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
    outer_frame = _anonymous_p1_frame(source_view, bonded)
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
    if normalized_p1.bonds is not None:
        try:
            network = _matched(normalized_p1, winner, winner.transform.operation.inverse())
        except (_AmbiguousMatch, _ShapeConflict, _CoincidentAtoms) as error:
            _dropped(f"recognition: {error}")
            network = None
        else:
            if network is None:
                _dropped("recognition: the bond set is not invariant under the recognized symmetry")
        winner = winner._with_bonds(network)
    if lift:
        from httk.atomistic.symmetry.lift import canonicalize_legacy

        winner = canonicalize_legacy(winner, tolerance=base, preserve_chirality=preserve_chirality).asu
        preserve_chirality = True
    outer_by_name = {anonymous.name: index for index, anonymous in enumerate(outer_frame.structure.species)}
    if named:
        # every tied frame assignment names an equivalent description of the same crystal
        assignment = tuple(outer_frame.assignments[0][outer_by_name[species.name]] for species in winner.species)
        named_winner = _restore_assignment(winner, assignment, source_view)
        result = _canonical_protostructure_asu(named_winner, preserve_chirality=preserve_chirality)
        return (_derived_precision(result, source_view),)
    inner = _canonical_protostructure_assignments_asu(winner, preserve_chirality=preserve_chirality)
    restored = []
    for candidate in inner:
        _checkpoint()
        for outer_assignment in outer_frame.assignments:
            _checkpoint()
            assignment = tuple(outer_assignment[outer_by_name[species.name]] for species in candidate.species)
            result = _restore_assignment(candidate, assignment, source_view)
            restored.append(_derived_precision(result, source_view))
    return _deduplicate_results(restored)


def canonical_asu_protostructure(
    structure: StructureLike,
    *,
    tolerance: float | None = None,
    factors: tuple[Fraction | float | int, ...] = (Fraction(1, 5), 1, 5),
    lift: bool = False,
    preserve_chirality: bool = True,
) -> ASUStructure:
    """Return the canonical ASU whose species assignment minimises the bare assigned key.

    The species class names enter the discrete key before any geometry, as in
    :func:`~httk.atomistic.symmetry.canonical_classification.canonical_bare_protostructure`,
    so the result need not be a member of :func:`canonical_asu_protostructure_assignments`. As for
    the labels, the structure of species classes is recognized: every used species is replaced by
    its class first, so the result carries the class species and its bare label is the canonical
    bare protostructure label. :func:`~httk.atomistic.symmetry.canonical.canonicalize` instead
    keeps the source species and recognizes their own symmetry.

    :param structure: The measured structure to recognize.
    :param tolerance: Base Cartesian recognition tolerance, or ``None`` to derive it.
    :param factors: Multipliers for the recognition tolerance sweep.
    :param lift: Whether to search upward for pseudosymmetry after recognition.
    :param preserve_chirality: Whether to keep the recognized enantiomorphic group.
    :return: One protostructure-first canonical asymmetric unit.
    :raises ValueError: If the structure is unsupported or no tolerance member fits.
    """
    return _recognized_canonical_asus(
        _class_structure(structure),
        tolerance=tolerance,
        factors=factors,
        lift=lift,
        preserve_chirality=preserve_chirality,
        named=True,
        bonded=_bonded_domain(structure),
    )[0]
