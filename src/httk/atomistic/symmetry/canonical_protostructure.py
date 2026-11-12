"""Anonymous protostructure-first canonicalization of measured crystal structures.

This convention chooses the anonymous Wyckoff occupation pattern, then its species assignment,
and only then the exact geometry. Its finite affine search uses the same tabulated normalizer scope
as the legacy canonicalizer, taken in groups 2 to 15 also after the bare canonicalizers' kept cell
changes, and does not enumerate an infinite affine normalizer. In groups 3 to 15 the cell is then
reduced within the cell changes that keep every letter, so it does not depend on the input cell.
Public measured structure entry points optionally perform the established upward pseudosymmetry
search first.
"""

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import product
from typing import Any

from httk.core import FracVector, SurdVector

from httk.atomistic import data
from httk.atomistic.models.cell.cell import Cell
from httk.atomistic.models.prototype.notation import class_order_key
from httk.atomistic.models.species.classes import _class_structure, _species_class
from httk.atomistic.models.species.species import Species
from httk.atomistic.models.structure.asu import ASUStructure, FundamentalDomainStructure, WyckoffSite
from httk.atomistic.models.structure.comparison import same_crystal
from httk.atomistic.models.structure.like import StructureLike
from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
from httk.atomistic.models.structuretype.fundamental import FundamentalDomainTemplate
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
    _ext_gcd,
    _isomorphic_reduced_entry,
    _metric_automorphism_operations,
    _niggli_reduced_entry,
    _p1_metric_automorphism_operations,
    _primitive_reduced_entry,
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


def _discrete_candidates(
    structure: ASUStructure, names: dict[str, str] | None = None, *, proper: bool = False
) -> tuple[_Candidate, ...]:
    """Return every right-handed description with the least discrete key, its cell reduced within its class.

    In groups 2 to 15 the tabulated descriptions are also taken after each kept cell change
    ``sigma`` of the bare canonicalizers, so the least discrete key is the canonical bare one.
    In groups 3 to 15 every ``sigma`` reaching it contributes, and each description is moved by
    the cell changes ``K`` that keep every letter (:func:`_kernel_residues`) to its least
    ``(a·a, c·c, a·c)`` (:func:`_ac_reduced`), the reduction key; the cell does not depend on
    the input cell. In P-1 the earliest ``sigma`` is kept: the entry set is the Niggli cell with
    every proper metric automorphism, so that choice is already intrinsic. The structure must be
    right-handed (:func:`_right_handed`); left-handed descriptions are skipped, since an improper
    group operation gives each a right-handed twin with the same keys. With ``proper`` (a Sohncke
    group) only proper descriptions compete, so the key is the hand-preserving bare one.
    """
    from httk.atomistic.symmetry.canonical_classification import _cell_changes, _letter_maps  # imports this module

    identity_matrix = FracVector.eye((3, 3))
    point_operations = tuple((operation, operation.matrix.T().inv()) for operation in _point_operations(structure))
    metric_keys: dict[FracVector, tuple[Any, ...]] = {}
    source_metric = structure.cell.metric()
    group = structure.spacegroup
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
    if residues is None:
        # P-1: the entry set is the Niggli(-proxy) cell with every tied proper metric automorphism,
        # closed under the input cell, so the earliest sigma of the fixed table that reaches the key
        # is already intrinsic. The stored cell is sigma times the Niggli cell: intrinsic, but not
        # reduced within its class. Every other group outside 3 to 15 has only the identity sigma.
        sigmas = (sigmas[sigma_keys.index(best_key)],)
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
        _basis_key(represented.cell.basis),
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
    structure keeps only its cell, sites, species, coordinate precision and charge; sites lose
    retained representatives.
    """
    if isinstance(domain, ASUStructure) and not _left_handed(domain):
        return domain
    structure = _standard_input(
        ASUStructure(
            domain.cell,
            domain.spacegroup,
            domain.wyckoff_sites,
            domain.species,
            transform=domain.transform,
            coordinate_precision=domain.coordinate_precision,
            charge=domain.charge,
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
        return ASUStructure(
            Cell(-structure.cell.basis, precision=structure.cell.precision, periodicity=structure.cell.periodicity),
            target,
            sites,
            structure.species,
            transform=SettingTransform.identity(),
            coordinate_precision=structure.coordinate_precision,
            charge=structure.charge,
        )
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


def _terminal_result_key(structure: ASUStructure, names: dict[str, str] | None = None) -> tuple[Any, ...]:
    """Order tied lattice entries (and both hands) through the terminal's discrete and geometric tiers."""
    metric = structure.cell.metric()
    reduction = (
        (metric._element((0, 0)), metric._element((2, 2)), metric._element((0, 2)))
        if 3 <= structure.spacegroup.it_number <= 15
        else ()
    )
    return (
        _discrete_key(structure, AffineOperation.identity(), names),
        reduction,
        tuple(metric._element((row, column)) for row in range(3) for column in range(3)),
        _site_key(structure),
        _basis_key(structure.cell.basis),
    )


def _own_enantiomorph_sohncke(spacegroup: Spacegroup) -> bool:
    """Return whether a group has only proper operations and is its own enantiomorphic partner.

    The enantiomer of :func:`_mirrored` keeps the group, so it applies whether or not the
    inversion through the origin keeps the group (I4₁, I4₁22 and F4₁32 need an origin shift).
    """
    return spacegroup.record["it_number_enantiomorphic"] is None and _sohncke(spacegroup)


def _mirrored(structure: ASUStructure) -> ASUStructure:
    """Return the enantiomer: the same fractional coordinates and group in the negated (left-handed) basis."""
    return ASUStructure(
        Cell(-structure.cell.basis, precision=structure.cell.precision, periodicity=structure.cell.periodicity),
        structure.spacegroup,
        structure.wyckoff_sites,
        structure.species,
        transform=structure.transform,
        coordinate_precision=structure.coordinate_precision,
        charge=structure.charge,
    )


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
    else:
        current = _isomorphic_reduced_entry(current)
    entries = _triclinic_proxy_entries(current) if current.spacegroup.it_number == 2 else (current,)
    return min(
        (
            _canonical_protostructure_geometry_entry(
                entry,
                preserve_chirality=preserve_chirality,
                names=names,
            )
            for entry in entries
        ),
        key=lambda result: _terminal_result_key(result, names),
    )


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
    relate the hands), and both hands' candidates compete.
    """
    if not isinstance(structure, ASUStructure):
        raise TypeError(f"expected ASUStructure, got {type(structure).__name__}")
    _validate_exact_input(structure)
    hands = [structure]
    if not preserve_chirality and _own_enantiomorph_sohncke(structure.spacegroup):
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
        restored.append(_restore_unchanged_precision(result, structure))
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
    strings. Without ``named`` (template class names) chemistry is chosen only after geometry.
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
    )


def _recognized_canonical_asus(
    structure: StructureLike,
    *,
    tolerance: float | None,
    factors: tuple[Fraction | float | int, ...],
    lift: bool,
    preserve_chirality: bool,
    named: bool,
) -> tuple[ASUStructure, ...]:
    """Recognize ``structure``; return the anonymous assignment family, or with ``named`` the one named result."""
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
    outer_by_name = {anonymous.name: index for index, anonymous in enumerate(outer_frame.structure.species)}
    if named:
        # every tied frame assignment names an equivalent description of the same crystal
        assignment = tuple(outer_frame.assignments[0][outer_by_name[species.name]] for species in winner.species)
        named_winner = _restore_assignment(winner, assignment, source_view)
        result = _canonical_protostructure_asu(named_winner, preserve_chirality=preserve_chirality)
        return (_restore_unchanged_precision(result, source_view),)
    inner = _canonical_protostructure_assignments_asu(winner, preserve_chirality=preserve_chirality)
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
    )[0]
