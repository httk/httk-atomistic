"""Exact P1 lifts: trigonal/hexagonal metric, the lattice modular solver, and P1 canonicalization.

Phase 1 covers the exact lattice modular solve, the enforced hexagonal-axes metric for
trigonal/hexagonal parents, and the per-parent branch-cap guard.  Phase 2 adds the
normalizer-canonical state normal form (Wyckoff demotion plus the continuous-translation and
affine-normalizer-coset quotients) that makes :func:`canonicalize` terminate from a raw P1 input.
"""

import itertools
import random
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from typing import Any

import pytest
from httk.core import FracVector, SurdVector, load

import httk.atomistic.symmetry.lift as lift_module
from httk.atomistic import (
    ASUStructure,
    Cell,
    Spacegroup,
    Species,
    UnitcellStructureView,
    WyckoffSite,
    backward_lift,
    canonical_asu,
    canonicalize,
    data,
    highest_symmetry,
    same_crystal,
    subgroup_representation,
)
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.symmetry.lift import (
    _canonical_without_bfs,
    _cell_for_transform,
    _integer_options,
    _linear_solve,
    _normal_form,
    _rational_null_space,
    _site_key,
    _structure_signature,
)
from httk.atomistic.symmetry.recognition import structure_tolerance
from httk.atomistic.symmetry.subgroups import _standard_input, subgroup_transforms


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _p1(cell: Cell, sites: list[WyckoffSite]) -> ASUStructure:
    return ASUStructure(cell, 1, sites, _species(*sorted({site.species for site in sites})))


_CSCL_CELL = Cell(((4, 0, 0), (0, 4, 0), (0, 0, 4)))


def _cscl_sites() -> list[WyckoffSite]:
    return [
        WyckoffSite("a", FracVector((0, 0, 0)), "Cs"),
        WyckoffSite("a", FracVector((F(1, 2), F(1, 2), F(1, 2))), "Cl"),
    ]


def _cscl(sites: list[WyckoffSite]) -> ASUStructure:
    return ASUStructure(_CSCL_CELL, 1, sites, _species("Cs", "Cl"))


def _result_key(result: Any) -> tuple[Any, ...]:
    return (result.spacegroup.it_number, _site_key(result.asu), result.asu.cell.basis)


def _wrapped_shift(values: list[F], shift: tuple[F, F, F]) -> FracVector:
    return FracVector(tuple((value + delta) % 1 for value, delta in zip(values, shift)))


_CYCLIC = FracVector(((0, 1, 0), (0, 0, 1), (1, 0, 0)))
_SHEAR = FracVector(((1, 1, 0), (0, 1, 0), (0, 0, 1)))


def _rebased(cell_rows: object, sites: list[WyckoffSite], species: list[str], transform: FracVector) -> ASUStructure:
    """Return the same crystal in a rebased P1 cell: ``basis_new = M * basis``, ``f_new = f_old * M^-1``."""
    basis = SurdVector(transform) * Cell(cell_rows).basis
    inverse = transform.inv()
    rebased_sites = [
        WyckoffSite(site.wyckoff, (FracVector(site.free_params.to_fractions()) * inverse).normalize(), site.species)
        for site in sites
    ]
    return ASUStructure(Cell(basis), 1, rebased_sites, _species(*species))


def _invariance_variants(cell_rows: object, sites: list[WyckoffSite], species: list[str]) -> dict[str, ASUStructure]:
    """The six same-crystal P1 descriptions that must canonicalize identically."""

    def shifted(shift: tuple[F, F, F]) -> ASUStructure:
        return ASUStructure(
            Cell(cell_rows),
            1,
            [WyckoffSite(s.wyckoff, _wrapped_shift(s.free_params.to_fractions(), shift), s.species) for s in sites],
            _species(*species),
        )

    return {
        "base": ASUStructure(Cell(cell_rows), 1, sites, _species(*species)),
        "shift_a": shifted((F(1, 5), F(1, 7), F(1, 3))),
        "shift_b": shifted((F(1, 2), F(0), F(0))),
        "reversed": ASUStructure(Cell(cell_rows), 1, list(reversed(sites)), _species(*species)),
        "axis_relabel": _rebased(cell_rows, sites, species, _CYCLIC),
        "shear": _rebased(cell_rows, sites, species, _SHEAR),
    }


# --- 1a: trigonal/hexagonal metric enforcement -----------------------------------------------


def test_hexagonal_child_lifts_into_trigonal_parent_unchanged() -> None:
    parent = ASUStructure(
        Cell(CellParams((5, 5, 12, 90, 90, 120)).basis),
        166,
        [WyckoffSite("a", FracVector(()), "Bi")],
        _species("Bi"),
    )
    child = subgroup_representation(parent, 148).asu
    results = backward_lift(child, 166, tolerance=1e-3)
    assert any(result.residual == F(0) and same_crystal(result.asu, parent) for result in results)
    # The parent metric (a=b, alpha=beta=90, gamma=120) is reproduced exactly, not merely accepted.
    assert any(result.asu.cell.basis == parent.cell.basis for result in results)


def test_cubic_cell_is_rejected_before_the_solver_for_a_trigonal_parent() -> None:
    transform = subgroup_transforms(143, 1)[0]
    assert transform.parent.crystal_system == "trigonal"
    cubic = _p1(Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))), [WyckoffSite("a", FracVector((0, 0, 0)), "X")])
    assert _cell_for_transform(cubic, transform, 1e-3) is None
    hexagonal = _p1(Cell(CellParams((5, 5, 12, 90, 90, 120)).basis), [WyckoffSite("a", FracVector((0, 0, 0)), "X")])
    accepted = _cell_for_transform(hexagonal, transform, 1e-3)
    assert accepted is not None and accepted.cartesian_deviation == 0.0
    assert accepted.cell.basis == hexagonal.cell.basis


# --- 1b: exact lattice-based modular solve equals the old product enumeration -----------------


def _reference_exact_solution(
    matrix: tuple[tuple[F, ...], ...],
    constants: tuple[F, ...],
    options: tuple[range, ...],
) -> tuple[F, ...] | None:
    """Reference: the original full-product enumeration, returning the first exact boxed solution."""
    for integers in itertools.product(*(tuple(option) for option in options)):
        rhs = tuple(F(integer) - constant for integer, constant in zip(integers, constants))
        solution = _linear_solve(matrix, rhs)
        if solution is not None:
            return solution
    return None


def test_lattice_modular_solver_matches_brute_force_on_real_hops(monkeypatch: pytest.MonkeyPatch) -> None:
    real_solver = lift_module._solve_modular
    captured: list[tuple] = []

    def spy(equations: tuple) -> object:
        captured.append(equations)
        return real_solver(equations)

    monkeypatch.setattr(lift_module, "_solve_modular", spy)
    probe = _p1(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        [WyckoffSite("a", FracVector((F(1, 7), F(2, 11), F(3, 13))), "Po")],
    )
    for parent in (2, 3, 5):
        backward_lift(probe, parent, tolerance=1e-3)
    monkeypatch.undo()
    assert captured

    compared = underdetermined = infeasible = 0
    for equations in captured:
        matrix = tuple(row for equation in equations for row in equation.matrix)
        constants = tuple(value for equation in equations for value in equation.constant)
        if not matrix:
            continue
        options = tuple(_integer_options(row, constant) for row, constant in zip(matrix, constants))
        product = 1
        for option in options:
            product *= len(option)
        if product == 0 or product > 4000:
            continue
        compared += 1
        reference = _reference_exact_solution(matrix, constants, options)
        result = real_solver(equations)
        found = result[0] if (result is not None and result[2]) else None
        assert found == reference
        if reference is None:
            infeasible += 1
        if _rational_null_space(matrix, len(matrix[0])):
            underdetermined += 1
    assert compared
    # P1's three continuous-normalizer shift columns make the real systems underdetermined, and the
    # candidate search assembles systems with no exact boxed wrap; both branches must be exercised.
    assert underdetermined
    assert infeasible


def test_lattice_modular_solver_handles_zero_unknown_and_infeasible_systems() -> None:
    # Zero unknowns, integer constant: the wrap is forced and the exact solution is empty.
    exact = lift_module._solve_modular((lift_module._Equation(((),), (F(-2),)),))
    assert exact == ((), F(0), True)
    # Zero unknowns, non-integer constant: no integer wrap exists, so there is no exact solution.
    inexact = lift_module._solve_modular((lift_module._Equation(((),), (F(1, 2),)),))
    assert inexact is None or inexact[2] is False


# --- 1c: P1 one-hop lifts terminate cleanly (root cause A fixed from P1) ----------------------


def test_p1_one_hop_lifts_never_hit_the_solver_cap() -> None:
    po = _p1(Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))), [WyckoffSite("a", FracVector((0, 0, 0)), "Po")])
    # Every minimal supergroup of SG 1 must complete without the modular solver's branch cap; a
    # single P1 atom at the origin has exact lifts into the compatible triclinic/monoclinic parents.
    exact_targets = 0
    for parent in (2, 3, 4, 5, 6, 7, 8, 9, 143, 144, 145, 146):
        results = backward_lift(po, parent, tolerance=1e-3)
        if any(result.residual == F(0) for result in results):
            exact_targets += 1
    assert exact_targets


def test_p1_bfs_guard_skips_a_capped_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    po = _p1(Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))), [WyckoffSite("a", FracVector((0, 0, 0)), "Po")])
    real_raw = lift_module._raw_lifts

    def raw(structure: ASUStructure, target: Spacegroup, tolerance: float) -> object:
        if target.it_number == 3:
            raise ValueError("exact modular lift solver branch cap exceeded for 3 -> 1")
        return real_raw(structure, target, tolerance)

    monkeypatch.setattr(lift_module, "_raw_lifts", raw)
    # A branch-cap failure on one parent target is reported and skipped, not propagated; the search
    # still returns the lifts assembled from the other minimal supergroups.
    lifts = lift_module._highest_lifts(po, 1e-3)
    assert lifts
    assert all(result.spacegroup.it_number != 3 for result in lifts)


# --- phase 2: normal form (Wyckoff demotion + normalizer quotients) ---------------------------


def test_demote_sites_relabels_a_general_site_on_a_special_coordinate() -> None:
    cell = Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5)))
    # SG 2 general position "i" placed on the special coordinate (0,0,1/2) demotes to letter "b";
    # (1/2,1/2,1/2) demotes to "h" (verified against the vendored SG 2 Wyckoff table).  Tested on
    # ``_demote_sites`` directly, since the full normal form then collapses every SG-2 special
    # position to "a" through the discrete-translation quotient.
    at_b = ASUStructure(cell, 2, [WyckoffSite("i", FracVector((0, 0, F(1, 2))), "Po")], _species("Po"))
    assert _site_key(lift_module._demote_sites(at_b)) == (("Po", "b", ()),)
    at_h = ASUStructure(cell, 2, [WyckoffSite("i", FracVector((F(1, 2), F(1, 2), F(1, 2))), "Po")], _species("Po"))
    assert _site_key(lift_module._demote_sites(at_h)) == (("Po", "h", ()),)
    # A genuine general position (non-degenerate orbit) keeps its letter.
    general = ASUStructure(
        Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        2,
        [WyckoffSite("i", FracVector((F(1, 7), F(2, 11), F(3, 13))), "Po")],
        _species("Po"),
    )
    assert lift_module._demote_sites(general).wyckoff_sites[0].wyckoff == "i"


def _bfs_state_multiset_counts(structure: ASUStructure) -> Counter:
    """Walk the normal-form breadth-first search and count states per (IT number, Wyckoff multiset)."""
    current = _normal_form(_standard_input(structure))
    tolerance = structure_tolerance(current)
    queue = [current]
    visited = {(current.spacegroup.it_number, _structure_signature(current))}
    counts: Counter = Counter()
    while queue:
        state = queue.pop(0)
        for result in lift_module._highest_lifts(state, tolerance):
            image = _normal_form(result.asu)
            key = (image.spacegroup.it_number, _structure_signature(image))
            if key in visited:
                continue
            visited.add(key)
            queue.append(image)
            multiset = tuple(sorted(Counter((s.species, s.wyckoff) for s in image.wyckoff_sites).items()))
            counts[(image.spacegroup.it_number, multiset)] += 1
    return counts


@pytest.mark.extended
def test_p1_single_atom_breakdown_has_no_wyckoff_duplicates() -> None:
    po = _p1(Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))), [WyckoffSite("a", FracVector((0, 0, 0)), "Po")])
    counts = _bfs_state_multiset_counts(po)
    # Demotion removes the two IT-2 duplicates; every (IT, Wyckoff multiset) is now unique.
    assert counts and max(counts.values()) == 1


# --- phase 2: end-to-end P1 canonicalization (extended: full-depth BFS) ------------------------


@pytest.mark.extended
def test_p1_single_atom_canonicalizes_to_cubic() -> None:
    po = _p1(Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))), [WyckoffSite("a", FracVector((0, 0, 0)), "Po")])
    result = canonicalize(po, tolerance=1e-3)
    assert result.spacegroup.it_number == 221
    assert _site_key(result.asu) == (("Po", "a", ()),)


@pytest.mark.extended
def test_p1_cscl_canonicalizes_to_cubic() -> None:
    result = canonicalize(_cscl(_cscl_sites()), tolerance=1e-3)
    assert result.spacegroup.it_number == 221
    # The deterministic min-key canonical origin places the alphabetically-smaller species (Cl) on
    # 1a; "Cs on 1a" is the same crystal at the (1/2,1/2,1/2)-shifted origin.
    assert _site_key(result.asu) == (("Cl", "a", ()), ("Cs", "b", ()))
    # Idempotency: canonicalizing the canonical result reproduces it exactly (F4 -- the orientation
    # rebuild must not perturb the exact metric).
    assert _result_key(canonicalize(result.asu, tolerance=1e-3)) == _result_key(result)


@pytest.mark.extended
def test_p1_cscl_is_origin_site_order_and_run_invariant() -> None:
    base = _cscl_sites()
    reference = _result_key(canonicalize(_cscl(base), tolerance=1e-3))
    # Determinism: a second run is identical.
    assert _result_key(canonicalize(_cscl(base), tolerance=1e-3)) == reference
    # Site-order invariance: reversed site list.
    assert _result_key(canonicalize(_cscl(list(reversed(base))), tolerance=1e-3)) == reference
    # Origin invariance: every coordinate rigidly wrapped-shifted.
    for shift in ((F(1, 5), F(1, 7), F(1, 3)), (F(1, 2), F(0), F(0))):
        shifted = _cscl(
            [WyckoffSite(s.wyckoff, _wrapped_shift(s.free_params.to_fractions(), shift), s.species) for s in base]
        )
        assert _result_key(canonicalize(shifted, tolerance=1e-3)) == reference


@pytest.mark.extended
def test_p1_and_direct_cubic_entries_are_fully_coherent() -> None:
    from_p1 = canonicalize(_cscl(_cscl_sites()), tolerance=1e-3)
    direct = canonicalize(
        ASUStructure(
            _CSCL_CELL,
            221,
            [WyckoffSite("a", FracVector(()), "Cs"), WyckoffSite("b", FracVector(()), "Cl")],
            _species("Cs", "Cl"),
        ),
        tolerance=1e-3,
    )
    # The discrete Euclidean-normalizer translation (1/2,1/2,1/2) of Pm-3m relates the two origin
    # choices, so the P1 entry and the directly-built 221 entry now agree exactly.
    assert _result_key(from_p1) == _result_key(direct)
    assert from_p1.spacegroup.it_number == 221
    assert _site_key(from_p1.asu) == (("Cl", "a", ()), ("Cs", "b", ()))


# --- phase 3: discrete-normalizer translations and full basis/origin invariance ----------------


def _continuous_directions(spacegroup: Spacegroup) -> list[tuple[F, ...]]:
    """The zero-diagonal columns of the integer diagonalization: the continuous normalizer axes."""
    seen: set[tuple[tuple[int, ...], ...]] = set()
    rows: list[tuple[int, ...]] = []
    for operation in spacegroup.symmetry_operations:
        linear = tuple(tuple(int(v) for v in row) for row in operation.matrix.to_fractions())
        if linear in seen:
            continue
        seen.add(linear)
        for i in range(3):
            rows.append(tuple((1 if i == j else 0) - linear[i][j] for j in range(3)))
    diagonal, transform = lift_module._integer_diagonalize(tuple(rows))
    return [tuple(F(transform[r][c]) for r in range(3)) for c in range(3) if diagonal[c] == 0]


def _rank(vectors: list[tuple[F, ...]]) -> int:
    return 3 - len(_rational_null_space(tuple(vectors), 3)) if vectors else 0


def test_discrete_and_continuous_normalizer_translations() -> None:
    zero = (F(0), F(0), F(0))
    body = (F(1, 2), F(1, 2), F(1, 2))
    # P-1: the eight half-translations relating the inversion centres a-h.
    eight = lift_module._discrete_normalizer_translations(Spacegroup.standard(2))
    assert len(eight) == 8 and zero in eight and body in eight
    # Pm-3m: identity and the body-centring translation (fixes the CsCl cross-entry divergence).
    assert set(lift_module._discrete_normalizer_translations(Spacegroup.standard(221))) == {zero, body}
    # Computed continuous directions span the same subspace as the vendored continuous-normalizer basis.
    for number in (1, 3, 5, 25, 99):
        spacegroup = Spacegroup.standard(number)
        computed = _continuous_directions(spacegroup)
        vendored = [
            tuple(F(value) for value in vector)
            for vector in data.spacegroup_subgroup_record(number)["continuous_normalizer"]["basis_vectors"]
        ]
        assert _rank(computed) == _rank(vendored) == _rank(computed + vendored)


# --- phase 3 review fixes: chirality, orbit representative, centred lattices, exact orientation ---


def _signed_volume(structure: ASUStructure) -> float:
    """Min-image signed volume of the inter-species vectors (its sign is the crystal's chirality)."""
    ordered = sorted(structure.wyckoff_sites, key=lambda site: site.species)
    points = [
        FracVector(structure.spacegroup.wyckoff_position(site.wyckoff).coordinates(site.free_params)[0]).normalize()
        for site in ordered
    ]
    basis = structure.cell.basis

    def cartesian(fractional: FracVector) -> list[float]:
        return (SurdVector(fractional) * basis).to_floats()

    def minimum_image(vector: FracVector) -> FracVector:
        return FracVector([value - round(float(value)) for value in vector.to_fractions()])

    a, b, c = (cartesian(minimum_image(points[index] - points[0])) for index in (1, 2, 3))
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])


def _chiral_sites() -> list[WyckoffSite]:
    return [
        WyckoffSite("a", FracVector((F(1, 7), F(2, 11), F(3, 13))), "Fe"),
        WyckoffSite("a", FracVector((F(5, 7), F(3, 11), F(1, 13))), "Co"),
        WyckoffSite("a", FracVector((F(2, 7), F(9, 11), F(5, 13))), "Ni"),
        WyckoffSite("a", FracVector((F(4, 7), F(6, 11), F(11, 13))), "Cu"),
    ]


def test_discrete_translations_cover_centred_lattices() -> None:
    quarter = (F(1, 4), F(1, 4), F(1, 4))
    # F-43m gains the (1/4,1/4,1/4) translation relating its 4a/4b/4c/4d origins (the Z^3 criterion
    # drops it).
    assert quarter in lift_module._discrete_normalizer_translations(Spacegroup.standard(216))
    # All seven centred-lattice groups gain the quarter-translations the too-strict criterion missed.
    for number in (22, 82, 119, 120, 196, 216, 219):
        translations = lift_module._discrete_normalizer_translations(Spacegroup.standard(number))
        assert any(value.denominator == 4 for vector in translations for value in vector), number


@pytest.mark.extended
def test_zincblende_entries_are_coherent() -> None:
    cell = Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5)))
    zn_ac = ASUStructure(
        cell, 216, [WyckoffSite("a", FracVector(()), "Zn"), WyckoffSite("c", FracVector(()), "S")], _species("Zn", "S")
    )
    zn_cb = ASUStructure(
        cell, 216, [WyckoffSite("c", FracVector(()), "Zn"), WyckoffSite("b", FracVector(()), "S")], _species("Zn", "S")
    )
    # (1/4,1/4,1/4) relates the two origin descriptions of the same zincblende crystal.
    assert _result_key(canonicalize(zn_ac, tolerance=1e-3)) == _result_key(canonicalize(zn_cb, tolerance=1e-3))


def test_normal_form_is_orbit_representative_invariant() -> None:
    cell = Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7)))
    at_x = ASUStructure(cell, 2, [WyckoffSite("i", FracVector((F(6, 7), F(9, 11), F(10, 13))), "Si")], _species("Si"))
    at_minus_x = ASUStructure(
        cell, 2, [WyckoffSite("i", FracVector((F(1, 7), F(2, 11), F(3, 13))), "Si")], _species("Si")
    )
    first = lift_module._normal_form(at_x)
    second = lift_module._normal_form(at_minus_x)
    # x and -x are the same orbit under inversion: identical key and identical stored free params.
    assert _site_key(first) == _site_key(second)
    assert first.wyckoff_sites[0].free_params.to_fractions() == second.wyckoff_sites[0].free_params.to_fractions()


def test_canonical_orientation_preserves_chirality() -> None:
    structure = ASUStructure(
        Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7))), 1, _chiral_sites(), _species("Fe", "Co", "Ni", "Cu")
    )
    canonical = _canonical_without_bfs(structure)
    # The canonical representative is right-handed and keeps the input's chirality: no enantiomorph.
    assert canonical.cell.basis.det().sign() == 1
    assert (_signed_volume(structure) > 0) == (_signed_volume(canonical) > 0)
    # A det<0 basis is still repaired to a right-handed one with identical Cartesian geometry.
    left_handed = ASUStructure(
        Cell(((-5, 0, 0), (0, 6, 0), (0, 0, 7))), 1, _chiral_sites(), _species("Fe", "Co", "Ni", "Cu")
    )
    repaired = lift_module._canonical_orientation(left_handed)
    assert repaired.cell.basis.det().sign() == 1
    assert (_signed_volume(left_handed) > 0) == (_signed_volume(repaired) > 0)


def test_canonical_orientation_keeps_the_gram_exact() -> None:
    # Rational Gram but an angle with no exact Fraction: the direct exact triangular construction
    # must preserve the Gram rather than introducing the old CellParams ~1e-10 drift.
    structure = ASUStructure(
        Cell(((3, 0, 4), (0, 6, 0), (0, 0, 13))), 3, [WyckoffSite("a", FracVector([F(1, 3)]), "Si")], _species("Si")
    )
    oriented = lift_module._canonical_orientation(structure)
    assert oriented.cell.metric() == structure.cell.metric()
    canonical = _canonical_without_bfs(structure)
    assert _canonical_without_bfs(canonical) == canonical


def test_canonical_orientation_breaks_an_unrepresentable_cubic_frame_tie() -> None:
    """Point-group-related frames agree even when exact Cholesky needs nested radicals."""
    basis = SurdVector(
        {
            1: FracVector(((5427, 0, -4418), (0, 0, -4418), (-5427, 0, -4418)), denom=2000),
            3: FracVector(((0, 1809, 0), (0, -3618, 0), (0, 1809, 0)), denom=2000),
        },
        (3, 3),
    )
    point_rotation = SurdVector(((0, 0, -1), (0, -1, 0), (-1, 0, 0)))
    sites = [WyckoffSite("a", FracVector(()), "Si")]
    first = ASUStructure(Cell(basis), 221, sites, _species("Si"))
    second = ASUStructure(Cell(point_rotation * basis), 221, sites, _species("Si"))

    # sqrt(g00) is not a single supported squarefree radical here, so the metric-only triangular
    # rebuild deliberately declines. The terminal point-group orbit must still select one frame.
    assert lift_module._exact_triangular_basis(first.cell.metric(), left_handed=False) is None
    assert _canonical_without_bfs(first) == _canonical_without_bfs(second)


def test_left_handed_enantiomorphic_cell_is_handled_gracefully() -> None:
    # SG 144 (P3_1) is enantiomorphic: the -I re-expression conjugates to P3_2 and is rejected, so
    # the exact left-handed cell must be kept (N1) rather than the assert firing or -O silently
    # producing the enantiomorph.  A single 3-fold orbit is coplanar, so the frame handedness
    # (det sign) is the chirality statement here.
    left_handed = Cell(SurdVector(((1, 0, 0), (0, 1, 0), (0, 0, -1))) * CellParams((5, 5, 12, 90, 90, 120)).basis)
    assert left_handed.basis.det().sign() < 0
    structure = ASUStructure(
        left_handed, 144, [WyckoffSite("a", FracVector((F(1, 7), F(2, 11), F(3, 13))), "Si")], _species("Si")
    )
    first = _canonical_without_bfs(structure)
    second = _canonical_without_bfs(structure)
    assert first.spacegroup.it_number == 144
    assert first.cell.basis.det().sign() < 0  # chirality preserved, no enantiomorph
    assert _site_key(first) == _site_key(second) and first.cell.basis == second.cell.basis  # deterministic
    assert _canonical_without_bfs(first) == first


def test_chiral_multi_species_invariance_battery() -> None:
    """Chiral four-species invariance via the production path minus the BFS.

    The variants are compared through ``_canonical_without_bfs`` (Niggli entry + normal form +
    canonical orientation), which equals ``canonicalize`` for a symmetryless crystal whose highest
    symmetry is its own group.  Full ``canonicalize`` on this four-site P1 input does not complete
    within ~5 minutes (its failed-lift BFS is the slow part, not the canonical form), so the full
    end-to-end equivalence is not asserted here; multi-species full-``canonicalize`` coherence is
    covered by the fast zincblende test above.
    """
    # Chiral, four-species content in a rational-length cell.  Variants exercise the origin- and
    # orbit-representative-choice invariance that the F1/F2 fixes provide, and chirality preservation.
    # Axis-relabel/shear are NOT included here: for a symmetryless crystal they additionally expose
    # the lattice-holohedry orientation choice (a 2_y-equivalent Niggli cell), which is outside the
    # four review defects; basis-choice invariance for symmetric crystals is covered by the Po/CsCl
    # batteries above.
    cell = ((5, 0, 0), (0, 6, 0), (0, 0, 7))
    base_sites = _chiral_sites()
    species = ["Fe", "Co", "Ni", "Cu"]
    base = ASUStructure(Cell(cell), 1, base_sites, _species(*species))

    def key(structure: ASUStructure) -> tuple[Any, ...]:
        return (structure.spacegroup.it_number, _site_key(structure), structure.cell.basis)

    reference_key = key(_canonical_without_bfs(base))

    translated = list(base_sites)
    translated[0] = WyckoffSite("a", FracVector((F(1, 7) + 1, F(2, 11), F(3, 13) - 1)), "Fe")
    variants = {
        "base": base,
        "origin_shift": ASUStructure(
            Cell(cell),
            1,
            [
                WyckoffSite(
                    s.wyckoff, _wrapped_shift(s.free_params.to_fractions(), (F(1, 5), F(1, 7), F(1, 3))), s.species
                )
                for s in base_sites
            ],
            _species(*species),
        ),
        "lattice_translate": ASUStructure(Cell(cell), 1, translated, _species(*species)),
    }
    for name, structure in variants.items():
        canonical = _canonical_without_bfs(structure)
        # (a) every same-crystal description yields the identical canonical representative (full key,
        # including the exact cell basis).
        assert key(canonical) == reference_key, name
        # (b) chirality is preserved: the canonical signed volume keeps the input's sign.
        assert (_signed_volume(structure) > 0) == (_signed_volume(canonical) > 0), name


# --- exact supercell collapse (translational reduction at P1 entry) ---------------------------


def _reduce(cell: object, sites: list[WyckoffSite], species: list[str]) -> ASUStructure:
    return lift_module._primitive_reduced_entry(ASUStructure(Cell(cell), 1, sites, _species(*species)))


def test_primitive_reduction_collapses_an_exact_supercell() -> None:
    # Po 1x1x2 (two atoms, translation (0,0,1/2)) collapses to the primitive cubic cell, one atom;
    # its expansion is the same crystal as the primitive Po.
    reduced = _reduce(
        ((5, 0, 0), (0, 5, 0), (0, 0, 10)),
        [WyckoffSite("a", FracVector((0, 0, 0)), "Po"), WyckoffSite("a", FracVector((0, 0, F(1, 2))), "Po")],
        ["Po"],
    )
    assert len(reduced.wyckoff_sites) == 1
    assert reduced.cell.basis == Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))).basis
    base = _p1(Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))), [WyckoffSite("a", FracVector((0, 0, 0)), "Po")])
    assert same_crystal(UnitcellStructureView(reduced), UnitcellStructureView(base))


def test_primitive_reduction_ignores_near_and_defective_supercells() -> None:
    # (a) a near-translation (0,0,0.4999) is not exact -> no reduction.
    near = _reduce(
        ((5, 0, 0), (0, 5, 0), (0, 0, 10)),
        [WyckoffSite("a", FracVector((0, 0, 0)), "Po"), WyckoffSite("a", FracVector((0, 0, F(4999, 10000))), "Po")],
        ["Po"],
    )
    assert len(near.wyckoff_sites) == 2
    # (b) an exact supercell with one atom displaced (a defect) has no self-translation -> no reduction.
    defect = _reduce(
        ((5, 0, 0), (0, 5, 0), (0, 0, 10)),
        [WyckoffSite("a", FracVector((0, 0, 0)), "Po"), WyckoffSite("a", FracVector((F(1, 10), 0, F(1, 2))), "Po")],
        ["Po"],
    )
    assert len(defect.wyckoff_sites) == 2


def test_primitive_reduction_is_site_order_invariant_and_scales_charge() -> None:
    # (c, the uniqueness pin at fixed cell) A genuine 2x two-species supercell reduces with n=2, and
    # the reduction is invariant to input site order -- the finer lattice is generated by ALL exact
    # translations and its HNF basis is a function of that lattice alone.  (Two *different* supercell
    # cells, diagonal vs sheared, reduce to different-orientation primitive cells that the downstream
    # Niggli + normal form unify; that end-to-end equality is the battery test below.)
    cell = ((4, 0, 0), (0, 4, 0), (0, 0, 8))
    sites = [
        WyckoffSite("a", FracVector((0, 0, 0)), "Cs"),
        WyckoffSite("a", FracVector((0, 0, F(1, 2))), "Cs"),
        WyckoffSite("a", FracVector((F(1, 2), F(1, 2), F(1, 4))), "Cl"),
        WyckoffSite("a", FracVector((F(1, 2), F(1, 2), F(3, 4))), "Cl"),
    ]

    def key(asu: ASUStructure) -> tuple[Any, ...]:
        return (
            asu.cell.basis,
            tuple(sorted((s.species, tuple(s.free_params.to_fractions())) for s in asu.wyckoff_sites)),
        )

    forward = _reduce(cell, sites, ["Cs", "Cl"])
    reversed_ = _reduce(cell, list(reversed(sites)), ["Cs", "Cl"])
    assert len(forward.wyckoff_sites) == 2  # n = 2
    assert key(forward) == key(reversed_)
    # charge is per-cell content, so it divides by the multiplicity.
    charged = ASUStructure(Cell(cell), 1, sites, _species("Cs", "Cl"), charge=F(6))
    assert lift_module._primitive_reduced_entry(charged).charge == F(3)


def _supercell_result_key(structure: ASUStructure) -> tuple[Any, ...]:
    result = canonicalize(structure, tolerance=1e-3)
    return (result.spacegroup.it_number, _site_key(result.asu), result.asu.cell.basis)


def test_primitive_reduction_preserves_cartesian_precision() -> None:
    # coordinate_precision is fractional; the subdivided cell must rescale it so the *Cartesian*
    # precision (and the derived tolerance) is invariant, not silently tightened.
    supercell = ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 10)), precision=F(1, 10000)),
        1,
        [WyckoffSite("a", FracVector((0, 0, 0)), "Po"), WyckoffSite("a", FracVector((0, 0, F(1, 2))), "Po")],
        _species("Po"),
        coordinate_precision=F(1, 10000),
    )
    reduced = lift_module._primitive_reduced_entry(supercell)
    before = UnitcellStructureView(supercell)
    after = UnitcellStructureView(reduced)
    assert after.cartesian_precision() == before.cartesian_precision()
    assert structure_tolerance(after) == structure_tolerance(before)


def test_primitive_reduction_tolerates_duplicate_sites() -> None:
    # A duplicated (species, coordinate) site has no exact self-translation, so the reduction must
    # pass it through untouched rather than reaching its internal "uneven copies" consistency error.
    duplicated = ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 10))),
        1,
        [
            WyckoffSite("a", FracVector((0, 0, 0)), "Po"),
            WyckoffSite("a", FracVector((0, 0, 0)), "Po"),
            WyckoffSite("a", FracVector((0, 0, F(1, 2))), "Po"),
        ],
        _species("Po"),
    )
    reduced = lift_module._primitive_reduced_entry(duplicated)  # must not raise
    assert len(reduced.wyckoff_sites) == 3


# --- all_paths: alternate Baernighausen routes to one terminal ---------------------------------


def _cubic_c() -> ASUStructure:
    return ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))), 221, [WyckoffSite("a", FracVector(()), "C")], _species("C")
    )


def test_all_paths_returns_every_route_to_one_terminal() -> None:
    # A 221 crystal expressed in its orthorhombic subgroup 47 lifts back to 221 by two distinct
    # Baernighausen routes (verified: two different paths, one terminal).
    child = subgroup_representation(_cubic_c(), 47).asu
    default = highest_symmetry(child, tolerance=1e-3)
    everything = highest_symmetry(child, tolerance=1e-3, all_paths=True)
    # Default collapses to one path per terminal; all_paths returns the superset.
    assert len(default) == 1
    assert len(everything) > len(default)
    terminal = lambda result: (result.spacegroup.it_number, _site_key(result.asu), result.asu.cell.basis)
    assert terminal(default[0]) == (221, (("C", "a", ()),), _cubic_c().cell.basis)
    # Every all_paths result is the same terminal representative; only the routes differ.
    assert {terminal(result) for result in everything} == {terminal(default[0])}
    assert len({result.path for result in everything}) == len(everything)
    # No regression: default mode is bit-identical to the flag-less call and contains default[0]'s route.
    assert highest_symmetry(child, tolerance=1e-3, all_paths=False) == default
    assert default[0].path in {result.path for result in everything}


# --- scramble invariance harness (spirit of symmetry_finder's --scramble; test-only) ----------


def _seed_unimodular(rng: random.Random) -> FracVector:
    """A deterministic unimodular integer matrix: a product of random elementary shears (det 1)."""
    matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for _ in range(4):
        i, j = rng.sample(range(3), 2)
        multiple = rng.choice([-2, -1, 1, 2])
        for column in range(3):
            matrix[i][column] += multiple * matrix[j][column]
    return FracVector(matrix)


def _expanded_p1(reference: ASUStructure) -> ASUStructure:
    """The reference crystal as a plain P1 ASU (one site per expanded atom)."""
    view = UnitcellStructureView(reference)
    species = list(view.species_at_sites)
    sites = [
        WyckoffSite("a", FracVector(coordinate).normalize(), name)
        for coordinate, name in zip(view.sites.reduced_coords.to_fractions(), species)
    ]
    return ASUStructure(Cell(view.cell.basis), 1, sites, reference.species)


def _scrambled_p1(reference: ASUStructure, seed: int) -> ASUStructure:
    """A seed-deterministic P1 description of the same crystal: unimodular shear + origin shift + reorder.

    ``basis' = M basis`` (M unimodular) with coordinates ``f' = f M^-1 + t`` (rational origin ``t``),
    then the site list is permuted -- the transformations `canonicalize` must undo exactly.
    """
    rng = random.Random(seed)
    view = UnitcellStructureView(reference)
    coordinates = [FracVector(coordinate) for coordinate in view.sites.reduced_coords.to_fractions()]
    species = list(view.species_at_sites)
    matrix = _seed_unimodular(rng)
    inverse = matrix.inv()
    denominator = rng.choice([2, 3, 4, 5, 6])
    shift = FracVector([F(rng.randrange(denominator), denominator) for _ in range(3)])
    scrambled = [(coordinate * inverse + shift).normalize() for coordinate in coordinates]
    order = list(range(len(species)))
    rng.shuffle(order)
    sites = [WyckoffSite("a", scrambled[index], species[index]) for index in order]
    return ASUStructure(Cell(SurdVector(matrix) * view.cell.basis), 1, sites, reference.species)


def _scramble_reference(*sites: WyckoffSite, cell: Cell, spacegroup: int, species: list[str]) -> ASUStructure:
    return ASUStructure(cell, spacegroup, list(sites), _species(*species))


_SCRAMBLE_BATTERY = {
    "Po-221": _scramble_reference(
        WyckoffSite("a", FracVector(()), "Po"),
        cell=Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        spacegroup=221,
        species=["Po"],
    ),
    "CsCl-221": _scramble_reference(
        WyckoffSite("a", FracVector(()), "Cs"),
        WyckoffSite("b", FracVector(()), "Cl"),
        cell=Cell(((4, 0, 0), (0, 4, 0), (0, 0, 4))),
        spacegroup=221,
        species=["Cs", "Cl"],
    ),
    "P4mmm-123-tetragonal": _scramble_reference(
        WyckoffSite("a", FracVector(()), "Ti"),
        cell=Cell(((4, 0, 0), (0, 4, 0), (0, 0, 6))),
        spacegroup=123,
        species=["Ti"],
    ),
    "P6mmm-191-hexagonal": _scramble_reference(
        WyckoffSite("a", FracVector(()), "Mg"),
        cell=Cell(CellParams((4, 4, 6, 90, 90, 120)).basis),
        spacegroup=191,
        species=["Mg"],
    ),
    # FCC: the Niggli primitive strands at a lower symmetry until the conventional-cell re-choice tier
    # re-expresses the child so the F-centred cubic hop fits.
    "NaCl-225-fcc": _scramble_reference(
        WyckoffSite("a", FracVector(()), "Na"),
        WyckoffSite("b", FracVector(()), "Cl"),
        cell=Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        spacegroup=225,
        species=["Na", "Cl"],
    ),
    # R-centred trigonal: the hexagonal conventional cell is an intrinsic threefold supercell of the
    # carried child's conventional lattice, so only the centred recell arm's primitive-lattice
    # search (plus its discrete-translation crossing) can present the R-centred hop.
    "Bi-166-rhombohedral": _scramble_reference(
        WyckoffSite("c", FracVector((F(234, 1000),)), "Bi"),
        cell=Cell(CellParams((F(45, 10), F(45, 10), F(118, 10), 90, 90, 120)).basis),
        spacegroup=166,
        species=["Bi"],
    ),
}


@pytest.mark.extended
def test_scrambled_single_atom_po_canonicalizes_to_cubic() -> None:
    reference = _result_key(canonicalize(_expanded_p1(_SCRAMBLE_BATTERY["Po-221"]), tolerance=1e-3))
    assert reference[0] == 221
    assert _result_key(canonicalize(_scrambled_p1(_SCRAMBLE_BATTERY["Po-221"], 1), tolerance=1e-3)) == reference


def test_p1_cubic_metric_stabilizer_normalizes_a_signed_permutation() -> None:
    """A P1 result remains invariant across the finite boundary stabilizer of a cubic Niggli Gram."""
    cell_rows = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    sites = [
        WyckoffSite("a", FracVector((F(1, 7), F(2, 7), F(3, 7))), "C"),
        WyckoffSite("a", FracVector((F(2, 7), F(4, 7), F(6, 7))), "O"),
        WyckoffSite("a", FracVector((F(3, 7), F(5, 7), F(1, 7))), "N"),
    ]
    base = ASUStructure(Cell(cell_rows), 1, sites, _species("C", "O", "N"))
    # det=+1; this is another Niggli-reduced cubic basis, not a shear to be removed by one path.
    permuted = _rebased(cell_rows, sites, ["C", "O", "N"], FracVector(((0, 1, 0), (1, 0, 0), (0, 0, -1))))

    first = canonical_asu(base, lift=False)
    second = canonical_asu(permuted, lift=False)

    assert first.spacegroup.it_number == second.spacegroup.it_number == 1
    assert first.cell.basis == second.cell.basis
    assert first.wyckoff_sites == second.wyckoff_sites
    assert _canonical_without_bfs(first) == first


def test_p1_elongated_metric_stabilizer_enumeration_is_exact() -> None:
    gram = ((F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(10_000)))
    operations = lift_module._metric_automorphism_operations(gram)

    assert len(operations) == 16
    for operation in operations:
        basis_change = operation.matrix.T().inv()
        assert basis_change * FracVector(gram) * basis_change.T() == FracVector(gram)


@pytest.mark.parametrize("number", (43, 82, 157, 215))
@pytest.mark.extended
def test_canonical_asu_fixture_scramble_normalizes_full_affine_cosets(number: int) -> None:
    """A tabulated coset representative is expanded by its group members before normal-form keying."""
    fixture = Path(__file__).with_name("fixtures") / "structreading" / f"{number}.cif"
    source = load(str(fixture), repair=True)

    reference = canonical_asu(UnitcellStructureView(source), lift=False)
    scrambled = canonical_asu(_scrambled_p1(source, number * 1000 + 1), lift=False)

    assert scrambled.spacegroup == reference.spacegroup
    assert scrambled.cell.basis == reference.cell.basis
    assert scrambled.wyckoff_sites == reference.wyckoff_sites
    assert _canonical_without_bfs(reference) == reference
    if number == 43:
        # A proper Cartesian rotation changes no fractional data or chirality and must not influence
        # the final tie between normalizer-equivalent cell bases.
        rotation = SurdVector(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
        rotated = ASUStructure(
            Cell(
                source.cell.basis * rotation,
                precision=source.cell.precision,
                periodicity=source.cell.periodicity,
            ),
            source.spacegroup,
            source.wyckoff_sites,
            source.species,
            transform=source.transform,
            coordinate_precision=source.coordinate_precision,
            charge=source.charge,
        )
        rotated_result = canonical_asu(UnitcellStructureView(rotated), lift=False)
        assert rotated_result.cell.basis == reference.cell.basis
        assert rotated_result.wyckoff_sites == reference.wyckoff_sites


@pytest.mark.parametrize(
    ("number", "seed"),
    ((1, 1001), (6, 6010), (10, 10005), (202, 202001)),
)
@pytest.mark.extended
def test_canonical_asu_fixture_scramble_avoids_pathological_exact_arithmetic(number: int, seed: int) -> None:
    """Sheared and large recognized cells retain the fixture's exact no-lift normal form."""
    fixture = Path(__file__).with_name("fixtures") / "structreading" / f"{number}.cif"
    source = load(str(fixture), repair=True)

    reference = canonical_asu(UnitcellStructureView(source), lift=False, preserve_chirality=True)
    scrambled = canonical_asu(_scrambled_p1(source, seed), lift=False, preserve_chirality=True)

    assert scrambled.spacegroup == reference.spacegroup
    assert scrambled.cell.basis == reference.cell.basis
    assert scrambled.wyckoff_sites == reference.wyckoff_sites


def _count_recell_searches(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]
    real = lift_module._search_conventional_basis

    def counting(gram: Any, system: str) -> Any:
        calls[0] += 1
        return real(gram, system)

    monkeypatch.setattr(lift_module, "_search_conventional_basis", counting)
    return calls


def _capture_recell_applications(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every re-expression operation the re-choice tier actually APPLIES.

    Only ``_apply_normalizer_operation`` calls made while ``_recell_lifts`` is on the stack are
    captured (the tier makes no others), so the list is exactly the operations the tier applies --
    integer-arm re-choices, centred-arm half-integer re-choices, and the centred arm's discrete
    normalizer translations.
    """
    operations: list[Any] = []
    active = [False]
    real_recell = lift_module._recell_lifts
    real_apply = lift_module._apply_normalizer_operation

    def recell_spy(structure: Any, target: Any, tolerance: float) -> Any:
        active[0] = True
        try:
            return real_recell(structure, target, tolerance)
        finally:
            active[0] = False

    def apply_spy(structure: Any, operation: Any) -> Any:
        if active[0]:
            operations.append(operation)
        return real_apply(structure, operation)

    monkeypatch.setattr(lift_module, "_recell_lifts", recell_spy)
    monkeypatch.setattr(lift_module, "_apply_normalizer_operation", apply_spy)
    return operations


@pytest.mark.extended
def test_conventional_recell_tier_is_dormant_on_a_p_lattice(monkeypatch: pytest.MonkeyPatch) -> None:
    # A state that climbs normally never reaches the third tier, so the exact re-choice search never
    # runs on a P-lattice input -- zero firings, not merely zero accepted lifts.
    calls = _count_recell_searches(monkeypatch)
    result = canonicalize(_expanded_p1(_SCRAMBLE_BATTERY["Po-221"]), tolerance=1e-3)
    assert result.spacegroup.it_number == 221
    assert calls[0] == 0


@pytest.mark.extended
def test_conventional_recell_tier_lands_r_centred_bi(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sanity anchor for the centred recell arm: the R-centred hexagonal conventional cell is an
    # intrinsic threefold supercell of the carried child's CONVENTIONAL lattice, so the integer
    # arm's conventional-lattice search can never produce it.  What lands Bi is (a) the parent-cell
    # search on the child's PRIMITIVE lattice (the det-3 hexagonal basis lives there) and (b) the
    # discrete-normalizer-translation crossing that places the atoms on the tabulated splitting's
    # origin coset.  (The applied re-choice matrix itself happens to be integer; half-integer maps
    # are a supported generalization no probed group has exercised.)
    applied = _capture_recell_applications(monkeypatch)
    landings = [0]
    real_centred = lift_module._centred_recell_lifts

    def centred_spy(structure: Any, transform: Any, child_gram: Any, tolerance: float) -> Any:
        lifted = real_centred(structure, transform, child_gram, tolerance)
        if lifted:
            landings[0] += 1
        return lifted

    monkeypatch.setattr(lift_module, "_centred_recell_lifts", centred_spy)
    bismuth = _SCRAMBLE_BATTERY["Bi-166-rhombohedral"]
    result = canonicalize(_expanded_p1(bismuth), tolerance=1e-3)
    assert result.spacegroup.it_number == 166
    assert landings[0] > 0, "the centred arm must be what landed the R-centred hop"
    # The winning re-choice rides the primitive-lattice search (a det-3 parent basis unreachable by
    # the integer arm's conventional-lattice search) and the discrete-normalizer-translation
    # crossing -- the half-lattice shift is observable in the applied operations.
    shifted = [
        operation
        for operation in applied
        if operation.matrix == FracVector.eye((3, 3))
        and any(value != 0 for value in FracVector(operation.vector).to_fractions())
    ]
    assert shifted, "the centred arm's discrete-translation crossing must have been applied"
    # The tier invariant extends to the centred arm: every applied operation is volume- and
    # orientation-preserving (det exactly +1; half-integer entries allowed).
    assert all(operation.determinant() == 1 for operation in applied)
    # Direct entry at 166 and the P1 climb agree exactly -- the arm restores full entry coherence.
    direct = canonicalize(bismuth, tolerance=1e-3)
    assert _result_key(result) == _result_key(direct)


def test_non_integer_centred_automorphisms_are_rejected_at_the_op_set_gate() -> None:
    # The centred arm's half-integer branch is the mathematically complete formulation, but no
    # centred group is known to exercise it: bounded review probes (~91k non-integer det-1
    # centred-unimodular maps across all 36 centred standard settings) found ZERO that pass the
    # op-set equality check, and every map the arm has accepted in practice is integer.  This pins
    # the current observable contract -- non-integer candidates are enumerated and rejected at the
    # op-set gate -- for one group per centring class (C-mono, F-ortho, R-trigonal, I-cubic).
    identity = tuple(tuple(F(1) if row == column else F(0) for column in range(3)) for row in range(3))
    bumps = [identity] + [
        tuple(tuple(F(1) if r == c else (F(1) if (r, c) == (i, j) else F(0)) for c in range(3)) for r in range(3))
        for i in range(3)
        for j in range(3)
        if i != j
    ]
    for it_number in (12, 69, 166, 229):
        spacegroup = Spacegroup.standard(it_number)
        lattice = lift_module._translation_lattice(spacegroup)
        lattice_transpose = tuple(tuple(lattice[row][column] for row in range(3)) for column in range(3))
        index = round(1 / float(lift_module._rational_determinant(lattice_transpose)))
        # Deterministic candidate family: diag(d) * elementary-bump with det = centring index, so
        # V = X * C^T is volume-preserving; keep the non-integer ones.
        diagonals = sorted(
            {
                combo
                for combo in itertools.product((1, 2, 3, 4), repeat=3)
                if combo[0] * combo[1] * combo[2] == abs(index)
            }
        )
        non_integer = []
        for diagonal in diagonals:
            for bump in bumps:
                x_matrix = tuple(tuple(F(diagonal[r]) * bump[r][c] for c in range(3)) for r in range(3))
                candidate = lift_module._matmul3(x_matrix, lattice_transpose)
                if abs(lift_module._rational_determinant(candidate)) != 1:
                    continue
                if all(value.denominator == 1 for row in candidate for value in row):
                    continue
                non_integer.append(candidate)
        assert non_integer, it_number  # the candidate family genuinely reaches the branch
        for candidate in non_integer:
            assert not lift_module._resetting_preserves_group(spacegroup, FracVector(candidate).T().inv()), (
                it_number,
                candidate,
            )
        # Positive control: the identity trivially preserves the op set, so the gate discriminates.
        assert lift_module._resetting_preserves_group(spacegroup, FracVector(identity))
