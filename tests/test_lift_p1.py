"""Exact P1 lifts: trigonal/hexagonal metric, the lattice modular solver, and P1 canonicalization.

Phase 1 covers the exact lattice modular solve, the enforced hexagonal-axes metric for
trigonal/hexagonal parents, and the per-parent branch-cap guard.  Phase 2 adds the
normalizer-canonical state normal form (Wyckoff demotion plus the continuous-translation and
affine-normalizer-coset quotients) that makes :func:`canonicalize` terminate from a raw P1 input.
"""

import itertools
from collections import Counter
from fractions import Fraction as F
from typing import Any

import pytest
from httk.core import FracVector, SurdVector

import httk.atomistic.symmetry.lift as lift_module
from httk.atomistic import (
    ASUStructure,
    Cell,
    Spacegroup,
    Species,
    UnitcellStructureView,
    WyckoffSite,
    backward_lift,
    canonicalize,
    data,
    same_crystal,
    subgroup_representation,
)
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.symmetry.lift import (
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


def _invariance_variants(
    cell_rows: object, sites: list[WyckoffSite], species: list[str]
) -> dict[str, ASUStructure]:
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


@pytest.mark.extended
def test_po_p1_invariance_battery() -> None:
    variants = _invariance_variants(((5, 0, 0), (0, 5, 0), (0, 0, 5)), [WyckoffSite("a", FracVector((0, 0, 0)), "Po")], ["Po"])
    reference = _result_key(canonicalize(variants["base"], tolerance=1e-3))
    assert reference[0] == 221 and reference[1] == (("Po", "a", ()),)
    for name, structure in variants.items():
        assert _result_key(canonicalize(structure, tolerance=1e-3)) == reference, name
    # Expansion sanity: the canonical result is the same crystal as the (unrotated) input.
    canonical = canonicalize(variants["base"], tolerance=1e-3)
    assert same_crystal(UnitcellStructureView(canonical.asu), UnitcellStructureView(variants["base"]))


@pytest.mark.extended
def test_cscl_p1_invariance_battery() -> None:
    variants = _invariance_variants(((4, 0, 0), (0, 4, 0), (0, 0, 4)), _cscl_sites(), ["Cs", "Cl"])
    reference = _result_key(canonicalize(variants["base"], tolerance=1e-3))
    assert reference[0] == 221 and reference[1] == (("Cl", "a", ()), ("Cs", "b", ()))
    for name, structure in variants.items():
        assert _result_key(canonicalize(structure, tolerance=1e-3)) == reference, name
