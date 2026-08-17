"""Exact P1 lifts: trigonal/hexagonal metric enforcement and the lattice-based modular solver.

These cover phase 1 of making :func:`canonicalize` usable from a P1 (SG 1) input: the exact
lattice modular solve, the enforced hexagonal-axes metric for trigonal/hexagonal parents, and the
per-parent branch-cap guard in the breadth-first search.  The full raw-P1 canonicalization is not
asserted here; see the note at the end of the file.
"""

import itertools
from fractions import Fraction as F

import pytest
from httk.core import FracVector

import httk.atomistic.symmetry.lift as lift_module
from httk.atomistic import (
    ASUStructure,
    Cell,
    Spacegroup,
    Species,
    WyckoffSite,
    backward_lift,
    same_crystal,
    subgroup_representation,
)
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.symmetry.lift import (
    _cell_for_transform,
    _integer_options,
    _linear_solve,
    _rational_null_space,
)
from httk.atomistic.symmetry.subgroups import subgroup_transforms


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _p1(cell: Cell, sites: list[WyckoffSite]) -> ASUStructure:
    return ASUStructure(cell, 1, sites, _species(*sorted({site.species for site in sites})))


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


# --- 1c: end-to-end P1 canonicalization -------------------------------------------------------
#
# The brief's Po/CsCl "canonicalize a raw P1 cell all the way to IT 221" assertions are NOT
# shipped as tests here, on purpose.  Phase 1 makes every one-hop lift exact and fast (covered
# above), but ``highest_symmetry``'s breadth-first search does not yet collapse origin-equivalent
# parent representations: from a raw P1 cell the origin is a free continuous parameter, so a single
# atom has an enormous number of exact, origin-shifted higher-symmetry descriptions, each a
# distinct BFS state.  Measured here, the CsCl P1 search does not complete ~200 states in 5 minutes
# of CPU and keeps growing, so a full P1 canonicalization is not a viable test in any profile until
# the phase-2 Niggli pre-reduction and phase-3 normalizer post-step shrink that state space.  The
# manual measurement lives in the task scratchpad (``run_canon.py``/``bfs_cscl.py``).
