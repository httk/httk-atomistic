"""Tests for exact one-hop backward symmetry lifts."""

from fractions import Fraction

import pytest
from httk.core import FracVector, SurdVector

import httk.atomistic.symmetry.lift as lift_module
from httk.atomistic import (
    ASUStructure,
    Cell,
    Spacegroup,
    Species,
    WyckoffSite,
    backward_lift,
    canonicalize,
    data,
    highest_symmetry,
    lift_candidates,
    rerepresent,
    same_crystal,
    subgroup_representation,
)
from httk.atomistic.composition import Assembly
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.lift import (
    _apply_normalizer,
    _canonical_result_key,
    _highest_lifts,
    _linear_solve,
    _normalizer_retries,
    _structure_signature,
)
from httk.atomistic.symmetry.subgroups import subgroup_transforms

F = Fraction
NO_PARAMETERS = FracVector(())


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _parent(
    it_number: int, sites: list[WyckoffSite], basis: object = ((5, 0, 0), (0, 5, 0), (0, 0, 5))
) -> ASUStructure:
    return ASUStructure(Cell(basis), it_number, sites, _species(*(sorted({site.species for site in sites}))))


def _manual_affine_image(structure: ASUStructure, operation: AffineOperation) -> ASUStructure:
    sites = []
    for site in structure.wyckoff_sites:
        position = structure.spacegroup.wyckoff_position(site.wyckoff)
        transformed = [operation.apply_wrapped(point) for point in position.coordinates(site.free_params)]
        match = structure.spacegroup.identify_wyckoff(transformed[0])
        assert match is not None
        sites.append(WyckoffSite(match[0].letter, match[1], site.species))
    return ASUStructure(
        Cell(SurdVector(operation.matrix.T().inv()) * structure.cell.basis),
        structure.spacegroup,
        sites,
        structure.species,
    )


@pytest.mark.parametrize(
    ("parent_number", "child_number", "sites", "basis"),
    (
        (
            221,
            123,
            (
                WyckoffSite("a", NO_PARAMETERS, "Sr"),
                WyckoffSite("b", NO_PARAMETERS, "Ti"),
                WyckoffSite("c", NO_PARAMETERS, "O"),
            ),
            ((5, 0, 0), (0, 5, 0), (0, 0, 5)),
        ),
        (15, 2, (WyckoffSite("e", FracVector([F(1, 3)]), "Si"),), ((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        (3, 4, (WyckoffSite("a", FracVector([F(1, 3)]), "Si"),), ((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        (166, 148, (WyckoffSite("a", NO_PARAMETERS, "Bi"),), CellParams((5, 5, 12, 90, 90, 120)).basis),
    ),
)
def test_descent_lift_round_trip(
    parent_number: int, child_number: int, sites: tuple[WyckoffSite, ...], basis: object
) -> None:
    parent = _parent(parent_number, list(sites), basis)
    child = subgroup_representation(parent, child_number).asu
    results = backward_lift(child, parent_number, tolerance=1e-3)
    assert any(result.residual == F(0) and same_crystal(result.asu, parent) for result in results)
    assert all(result.shift.dim == (3,) for result in results)


def test_lift_candidates_are_complete_and_deterministic() -> None:
    parent = _parent(221, [WyckoffSite("a", NO_PARAMETERS, "C")])
    child = subgroup_representation(parent, 123).asu
    first = lift_candidates(child, tolerance=1e-3)
    second = lift_candidates(child, tolerance=1e-3)
    assert first == second
    assert any(result.spacegroup.it_number == 221 and result.residual == F(0) for result in first)


def test_noisy_parameter_is_accepted_only_at_the_requested_tolerance() -> None:
    parent = _parent(15, [WyckoffSite("e", FracVector([F(1, 3)]), "Si")], ((5, 0, 0), (0, 6, 0), (0, 0, 7)))
    child = subgroup_representation(parent, 2).asu
    site = child.wyckoff_sites[0]
    values = site.free_params.to_fractions()
    values[2] += F(1, 100000)
    noisy = ASUStructure(child.cell, child.spacegroup, [WyckoffSite(site.wyckoff, values, site.species)], child.species)
    accepted = backward_lift(noisy, 15, tolerance=1e-3)
    assert accepted and all(result.residual > 0 for result in accepted)
    assert not backward_lift(noisy, 15, tolerance=1e-9)


def test_continuous_normalizer_shift_on_polar_child() -> None:
    parent = _parent(5, [WyckoffSite("a", FracVector([F(2, 17)]), "Si")], ((5, 0, 0), (0, 6, 0), (0, 0, 7)))
    child = subgroup_representation(parent, 3).asu
    translated = ASUStructure(
        child.cell,
        child.spacegroup,
        [
            WyckoffSite(site.wyckoff, FracVector([site.free_params.to_fractions()[0] + F(1, 7)]), site.species)
            for site in child.wyckoff_sites
        ],
        child.species,
    )
    results = backward_lift(translated, 5, tolerance=1e-3)
    assert any(result.residual == F(0) and result.shift.to_fractions()[1] != 0 for result in results)


def test_wrong_relation_and_rejection_guards() -> None:
    parent = _parent(15, [WyckoffSite("e", FracVector([F(1, 3)]), "Si")], ((5, 0, 0), (0, 6, 0), (0, 0, 7)))
    child = subgroup_representation(parent, 2).asu
    with pytest.raises(ValueError, match="221.*2"):
        backward_lift(child, 221)
    with pytest.raises(ValueError, match="site moments"):
        backward_lift(
            ASUStructure(
                child.cell,
                child.spacegroup,
                [WyckoffSite("i", child.wyckoff_sites[0].free_params, "Si", moment=CollinearSiteMoments([1]))],
                child.species,
            ),
            15,
        )
    with pytest.raises(ValueError, match="assemblies"):
        backward_lift(
            ASUStructure(
                child.cell, child.spacegroup, child.wyckoff_sites, child.species, assemblies=(Assembly(((0,),), (1,)),)
            ),
            15,
        )
    with pytest.raises(ValueError, match="molecular"):
        backward_lift(
            ASUStructure(child.cell, child.spacegroup, child.wyckoff_sites, child.species, molecular=True), 15
        )


def test_generic_p_minus_one_has_no_lift_when_species_counts_do_not_split() -> None:
    structure = ASUStructure(
        Cell([[5, 0, 0], [0, 6, 0], [0, 0, 7]]),
        2,
        [WyckoffSite("i", FracVector([F(1, 7), F(2, 11), F(3, 13)]), "Si")],
        _species("Si"),
    )
    assert backward_lift(structure, 11) == ()
    assert lift_candidates(structure) == ()


def test_noisy_parent_cell_is_snapped_by_the_metric_check() -> None:
    parent = _parent(221, [WyckoffSite("a", NO_PARAMETERS, "C")])
    child = subgroup_representation(parent, 123).asu
    noisy = ASUStructure(
        Cell(((F(500001, 100000), 0, 0), (0, 5, 0), (0, 0, 5))),
        child.spacegroup,
        child.wyckoff_sites,
        child.species,
    )
    accepted = backward_lift(noisy, 221)
    assert accepted and all(result.residual > 0 for result in accepted)
    assert all(len(set(result.asu.cell.lengths)) == 1 for result in accepted)
    assert not backward_lift(noisy, 221, tolerance=1e-9)


def test_canonicalize_recovers_two_level_cubic_parent() -> None:
    parent = _parent(221, [WyckoffSite("a", NO_PARAMETERS, "Sr")])
    child = subgroup_representation(parent, 148).asu
    result = canonicalize(child, tolerance=1e-3)
    assert result.spacegroup.it_number == 221
    assert len(result.path) == 2
    assert result.residual == F(0)
    assert same_crystal(result.asu, parent)


def test_terminal_and_generic_structures_canonicalize_to_themselves() -> None:
    terminal = _parent(221, [WyckoffSite("a", NO_PARAMETERS, "C")])
    terminal_result = highest_symmetry(terminal, tolerance=1e-3)
    assert len(terminal_result) == 1
    assert terminal_result[0].path == ()
    assert terminal_result[0].residual == F(0)
    assert terminal_result[0].asu == terminal
    generic = ASUStructure(
        Cell(((5, 0, 0), (0, 6, 0), (0, 0, 7))),
        2,
        [WyckoffSite("i", FracVector([F(1, 7), F(2, 11), F(3, 13)]), "Si")],
        _species("Si"),
    )
    generic_result = canonicalize(generic, tolerance=1e-3)
    assert generic_result.spacegroup.it_number == 2
    assert generic_result.path == ()
    assert generic_result.residual == F(0)


def test_normalizer_image_still_lifts_and_rerepresent_dispatches() -> None:
    parent = _parent(15, [WyckoffSite("e", FracVector([F(2, 17)]), "Si")], ((5, 0, 0), (0, 6, 0), (0, 0, 7)))
    child = subgroup_representation(parent, 2).asu
    record = data.affine_normalizer_coset_record(child.spacegroup.hall_entry)
    image = _apply_normalizer(child, record["affine_normalizer_cosets"][6])
    assert image is not None
    results = backward_lift(image, 15, tolerance=1e-3)
    transformed_parent = _apply_normalizer(parent, record["affine_normalizer_cosets"][6])
    assert transformed_parent is not None
    assert any(
        result.residual == F(0)
        and result.asu.cell.basis == transformed_parent.cell.basis
        and same_crystal(result.asu, transformed_parent)
        for result in results
    )
    dispatch_parent = _parent(5, [WyckoffSite("a", FracVector([F(2, 17)]), "Si")], ((5, 0, 0), (0, 6, 0), (0, 0, 7)))
    assert rerepresent(dispatch_parent, 5) is dispatch_parent
    assert same_crystal(rerepresent(dispatch_parent, 3), subgroup_representation(dispatch_parent, 3).asu)
    cubic_parent = _parent(221, [WyckoffSite("a", NO_PARAMETERS, "C")])
    cubic_child = subgroup_representation(cubic_parent, 123).asu
    assert same_crystal(rerepresent(cubic_child, 221, tolerance=1e-3), cubic_parent)
    with pytest.raises(ValueError, match="5.*15"):
        rerepresent(dispatch_parent, 15)


def test_highest_symmetry_is_deterministic() -> None:
    parent = _parent(221, [WyckoffSite("a", NO_PARAMETERS, "C")])
    child = subgroup_representation(parent, 123).asu
    assert highest_symmetry(child, tolerance=1e-3) == highest_symmetry(child, tolerance=1e-3)


def test_normalizer_retry_conjugates_through_a_nonidentity_hop() -> None:
    parent = _parent(15, [WyckoffSite("a", NO_PARAMETERS, "Si")])
    child = subgroup_representation(parent, 2).asu
    transform = next(item for item in subgroup_transforms(15, 2) if not item.operation.is_identity())
    assert transform.operation.matrix != FracVector.eye((3, 3))
    record = data.affine_normalizer_coset_record(child.spacegroup.hall_entry)
    operation = AffineOperation.from_record(record["affine_normalizer_cosets"][9])
    assert transform.operation.matrix * operation.matrix != operation.matrix * transform.operation.matrix
    cosetted = _manual_affine_image(child, operation)
    expected = _manual_affine_image(
        parent,
        transform.operation * operation * transform.operation.inverse(),
    )
    retry_results = _normalizer_retries(cosetted, Spacegroup.standard(15), 1e-3)
    assert retry_results
    assert any(
        result.asu.cell.basis == expected.cell.basis
        and tuple((site.wyckoff, site.free_params) for site in result.asu.wyckoff_sites)
        == tuple((site.wyckoff, site.free_params) for site in expected.wyckoff_sites)
        for result in retry_results
    )
    results = backward_lift(cosetted, 15, tolerance=1e-3)
    assert any(result.asu.cell.basis == expected.cell.basis for result in results)
    assert any(
        result.asu.cell.basis == expected.cell.basis
        and tuple((site.wyckoff, site.free_params) for site in result.asu.wyckoff_sites)
        == tuple((site.wyckoff, site.free_params) for site in expected.wyckoff_sites)
        for result in results
    )


def test_backward_lift_invokes_the_normalizer_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = _parent(
        5,
        [WyckoffSite("a", FracVector([F(2, 17)]), "Si")],
        ((5, 0, 0), (0, 6, 0), (0, 0, 7)),
    )
    child = subgroup_representation(parent, 1).asu
    operation = AffineOperation.from_record(
        data.affine_normalizer_coset_record(child.spacegroup.hall_entry)["affine_normalizer_cosets"][14]
    )
    cosetted = _manual_affine_image(child, operation)
    raw = lift_module._raw_lifts(cosetted, Spacegroup.standard(5), 1e-3)
    retry = _normalizer_retries(cosetted, Spacegroup.standard(5), 1e-3)
    retry_keys = {_canonical_result_key(result) for result in retry}
    raw_keys = {_canonical_result_key(result) for result in raw}
    retry_only = next(result for result in retry if _canonical_result_key(result) not in raw_keys)
    assert retry_only
    monkeypatch.setattr(lift_module, "_raw_lifts", lambda *_args: raw)
    monkeypatch.setattr(lift_module, "_normalizer_retries", lambda *_args: retry)
    results = backward_lift(cosetted, 5, tolerance=1e-3)
    result_keys = {_canonical_result_key(result) for result in results}
    assert raw_keys <= result_keys
    assert _canonical_result_key(retry_only) in result_keys
    assert retry_keys <= result_keys


def test_highest_symmetry_bfs_uses_retries_only_when_the_direct_lift_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _parent(5, [WyckoffSite("a", FracVector([F(2, 17)]), "Si")], ((5, 0, 0), (0, 6, 0), (0, 0, 7)))
    child = subgroup_representation(parent, 3).asu
    target = Spacegroup.standard(5)
    raw = lift_module._raw_lifts(child, target, 1e-3)
    retry = _normalizer_retries(child, target, 1e-3)
    retry_only = next(
        result for result in retry if _canonical_result_key(result) not in {_canonical_result_key(x) for x in raw}
    )
    monkeypatch.setattr(
        lift_module,
        "_normalizer_retries",
        lambda _structure, candidate, _tolerance: retry if candidate == target else (),
    )
    # The state normal form now collapses the variants unconditional retries used to add, so the BFS
    # path only falls back to retries when the direct lift found nothing for that parent target.
    monkeypatch.setattr(
        lift_module,
        "_raw_lifts",
        lambda _structure, candidate, _tolerance: raw if candidate == target else (),
    )
    assert retry_only not in _highest_lifts(child, 1e-3)
    monkeypatch.setattr(lift_module, "_raw_lifts", lambda _structure, _candidate, _tolerance: ())
    assert retry_only in _highest_lifts(child, 1e-3)


@pytest.mark.parametrize(
    "basis",
    (
        ((0, 5, 0), (0, 0, 5), (5, 0, 0)),
        CellParams((5, 5, 12, 90, 90, 120)).basis,
    ),
)
def test_noncanonical_exact_cells_survive_round_trip(basis: object) -> None:
    parent_number, child_number = (221, 123) if isinstance(basis, tuple) else (166, 148)
    parent = _parent(parent_number, [WyckoffSite("a", NO_PARAMETERS, "C")], basis)
    child = subgroup_representation(parent, child_number).asu
    results = backward_lift(child, parent_number, tolerance=1e-3)
    assert any(result.residual == F(0) and result.asu.cell.basis == parent.cell.basis for result in results)


def test_underdetermined_modular_solver_chooses_a_nonzero_free_variable() -> None:
    result = _linear_solve(((F(1, 2), F(-1)),), (F(-1, 4),))
    assert result == (F(1, 2), F(1, 2))


def test_bfs_signature_includes_the_exact_cell_gram_matrix() -> None:
    sites = [WyckoffSite("a", NO_PARAMETERS, "C")]
    first = _parent(221, sites, ((5, 0, 0), (0, 5, 0), (0, 0, 5)))
    second = _parent(221, sites, ((6, 0, 0), (0, 6, 0), (0, 0, 6)))
    assert _structure_signature(first) != _structure_signature(second)


def test_charged_k_hop_round_trip_unscales_exactly() -> None:
    parent = ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        221,
        [WyckoffSite("a", NO_PARAMETERS, "C")],
        _species("C"),
        charge=F(7),
    )
    child = subgroup_representation(parent, 148).asu
    assert child.charge == F(21)
    lifted = rerepresent(child, 221, tolerance=1e-3)
    assert lifted.charge == parent.charge
