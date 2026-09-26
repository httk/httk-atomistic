"""Unified public canonicalization, bounded search and resource-scope contracts."""

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as F

import pytest
from httk.core import FracVector

from httk.atomistic import (
    ASUStructure,
    ASUStructureView,
    BarePrototype,
    BarePrototypeView,
    BareProtostructure,
    CanonicalizationLimitError,
    Cell,
    Prototype,
    PrototypeView,
    Protostructure,
    Species,
    UnitcellStructureView,
    WyckoffSite,
    canonicalize,
    canonical_asu,
    canonical_asu_protostructure,
    canonical_bare_prototype,
    canonical_bare_protostructure,
    canonical_prototype,
    canonical_protostructure,
    search_supergroups,
)
from httk.atomistic.models.structure.asu import FundamentalDomainStructure
from httk.atomistic.models.structuretype.fundamental import FundamentalDomainTemplate
from httk.atomistic.symmetry import limits
import httk.atomistic.symmetry.lift as lift


def _source(group=47):
    return ASUStructure(
        Cell(((3, 0, 0), (0, 4, 0), (0, 0, 5))), group, (WyckoffSite("a", (), "Na"),), (Species("Na", ("Na",), (1,)),)
    )


def test_alias_and_structure_representation_independence(monkeypatch):
    pytest.importorskip("spglib")
    assert canonicalize is canonical_asu
    monkeypatch.setattr(lift, "_highest_lifts", lambda *args: pytest.fail("default invoked BFS"))
    source = _source()
    expected = canonical_asu_protostructure(source)
    for value in (
        source,
        ASUStructureView(source),
        UnitcellStructureView(source),
        UnitcellStructureView(source).unview(),
    ):
        actual = canonicalize(value)
        assert isinstance(actual, ASUStructure)
        assert actual == expected
    with pytest.raises(TypeError):
        canonical_asu(source, lift=True)


def test_declared_mode_never_recognizes_even_lazy_views(monkeypatch):
    import httk.atomistic.symmetry.canonical as module
    import httk.atomistic.models.structure.asu_view as views

    monkeypatch.setattr(module, "recognize_asu", lambda *args, **kwargs: pytest.fail("recognition"))
    monkeypatch.setattr(views, "recognize_asu", lambda *args, **kwargs: pytest.fail("recognition"))
    monkeypatch.setattr(lift, "_highest_lifts", lambda *args: pytest.fail("BFS"))
    source = _source()
    assert canonicalize(ASUStructureView(source), symmetry="declared") == canonicalize(source, symmetry="declared")
    unresolved = ASUStructureView(UnitcellStructureView(source).unview())
    with pytest.raises(TypeError, match="resolve"):
        canonicalize(unresolved, symmetry="declared")
    with pytest.raises(ValueError, match="recognition"):
        canonicalize(source, symmetry="declared", tolerance=1e-3)


def test_generic_classification_dispatch_and_examples():
    source = _source()
    domain = FundamentalDomainStructure(source.cell, source.spacegroup, source.wyckoff_sites, source.species)
    template = FundamentalDomainTemplate(source.cell, source.spacegroup, (WyckoffSite("a", (), "A"),), None)
    assigned = Protostructure(representative=domain)
    anonymous = Prototype(representative=template)
    cases = (
        (BarePrototype(47, (("a", "A"),)), canonical_bare_prototype),
        (BareProtostructure(47, (("a", source.species[0]),)), canonical_bare_protostructure),
        (assigned, canonical_protostructure),
        (anonymous, canonical_prototype),
        (PrototypeView(anonymous), canonical_prototype),
        (BarePrototypeView(anonymous), canonical_bare_prototype),
    )
    for value, explicit in cases:
        result = canonicalize(value)
        assert result == explicit(value)
        assert type(result) is type(explicit(value))
    with pytest.raises(ValueError, match="recognition"):
        canonicalize(anonymous, tolerance=1e-3)


@pytest.mark.parametrize("timeout", (0, -1, float("inf"), float("nan"), True))
def test_invalid_deadlines(timeout):
    with pytest.raises(ValueError, match="timeout"):
        canonicalize(_source(), symmetry="declared", timeout=timeout)
    with pytest.raises(ValueError, match="timeout"):
        search_supergroups(_source(), timeout=timeout)


def test_real_deadline_returns_no_canonical_value_and_resets_scope():
    with pytest.raises(CanonicalizationLimitError):
        canonicalize(_source(), symmetry="declared", timeout=1e-12)
    assert limits._active.get() is None
    search = search_supergroups(_source(), timeout=1e-12)
    assert not search.complete and not search.candidates
    assert search.reasons == ("deadline_exceeded",)
    assert limits._active.get() is None
    assert canonicalize(_source(), symmetry="declared", timeout=None).spacegroup.it_number == 47


def test_budget_rejects_late_native_result(monkeypatch):
    import httk.atomistic.symmetry.canonical_protostructure as terminal

    now = [0.0]
    monkeypatch.setattr(limits.time, "monotonic", lambda: now[0])

    def late_result(source, **kwargs):
        now[0] = 2.0
        return source

    monkeypatch.setattr(terminal, "_canonical_protostructure_asu", late_result)
    with pytest.raises(CanonicalizationLimitError):
        canonicalize(_source(), symmetry="declared", timeout=1)


def test_nested_budget_and_thread_isolation(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(limits.time, "monotonic", lambda: now[0])
    with limits._budget_scope(1):
        with limits._budget_scope(None):
            now[0] = 2
            with pytest.raises(CanonicalizationLimitError):
                limits._checkpoint()
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(limits._checkpoint).result() is None
    assert limits._active.get() is None


def test_complete_real_declared_search():
    source = _source()
    result = search_supergroups(source)
    assert result.complete and not result.reasons
    assert result.states_visited >= 1
    assert result.candidates
    assert result.candidates[0].asu == canonicalize(source, symmetry="declared")


def _stub_graph(monkeypatch):
    import httk.atomistic.symmetry.canonical_protostructure as terminal

    first, second, third = _source(47), _source(65), _source(69)
    monkeypatch.setattr(lift, "_canonical_entry", lambda x: x)
    monkeypatch.setattr(lift, "_normal_form", lambda x: x)
    monkeypatch.setattr(terminal, "_canonical_protostructure_asu", lambda x, **kwargs: x)
    results = tuple(lift.LiftResult(x, x.spacegroup, (), FracVector((0, 0, 0)), F(0)) for x in (second, third))
    return first, second, third, results


def test_state_limit_never_promotes_frontier_to_terminal(monkeypatch):
    first, _, _, results = _stub_graph(monkeypatch)
    monkeypatch.setattr(lift, "_highest_lifts", lambda x, tolerance: results)
    result = search_supergroups(first, max_states=1)
    assert not result.complete and not result.candidates
    assert result.reasons == ("state_limit_exceeded",)
    assert result.states_visited == 1


def test_partial_search_retains_only_finished_terminals(monkeypatch):
    first, second, third, results = _stub_graph(monkeypatch)
    now = [0.0]
    monkeypatch.setattr(limits.time, "monotonic", lambda: now[0])

    def children(value, tolerance):
        if value is first:
            return results
        if value is third:
            now[0] = 2
        return ()

    monkeypatch.setattr(lift, "_highest_lifts", children)
    result = search_supergroups(first, timeout=1)
    assert not result.complete and result.reasons == ("deadline_exceeded",)
    assert len(result.candidates) == 1
    assert result.candidates[0].spacegroup == second.spacegroup


def test_skipped_solver_work_makes_search_incomplete(monkeypatch):
    def skipped(value, tolerance):
        limits._incomplete("noisy_solver_branch_cap")
        return ()

    monkeypatch.setattr(lift, "_highest_lifts", skipped)
    result = search_supergroups(_source())
    assert not result.complete
    assert result.reasons == ("noisy_solver_branch_cap",)
    assert not result.candidates


def test_repeated_solver_skip_never_emits_later_unproven_terminal(monkeypatch):
    first, _, _, results = _stub_graph(monkeypatch)

    def children(value, tolerance):
        if value is first:
            return results
        limits._incomplete("noisy_solver_branch_cap")
        return ()

    monkeypatch.setattr(lift, "_highest_lifts", children)
    result = search_supergroups(first)
    assert result.states_visited == 3
    assert result.reasons == ("noisy_solver_branch_cap",)
    assert not result.complete and not result.candidates


def test_search_rejects_incomplete_terminal_normalization(monkeypatch):
    import httk.atomistic.symmetry.canonical_protostructure as terminal

    monkeypatch.setattr(lift, "_highest_lifts", lambda *args: ())

    def normalize(value, **kwargs):
        limits._incomplete("fourier_motzkin_limit")
        return value

    monkeypatch.setattr(terminal, "_canonical_protostructure_asu", normalize)
    result = search_supergroups(_source())
    assert not result.complete and not result.candidates
    assert result.reasons == ("fourier_motzkin_limit",)


def test_solver_limit_exception_becomes_incomplete_search(monkeypatch):
    def capped(value, tolerance):
        limits._incomplete("fourier_motzkin_limit")
        raise lift._SolverLimitError("Fourier-Motzkin constraint cap exceeded")

    monkeypatch.setattr(lift, "_highest_lifts", capped)
    result = search_supergroups(_source())
    assert not result.complete and not result.candidates
    assert result.reasons == ("fourier_motzkin_limit",)
    with pytest.raises(ValueError, match="constraint cap exceeded"):
        lift.highest_symmetry(_source())


def test_solver_skip_does_not_hide_unrelated_error(monkeypatch):
    def broken(value, tolerance):
        limits._incomplete("noisy_solver_branch_cap")
        raise ValueError("unrelated failure after skipped parent")

    monkeypatch.setattr(lift, "_highest_lifts", broken)
    with pytest.raises(ValueError, match="unrelated failure"):
        search_supergroups(_source())


def test_canonicalization_translates_solver_limit(monkeypatch):
    import httk.atomistic.symmetry.canonical_protostructure as terminal

    def capped(value, **kwargs):
        limits._incomplete("fourier_motzkin_limit")
        raise lift._SolverLimitError("Fourier-Motzkin constraint cap exceeded")

    monkeypatch.setattr(terminal, "_canonical_protostructure_asu", capped)
    with pytest.raises(CanonicalizationLimitError, match="cap exceeded"):
        canonicalize(_source(), symmetry="declared")
