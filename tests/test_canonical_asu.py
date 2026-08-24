"""Tests for :func:`canonical_asu`: tolerant recognition composed with exact canonicalization."""

from fractions import Fraction as F

import pytest
from httk.core import FracVector

import httk.atomistic.symmetry.canonical as canonical_module
from httk.atomistic import (
    ASUStructure,
    Cell,
    Protostructure,
    Prototype,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    build_supercell,
    canonical_asu,
)
from httk.atomistic.symmetry.canonical import _fits_within
from httk.atomistic.symmetry.lift import _site_key


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _perturbed(structure: UnitcellStructure, denominator: int) -> UnitcellStructure:
    """Nudge every coordinate by a distinct, deterministic offset (no RNG).

    Offsets are ``(site*3 + axis + 1)/denominator``; a large ``denominator`` keeps the Cartesian
    displacement well below the base tolerance in the small cells used here.
    """
    coords = [
        [value + F(index * 3 + axis + 1, denominator) for axis, value in enumerate(row)]
        for index, row in enumerate(structure.sites.reduced_coords.to_fractions())
    ]
    return UnitcellStructure(structure.cell, coords, structure.species, structure.species_at_sites)


def _nacl() -> ASUStructure:
    return ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        225,
        [WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("b", FracVector(()), "Cl")],
        _species("Na", "Cl"),
    )


def _rutile() -> ASUStructure:
    # P4_2/mnm: Ti on 2a, O on 4f with x = 3/10 (~0.305).  Realistic small tetragonal cell so the
    # fractional perturbation stays well below the base Cartesian tolerance.
    return ASUStructure(
        Cell(((F(46, 10), 0, 0), (0, F(46, 10), 0), (0, 0, F(30, 10)))),
        136,
        [WyckoffSite("a", FracVector(()), "Ti"), WyckoffSite("f", FracVector([F(3, 10)]), "O")],
        _species("Ti", "O"),
    )


def test_nacl_noise_below_tolerance_canonicalizes_to_the_reference() -> None:
    pytest.importorskip("spglib")
    reference = canonical_asu(UnitcellStructureView(_nacl()))
    assert reference.spacegroup.it_number == 225
    # Min-key canonical origin puts the alphabetically-smaller species (Cl) on 1a.
    assert _site_key(reference) == (("Cl", "a", ()), ("Na", "b", ()))
    noisy = canonical_asu(_perturbed(UnitcellStructure(*_expanded(_nacl())), 400_000))
    assert noisy.spacegroup.it_number == 225
    assert _site_key(noisy) == _site_key(reference)


def test_rutile_free_parameter_stays_near_the_reference() -> None:
    pytest.importorskip("spglib")
    reference = canonical_asu(UnitcellStructureView(_rutile()))
    noisy = canonical_asu(_perturbed(UnitcellStructure(*_expanded(_rutile())), 400_000))
    # Same group, setting and Wyckoff multiset; exact P1 preconditioning can absorb this perturbation
    # into the canonical origin, so the fitted free parameter may equal the reference exactly.
    assert noisy.spacegroup.it_number == reference.spacegroup.it_number == 136

    def multiset(asu: ASUStructure) -> list[tuple[str, str]]:
        return sorted((s.species, s.wyckoff) for s in asu.wyckoff_sites)

    assert multiset(noisy) == multiset(reference)
    reference_x = next(s.free_params.to_fractions()[0] for s in reference.wyckoff_sites if s.species == "O")
    noisy_x = next(s.free_params.to_fractions()[0] for s in noisy.wyckoff_sites if s.species == "O")
    assert abs(noisy_x - reference_x) < F(1, 1000)  # well within the fractional tolerance of the 4.6 cell


def test_noise_above_tolerance_on_one_atom_returns_lower_symmetry() -> None:
    pytest.importorskip("spglib")
    cell, coords, species, species_at = _expanded(_nacl())
    displaced = [list(row) for row in coords]
    displaced[0] = [displaced[0][0] + F(6, 10000), displaced[0][1], displaced[0][2]]  # 6e-4 frac = 3e-3 A > base
    result = canonical_asu(UnitcellStructure(cell, displaced, species, species_at))
    # spglib never recognizes IT 225 here -- one atom is too far off at every swept symprec.  The
    # members that fit within the base tolerance recognize the genuine lower symmetry (IT 99), while
    # the loosest member, whose snapping would pull the displaced atom far enough to look cubic, is
    # rejected because that snap exceeds the base tolerance.  A lower-symmetry model is returned.
    assert result.spacegroup.it_number != 225


def test_fits_within_requires_an_injective_match() -> None:
    # Two input sites cannot both claim one model site while a third model site is left orphaned.
    cell = Cell(((10, 0, 0), (0, 10, 0), (0, 0, 10)))
    inputs = ASUStructure(
        cell,
        1,
        [
            WyckoffSite("a", FracVector((0, 0, 0)), "Na"),
            WyckoffSite("a", FracVector((0, F(1, 1000), 0)), "Na"),
            WyckoffSite("a", FracVector((F(1, 2), F(1, 2), F(1, 2))), "Na"),
        ],
        _species("Na"),
    )
    orphaned = ASUStructure(
        cell,
        1,
        [
            WyckoffSite("a", FracVector((0, F(5, 10000), 0)), "Na"),
            WyckoffSite("a", FracVector((0, F(2, 10), 0)), "Na"),  # 2 A from any input: orphaned
            WyckoffSite("a", FracVector((F(1, 2), F(1, 2), F(1, 2))), "Na"),
        ],
        _species("Na"),
    )
    view = UnitcellStructureView(inputs)
    assert not _fits_within(view, orphaned, 0.02)
    # The same model with its second site placed on the second input is a valid injective match.
    valid = ASUStructure(
        cell,
        1,
        [
            WyckoffSite("a", FracVector((0, F(5, 10000), 0)), "Na"),
            WyckoffSite("a", FracVector((0, F(1, 1000), 0)), "Na"),
            WyckoffSite("a", FracVector((F(1, 2), F(1, 2), F(1, 2))), "Na"),
        ],
        _species("Na"),
    )
    assert _fits_within(view, valid, 0.02)


def test_fits_within_finds_a_non_greedy_bijective_match() -> None:
    # Input 0 can use either model, while input 1 can use only model 0. A nearest-pair greedy
    # algorithm consumes model 0 for input 0 and fails; the valid bijection is input 0 -> model 1,
    # input 1 -> model 0.
    cell = Cell(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    inputs = ASUStructure(
        cell,
        1,
        [
            WyckoffSite("a", FracVector((0, 0, 0)), "Na"),
            WyckoffSite("a", FracVector((F(1, 10), 0, 0)), "Na"),
        ],
        _species("Na"),
    )
    model = ASUStructure(
        cell,
        1,
        [
            WyckoffSite("a", FracVector((F(4, 100), 0, 0)), "Na"),
            WyckoffSite("a", FracVector((F(94, 100), 0, 0)), "Na"),
        ],
        _species("Na"),
    )

    assert _fits_within(UnitcellStructureView(inputs), model, 0.07)


def test_symprec_sweep_rescues_a_tolerance_boundary_flip() -> None:
    pytest.importorskip("spglib")
    # NaCl shifted +1e-4 on every coordinate: recognition fails at the tight symprec (base/5) but
    # RECOGNIZES IT 225 at the loosest member (base*5), which loosest-first takes and returns (not via
    # the lift), so the default (lift=False) rescues it where the tight member alone cannot.
    cell, coords, species, species_at = _expanded(_nacl())
    shifted = [[value + F(1, 10000) for value in row] for row in coords]
    noisy = UnitcellStructure(cell, shifted, species, species_at)
    assert canonical_asu(noisy, factors=(F(1, 5), 1, 5)).spacegroup.it_number == 225
    # P1 preconditioning removes the global shift before spglib, so even the tight member now sees
    # the same cubic geometry rather than needing the loose boundary rescue.
    assert canonical_asu(noisy, factors=(F(1, 5),)).spacegroup.it_number == 225


def test_symprec_sweep_rejects_a_structurally_invalid_loose_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lazy expansion error rejects one candidate rather than aborting the whole sweep."""
    fit_calls = 0

    def staged_fit(_view: UnitcellStructureView, _recognized: ASUStructure, _tolerance: float) -> bool:
        nonlocal fit_calls
        fit_calls += 1
        if fit_calls == 1:
            raise ValueError("loose symprec merged two split sites")
        return True

    monkeypatch.setattr(canonical_module, "recognize_asu", lambda *_args, **_kwargs: _nacl())
    monkeypatch.setattr(canonical_module, "_fits_within", staged_fit)

    winner, failures = canonical_module._recognition_sweep(
        UnitcellStructureView(_nacl()),
        0.1,
        (1, 5),
    )

    assert winner is not None and winner.spacegroup.it_number == 225
    assert fit_calls == 2
    assert failures == ["0.5: recognized model is structurally invalid: loose symprec merged two split sites"]


def test_all_members_failing_lists_every_attempted_tolerance() -> None:
    pytest.importorskip("spglib")
    # Two Na sites 5e-5 A apart merge at every swept symprec, so no member reproduces the input; the
    # error lists each attempted symprec (loosest first).
    two_close_na = UnitcellStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        [[0, 0, 0], [0, 0, F(1, 100000)], [F(1, 2), F(1, 2), F(1, 2)]],
        _species("Na", "Cl"),
        ["Na", "Na", "Cl"],
    )
    with pytest.raises(ValueError, match=r"tried \[0\.005.*0\.001.*0\.0002"):
        canonical_asu(two_close_na, tolerance=1e-3, factors=(F(1, 5), 1, 5))


def test_default_and_lift_agree_on_a_clean_structure() -> None:
    pytest.importorskip("spglib")
    view = UnitcellStructureView(_nacl())
    default = canonical_asu(view)  # lift=False
    lifted = canonical_asu(view, lift=True)
    assert default.spacegroup.it_number == lifted.spacegroup.it_number == 225
    assert _site_key(default) == _site_key(lifted)
    assert default.cell.basis == lifted.cell.basis


def test_recognized_supercell_scales_extensive_charge_to_the_standard_cell() -> None:
    pytest.importorskip("spglib")
    charged = ASUStructure(
        _nacl().cell,
        _nacl().spacegroup,
        _nacl().wyckoff_sites,
        _nacl().species,
        charge=F(4),
    )
    supercell = build_supercell(charged, 2).structure

    assert supercell.charge == 32
    result = canonical_asu(supercell, lift=False)
    assert result.cell.basis == charged.cell.basis
    assert result.charge == charged.charge == 4


def test_lift_finds_pseudosymmetry_the_default_leaves_at_the_recognized_group() -> None:
    pytest.importorskip("spglib")
    # A cubic NaCl motif in a slightly tetragonal cell (c = 5.0008): at the tight symprec spglib
    # recognizes P4/mmm (IT 123), which the default returns as-is, while lift=True snaps the near-cubic
    # metric and reaches Fm-3m (IT 221) -- the pseudosymmetry recognition missed.
    tetragonal = ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5 + F(1, 1250)))),
        1,
        [
            WyckoffSite("a", FracVector((0, 0, 0)), "Na"),
            WyckoffSite("a", FracVector((F(1, 2), F(1, 2), F(1, 2))), "Cl"),
        ],
        _species("Na", "Cl"),
    )
    view = UnitcellStructureView(tetragonal)
    assert canonical_asu(view, factors=(F(1, 5),)).spacegroup.it_number == 123
    assert canonical_asu(view, factors=(F(1, 5),), lift=True).spacegroup.it_number == 221


def test_loosest_fitting_member_wins_and_stops_the_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("spglib")
    recognitions = 0
    real_recognize = canonical_module.recognize_asu
    real_stage = canonical_module._canonical_without_bfs
    stages = 0

    def counting_recognize(*args, **kwargs):
        nonlocal recognitions
        recognitions += 1
        return real_recognize(*args, **kwargs)

    def counting_stage(structure: ASUStructure, **kwargs: object) -> ASUStructure:
        nonlocal stages
        stages += 1
        return real_stage(structure, **kwargs)

    monkeypatch.setattr(canonical_module, "recognize_asu", counting_recognize)
    monkeypatch.setattr(canonical_module, "_canonical_without_bfs", counting_stage)
    # Clean NaCl: the loosest symprec (base*5) already recognizes and fits, so the sweep stops after
    # one recognition; the exact canonicalization stage runs once for the recognized result (P1
    # preconditioning is now the lighter Niggli path) -- no matter how many factors are passed.
    canonical_asu(UnitcellStructureView(_nacl()), factors=(F(1, 5), 1, 5))
    assert recognitions == 1
    assert stages == 1


def test_p1_rescue_reverses_the_canonical_frame_instead_of_retrying_the_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The site-order-sensitive spglib retry must remain representation-independent."""
    frames: list[tuple[object, tuple[str, ...]]] = []

    def staged_sweep(
        view: UnitcellStructureView,
        _base: float,
        _factors: object,
    ) -> tuple[ASUStructure, list[str]]:
        frames.append((view.cell.basis, tuple(view.species_at_sites)))
        if len(frames) == 1:
            return canonical_module._exact_p1(view), []
        return _nacl(), []

    monkeypatch.setattr(canonical_module, "_recognition_sweep", staged_sweep)
    result = canonical_asu(UnitcellStructureView(_nacl()))

    assert result.spacegroup.it_number == 225
    assert len(frames) == 2
    assert frames[1][0] == frames[0][0]
    assert frames[1][1] == tuple(reversed(frames[0][1]))


def test_result_is_deterministic() -> None:
    pytest.importorskip("spglib")
    noisy = _perturbed(UnitcellStructure(*_expanded(_nacl())), 400_000)
    first = canonical_asu(noisy)
    second = canonical_asu(noisy)
    assert _site_key(first) == _site_key(second)
    assert first.cell.basis == second.cell.basis


def test_asu_input_matches_the_expanded_path() -> None:
    pytest.importorskip("spglib")
    from_asu = canonical_asu(_nacl())
    from_view = canonical_asu(UnitcellStructureView(_nacl()))
    assert _site_key(from_asu) == _site_key(from_view)
    assert from_asu.cell.basis == from_view.cell.basis


def _p4332() -> ASUStructure:
    # SG 213 (P4_3 32), the HIGHER member of the enantiomorphic 212/213 pair.
    return ASUStructure(
        Cell(((7, 0, 0), (0, 7, 0), (0, 0, 7))),
        213,
        [WyckoffSite("c", FracVector((F(1, 13),)), "Si")],
        _species("Si"),
    )


def test_enantiomorph_normalizes_to_the_lower_member_by_default() -> None:
    pytest.importorskip("spglib")
    view = UnitcellStructureView(_p4332())
    # A genuinely chiral cell recognized in the higher member (213) is normalized to the lower one.
    assert canonical_asu(view).spacegroup.it_number == 212
    assert canonical_asu(view, preserve_chirality=True).spacegroup.it_number == 213
    # Robust to sub-tolerance noise: recognition still lands in the pair, normalization still fires.
    noisy = _perturbed(UnitcellStructure(*_expanded(_p4332())), 400_000)
    assert canonical_asu(noisy).spacegroup.it_number == 212
    assert canonical_asu(noisy, preserve_chirality=True).spacegroup.it_number == 213


def test_structure_api_canonical_proto_values_collapse_enantiomorphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("spglib")
    real_canonical_asu = canonical_module.canonical_asu
    calls: list[bool] = []

    def tracking_canonical_asu(*args, preserve_chirality=True, **kwargs):
        calls.append(preserve_chirality)
        return real_canonical_asu(*args, preserve_chirality=preserve_chirality, **kwargs)

    monkeypatch.setattr(canonical_module, "canonical_asu", tracking_canonical_asu)
    structure = UnitcellStructureView(_p4332())

    protostructure = structure.canonical_protostructure()
    prototype = structure.canonical_prototype()

    assert type(protostructure) is Protostructure
    assert type(prototype) is Prototype
    assert protostructure.spacegroup.it_number == 212
    assert prototype.spacegroup.it_number == 212
    assert calls == [False, False]


def test_missing_spglib_raises_the_recognition_import_error() -> None:
    import sys

    view = UnitcellStructureView(_nacl())
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "spglib", None)
        with pytest.raises(ImportError, match=r"spglib.*httk-atomistic\[default\]"):
            canonical_asu(view)


def _expanded(asu: ASUStructure) -> tuple:
    """Return the (cell, coords, species, species_at_sites) of an expanded ASU, for perturbing."""
    view = UnitcellStructureView(asu)
    return view.cell, view.sites.reduced_coords.to_fractions(), asu.species, list(view.species_at_sites)
