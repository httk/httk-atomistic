"""Tests for :func:`canonical_asu`: tolerant recognition composed with exact canonicalization."""

from fractions import Fraction as F

import pytest
from httk.core import FracVector

from httk.atomistic import (
    ASUStructure,
    Cell,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
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


def test_rutile_free_parameter_is_a_least_squares_fit_near_the_reference() -> None:
    pytest.importorskip("spglib")
    reference = canonical_asu(UnitcellStructureView(_rutile()))
    noisy = canonical_asu(_perturbed(UnitcellStructure(*_expanded(_rutile())), 400_000))
    # Same group, setting and Wyckoff multiset; the free parameter is a least-squares fit of the
    # measured coordinates, so it lands NEAR the reference value but is not exactly equal.
    assert noisy.spacegroup.it_number == reference.spacegroup.it_number == 136

    def multiset(asu: ASUStructure) -> list[tuple[str, str]]:
        return sorted((s.species, s.wyckoff) for s in asu.wyckoff_sites)

    assert multiset(noisy) == multiset(reference)
    reference_x = next(s.free_params.to_fractions()[0] for s in reference.wyckoff_sites if s.species == "O")
    noisy_x = next(s.free_params.to_fractions()[0] for s in noisy.wyckoff_sites if s.species == "O")
    assert reference_x != noisy_x
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


def test_symprec_sweep_rescues_a_tolerance_boundary_flip() -> None:
    pytest.importorskip("spglib")
    # NaCl shifted +1e-4 on every coordinate: recognition fails at the tight symprec (base/5) but
    # returns IT 225 at the base symprec, so the full sweep rescues what the tight member cannot.
    cell, coords, species, species_at = _expanded(_nacl())
    shifted = [[value + F(1, 10000) for value in row] for row in coords]
    noisy = UnitcellStructure(cell, shifted, species, species_at)
    assert canonical_asu(noisy, factors=(F(1, 5), 1, 5)).spacegroup.it_number == 225
    with pytest.raises(ValueError, match="within tolerance"):
        canonical_asu(noisy, factors=(F(1, 5),))


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
