"""Tests for exact enumeration of a crystal's representations in a target group."""

from fractions import Fraction as F

import pytest
from httk.core import FracVector

from httk.atomistic import (
    ASUStructure,
    Cell,
    Species,
    UnitcellStructureView,
    WyckoffSite,
    backward_lift,
    build_supercell,
    canonicalize,
    canonicalize_full,
    list_representations,
    same_crystal,
    subgroup_representation,
)
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.symmetry.lift import rerepresent
from httk.atomistic.symmetry.subgroups import _standard_input


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _assignment(structure: ASUStructure) -> dict[str, str]:
    return {site.species: site.wyckoff for site in structure.wyckoff_sites}


def _key(structure: ASUStructure) -> tuple[object, ...]:
    metric = structure.cell.metric()
    return (
        tuple(sorted((s.species, s.wyckoff, tuple(s.free_params.to_fractions())) for s in structure.wyckoff_sites)),
        tuple(metric._element((r, c)) for r in range(3) for c in range(3)),
    )


def _cell_size(structure: ASUStructure) -> tuple[object, int]:
    return structure.cell.basis.det(), len(UnitcellStructureView(structure).sites)


def _same_crystal(representation: ASUStructure, source: ASUStructure) -> bool:
    # Representations of one crystal differ by origin/cell choice, so raw same_crystal (origin- and
    # cell-strict) is False between them; comparing their canonical (highest-symmetry) forms is the
    # origin-robust "same crystal" check.
    return same_crystal(canonicalize(representation).asu, canonicalize(source).asu)


def _same_crystal_same_size(representation: ASUStructure, reference: ASUStructure) -> bool:
    # "Same crystal at the same cell size": a supercell is the same crystal in a larger cell, so the
    # cheap cell-size gate rejects it before the canonicalize-based crystal check even runs.
    return _cell_size(representation) == _cell_size(reference) and _same_crystal(representation, reference)


def _zincblende() -> ASUStructure:
    return ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        216,
        [WyckoffSite("a", FracVector(()), "Zn"), WyckoffSite("c", FracVector(()), "S")],
        _species("Zn", "S"),
    )


def _rocksalt() -> ASUStructure:
    return ASUStructure(
        Cell(((5, 0, 0), (0, 5, 0), (0, 0, 5))),
        225,
        [WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("b", FracVector(()), "Cl")],
        _species("Na", "Cl"),
    )


def _p1(structure: ASUStructure) -> ASUStructure:
    view = UnitcellStructureView(structure)
    return ASUStructure(
        Cell(view.cell.basis),
        1,
        [
            WyckoffSite("a", FracVector(coordinate).normalize(), species)
            for coordinate, species in zip(view.sites.reduced_coords.to_fractions(), view.species_at_sites)
        ],
        _species(*sorted(set(view.species_at_sites))),
    )


def test_own_group_enumerates_both_letter_assignments() -> None:
    zincblende = _zincblende()
    representations = list_representations(zincblende, 216)
    assignments = [_assignment(r) for r in representations]
    # The discrete-translation pair from the phase-3 work: both letter assignments are present.
    assert {"Zn": "a", "S": "c"} in assignments
    assert {"Zn": "c", "S": "b"} in assignments
    assert all(r.spacegroup.it_number == 216 for r in representations)
    assert all(_same_crystal_same_size(r, zincblende) for r in representations)
    assert len({_key(r) for r in representations}) == len(representations)


def test_descent_representations_are_distinct_same_crystal_and_pinned() -> None:
    rocksalt = _rocksalt()
    representations = list_representations(rocksalt, 166)
    assert len(representations) == 2  # deterministic count, pinned
    assert all(r.spacegroup.it_number == 166 for r in representations)
    assert len({_key(r) for r in representations}) == len(representations)
    assert all(_same_crystal_same_size(r, representations[0]) for r in representations)
    assert _same_crystal(representations[0], rocksalt)


def test_ascent_returns_the_ground_truth_representation() -> None:
    rocksalt = _rocksalt()
    child = subgroup_representation(rocksalt, 12).asu  # the pinned GT C2/m image
    representations = list_representations(child, 166)
    assert {"Na": "a", "Cl": "b"} in [_assignment(r) for r in representations]
    assert all(r.spacegroup.it_number == 166 for r in representations)
    assert all(_same_crystal_same_size(r, representations[0]) for r in representations)
    assert _same_crystal(representations[0], rocksalt)


def test_size_proxy_rejects_a_same_crystal_supercell() -> None:
    # A doubled cell is the same crystal in a larger cell -- not a distinct representation at the same
    # cell size.  The strengthened proxy must reject it on size before its crystal identity matters.
    rocksalt = _rocksalt()
    minimal = list_representations(rocksalt, 12)[0]
    supercell = _p1(build_supercell(minimal, ((2, 0, 0), (0, 1, 0), (0, 0, 1))).structure)
    minimal_volume, minimal_count = _cell_size(minimal)
    assert _cell_size(supercell) == (minimal_volume * 2, minimal_count * 2)  # a genuine 2x supercell
    assert not _same_crystal_same_size(supercell, minimal)


def test_inequivalent_subgroup_embedding_is_out_of_scope() -> None:
    # Completeness boundary (documented scope): a representation from another descent chain
    # (225 -> 139 -> 69 -> 12) is the same crystal at the same cell size, yet is NOT returned --
    # list_representations enumerates the normalizer orbit of ONE canonical embedding, not the
    # inequivalent embeddings from other chains.  Any future widening must consciously update this.
    rocksalt = _rocksalt()
    representations = list_representations(rocksalt, 12)
    minimal = representations[0]
    other_chain = _standard_input(rerepresent(rerepresent(rerepresent(rocksalt, 139), 69), 12))
    assert _same_crystal_same_size(other_chain, minimal)
    assert _key(other_chain) not in {_key(r) for r in representations}


def test_canonicalize_full_is_least_deterministic_and_idempotent() -> None:
    rocksalt = _rocksalt()
    representations = list_representations(rocksalt, 166)
    least = canonicalize_full(rocksalt, 166)
    assert _key(least) == min(_key(r) for r in representations)
    assert _key(canonicalize_full(rocksalt, 166)) == _key(least)  # determinism across runs
    assert _key(canonicalize_full(least, 166)) == _key(least)  # idempotence


def test_canonicalize_full_on_own_group_matches_the_normal_form_pick() -> None:
    from httk.atomistic.symmetry.lift import _normal_form, _site_key

    rocksalt = _rocksalt()
    # On its own group the least representation is the normalizer-canonical one the upward search's
    # normal form selects, modulo the continuous quotient.
    assert _site_key(canonicalize_full(rocksalt, 225)) == _site_key(_normal_form(_standard_input(rocksalt)))


def test_enantiomorphic_group_emits_only_right_handed_representations() -> None:
    # A left-handed normalizer image of a Sohncke group is the enantiomorph -- a different crystal --
    # so _representation_orbit drops it.  Every emitted representation is right-handed, and re-listing
    # any of them reproduces the same orbit (same crystal), without the slow trigonal canonicalize.
    structure = ASUStructure(
        Cell(CellParams((4, 4, 6, 90, 90, 120)).basis),
        152,
        [WyckoffSite("c", FracVector((F(1, 7), F(2, 7), F(3, 7))), "Si")],
        _species("Si"),
    )
    representations = list_representations(structure, 152)
    assert representations
    assert all(r.cell.basis.det().sign() > 0 for r in representations)
    orbit = {_key(r) for r in representations}
    assert {_key(r) for r in list_representations(representations[0], 152)} == orbit


def test_all_lifts_of_a_hop_produce_one_orbit() -> None:
    # Subsumption: enumerating the normalizer orbit of ONE round-trip-valid lift equals the union over
    # every lift's orbit.  The 216 -> 160 descent lifts back to 216 in four ways; all four give the
    # same orbit.
    zincblende = _zincblende()
    child = subgroup_representation(zincblende, 160).asu
    lifts = backward_lift(child, 216, tolerance=1e-3)
    assert len(lifts) > 1
    orbits = [frozenset(_key(r) for r in list_representations(_standard_input(lift.asu), 216)) for lift in lifts]
    assert all(orbit == orbits[0] for orbit in orbits)


def test_unrelated_target_raises() -> None:
    with pytest.raises(ValueError):
        list_representations(_rocksalt(), 191)
    with pytest.raises(ValueError):
        canonicalize_full(_rocksalt(), 191)
