"""Default enantiomorph normalization: the higher member of a pair is flipped to the lower one.

Canonicalization maps a structure in the higher-numbered member of one of the 11 enantiomorphic
pairs to its lower-numbered partner by an exact chirality-flipping transformation (fractional
coordinates ``f -> (-f) mod 1`` with the cell basis unchanged), unless ``preserve_chirality=True`` or
the structure carries site moments.
"""

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
    canonicalize,
    same_crystal,
)
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.models.moments.collinear import CollinearSiteMoments
from httk.atomistic.symmetry.lift import (
    _canonical_entry,
    _canonical_without_bfs,
    _enantiomorph,
    _site_key,
)


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _p32() -> ASUStructure:
    # SG 145 (P3_2), the HIGHER member of the 144/145 pair, general-position site.
    cell = Cell(CellParams((5, 5, 12, 90, 90, 120)).basis)
    return ASUStructure(cell, 145, [WyckoffSite("a", FracVector((F(1, 7), F(2, 11), F(3, 13))), "Si")], _species("Si"))


def _p4332() -> ASUStructure:
    # SG 213 (P4_3 32), the HIGHER member of the 212/213 pair.
    cell = Cell(((7, 0, 0), (0, 7, 0), (0, 0, 7)))
    return ASUStructure(cell, 213, [WyckoffSite("c", FracVector((F(1, 13),)), "Si")], _species("Si"))


def _unitcell(structure: ASUStructure) -> UnitcellStructure:
    view = UnitcellStructureView(structure)
    return UnitcellStructure(view.cell, view.sites.reduced_coords, list(view.species), list(view.species_at_sites))


def _mirror(structure: ASUStructure) -> UnitcellStructure:
    """Return the exact Cartesian mirror ``f -> (-f) mod 1`` of a structure's expansion, same cell."""
    view = UnitcellStructureView(structure)
    coords = FracVector([[(-value) % 1 for value in row] for row in view.sites.reduced_coords.to_fractions()])
    return UnitcellStructure(view.cell, coords, list(view.species), list(view.species_at_sites))


def test_higher_member_normalizes_to_lower_by_default() -> None:
    asu = _p32()
    assert _canonical_without_bfs(asu).spacegroup.it_number == 144
    assert _canonical_without_bfs(asu, preserve_chirality=True).spacegroup.it_number == 145


def test_flip_is_the_exact_mirror_of_the_entry() -> None:
    # _enantiomorph produces the exact mirror (basis unchanged, f -> -f) in the partner group, before
    # the group-specific normal form / orientation move the origin.
    entry = _canonical_entry(_p32())
    flipped = _enantiomorph(entry)
    assert flipped is not None
    assert flipped.spacegroup.it_number == 144
    assert same_crystal(_unitcell(flipped), _mirror(entry))
    # Genuinely chiral: the mirror is NOT the same crystal as the entry itself.
    assert not same_crystal(_unitcell(flipped), _unitcell(entry))


def test_default_normalization_is_idempotent() -> None:
    first = _canonical_without_bfs(_p32())
    second = _canonical_without_bfs(first)
    assert second.spacegroup.it_number == 144
    assert _site_key(first) == _site_key(second)
    assert first.cell.basis == second.cell.basis


def test_lower_member_input_is_unchanged_by_default() -> None:
    # Companion to test_left_handed_enantiomorphic_cell_is_handled_gracefully (which uses SG 144, the
    # LOWER member): a right-handed 144 input already sits in the canonical member and is not flipped.
    cell = Cell(CellParams((5, 5, 12, 90, 90, 120)).basis)
    asu = ASUStructure(cell, 144, [WyckoffSite("a", FracVector((F(1, 7), F(2, 11), F(3, 13))), "Si")], _species("Si"))
    assert _canonical_without_bfs(asu).spacegroup.it_number == 144
    assert _enantiomorph(_canonical_entry(asu)) is None


def test_magnetic_enantiomorph_is_not_flipped() -> None:
    cell = Cell(CellParams((5, 5, 12, 90, 90, 120)).basis)
    magnetic = ASUStructure(
        cell,
        145,
        [WyckoffSite("a", FracVector((F(1, 7), F(2, 11), F(3, 13))), "Si", moment=CollinearSiteMoments([1]))],
        _species("Si"),
    )
    # _enantiomorph refuses a moment-carrying structure directly (axial vectors are out of scope), and
    # the canonical path keeps the group rather than mirroring the magnetic structure.
    assert _enantiomorph(magnetic) is None
    assert _canonical_without_bfs(magnetic).spacegroup.it_number == 145


def test_full_canonicalize_normalizes_the_cubic_pair() -> None:
    asu = _p4332()
    assert canonicalize(asu).asu.spacegroup.it_number == 212
    assert canonicalize(asu, preserve_chirality=True).asu.spacegroup.it_number == 213


def test_terminal_flip_fires_on_a_nonempty_lift_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both other canonicalize tests enter directly at the higher member (path=()).  This pins the
    # terminal-emission flip for a terminal reached through >=1 hop, without paying the slow real BFS:
    # the lift table is stubbed so any entry lifts in one hop to the SG 213 fixture, which is terminal.
    import httk.atomistic.symmetry.lift as lift_module
    from httk.atomistic.symmetry.subgroups import subgroup_transforms

    higher = _p4332()  # SG 213 terminal, the higher member of the 212/213 pair
    hop = subgroup_transforms(213, 198)[0]  # a real tabulated transform -> a genuinely nonempty path

    def fake_lifts(state: ASUStructure, tolerance: float) -> list[object]:
        if state.spacegroup.it_number == 213:
            return []  # the higher member is the terminal
        return [lift_module.LiftResult(higher, higher.spacegroup, (hop,), FracVector((0, 0, 0)), F(0))]

    seen: dict[int, int | None] = {}
    real_enantiomorph = lift_module._enantiomorph

    def spy(structure: ASUStructure) -> ASUStructure | None:
        result = real_enantiomorph(structure)
        seen[structure.spacegroup.it_number] = None if result is None else result.spacegroup.it_number
        return result

    monkeypatch.setattr(lift_module, "_highest_lifts", fake_lifts)
    monkeypatch.setattr(lift_module, "_enantiomorph", spy)

    entry = ASUStructure(
        Cell(((6, 0, 0), (0, 6, 0), (0, 0, 6))),
        1,
        [WyckoffSite("a", FracVector((F(1, 7), F(1, 11), F(1, 13))), "Si")],
        _species("Si"),
    )
    result = canonicalize(entry)
    # _enantiomorph was invoked on the hopped 213 terminal and produced 212 (its result was used):
    assert seen.get(213) == 212
    assert result.asu.spacegroup.it_number == 212
    assert len(result.path) >= 1  # the terminal was reached through at least one lift hop


@pytest.mark.extended
def test_full_canonicalize_normalizes_the_trigonal_pair() -> None:
    # Full breadth-first canonicalize on a general-position P3_2 input (its failed-lift BFS is slow).
    asu = _p32()
    assert canonicalize(asu).asu.spacegroup.it_number == 144
    assert canonicalize(asu, preserve_chirality=True).asu.spacegroup.it_number == 145
