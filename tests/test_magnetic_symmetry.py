"""Round-trip tests for magnetic symmetry finding.

The authoritative check is that expanding ``find_magnetic_symmetry(s)`` back to a full cell
reproduces ``s`` — same sites, species, and per-site moments, within the tolerance and up
to reordering.
"""

import fractions
import math
import re
import sys

import pytest

from httk.atomistic import (
    CartesianSiteMoments,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    find_magnetic_symmetry,
    same_crystal,
)

spglib = pytest.importorskip("spglib")

F = fractions.Fraction


def _fe() -> tuple[Species]:
    return (Species("Fe", ("Fe",), (1,)),)


def _hexagonal(a: float, c: float) -> list[list[float]]:
    return [[a, 0, 0], [-a / 2, a * math.sqrt(3) / 2, 0], [0, 0, c]]


def _assert_round_trips(structure: UnitcellStructure, tolerance: float = 1e-3) -> None:
    """Expanding the found symmetry reproduces the input sites, species, and moments."""
    result = find_magnetic_symmetry(structure, tolerance)
    view = UnitcellStructureView(result)

    assert same_crystal(view, structure), "nuclear part must round-trip"

    input_coords = [[float(value) for value in row] for row in structure.sites.reduced_coords.to_fractions()]
    input_moments = structure.site_moments.cartesian_moments.to_floats()
    output_coords = [[float(value) for value in row] for row in view.sites.reduced_coords.to_fractions()]
    output_moments = view.site_moments.cartesian_moments.to_floats()
    assert len(output_coords) == len(input_coords)

    for coord, moment in zip(input_coords, input_moments):
        matched = any(
            all(abs((left - right) - round(left - right)) < tolerance for left, right in zip(coord, other_coord))
            and all(abs(left - right) < tolerance for left, right in zip(moment, other_moment))
            for other_coord, other_moment in zip(output_coords, output_moments)
        )
        assert matched, f"no expanded site matches input {coord} with moment {moment}"


def _spglib_operation_count(structure: UnitcellStructure, tolerance: float = 1e-3) -> int:
    """Operation count of spglib's axial (vector) magnetic search -- the group the finder reports."""
    names = sorted(set(structure.species_at_sites))
    dataset = spglib.get_magnetic_symmetry_dataset(
        (
            structure.cell.basis.to_floats(),
            structure.sites.reduced_coords.to_floats(),
            [names.index(name) + 1 for name in structure.species_at_sites],
            [list(row) for row in structure.site_moments.cartesian_moments.to_floats()],
        ),
        symprec=tolerance,
    )
    return int(dataset.n_operations)


def test_collinear_antiferromagnet_detects_symmetry_and_round_trips() -> None:
    afm = UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 5]],
        [[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]],
        _fe(),
        ("Fe", "Fe"),
        site_moments=CartesianSiteMoments([[0, 0, 3], [0, 0, -3]]),
    )
    result = find_magnetic_symmetry(afm)

    assert len(result.symops) > 1
    assert re.fullmatch(r"\d+\.\d+", result.bns_number or "")
    assert result.bns_label is None
    _assert_round_trips(afm)


def test_ferromagnet_round_trips() -> None:
    ferromagnet = UnitcellStructure(
        [[3, 0, 0], [0, 4, 0], [0, 0, 5]],
        [[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]],
        _fe(),
        ("Fe", "Fe"),
        site_moments=CartesianSiteMoments([[0, 0, 3], [0, 0, 3]]),
    )
    result = find_magnetic_symmetry(ferromagnet)

    assert len(result.symops) > 1
    _assert_round_trips(ferromagnet)


def test_hexagonal_crsb_like_cell_detects_symmetry_and_round_trips() -> None:
    species = (Species("Cr", ("Cr",), (1,)), Species("Sb", ("Sb",), (1,)))
    crsb = UnitcellStructure(
        _hexagonal(4.1, 5.5),
        [[0, 0, 0], [0, 0, F(1, 2)], [F(1, 3), F(2, 3), F(1, 4)], [F(2, 3), F(1, 3), F(3, 4)]],
        species,
        ("Cr", "Cr", "Sb", "Sb"),
        site_moments=CartesianSiteMoments([[0, 0, 2], [0, 0, -2], [0, 0, 0], [0, 0, 0]]),
    )
    result = find_magnetic_symmetry(crsb)

    assert len(result.symops) > 1
    # The listed representatives are one Cr and one Sb; both must survive expansion.
    assert set(result.listed_species_at_sites) == {"Cr", "Sb"}
    _assert_round_trips(crsb)


def test_moments_breaking_all_symmetry_give_identity_only() -> None:
    broken = UnitcellStructure(
        [[3, 0, 0], [0, 4.1, 0], [0.2, 0.3, 5]],
        [[0, 0, 0], [F(3, 10), F(1, 10), F(1, 5)]],
        _fe(),
        ("Fe", "Fe"),
        site_moments=CartesianSiteMoments([[0, 0, 1], [0, 0, 2]]),
    )
    result = find_magnetic_symmetry(broken)

    assert len(result.symops) == 1
    operation, time_reversal = result.symops[0]
    assert operation.is_identity()
    assert time_reversal == 1
    # Every site is its own orbit, so both are listed and expansion is the identity.
    assert len(result.listed_sites) == 2
    assert _spglib_operation_count(broken) == 1
    _assert_round_trips(broken)


def test_reports_full_axial_magnetic_group() -> None:
    afm = UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 5]],
        [[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]],
        _fe(),
        ("Fe", "Fe"),
        site_moments=CartesianSiteMoments([[0, 0, 3], [0, 0, -3]]),
    )
    result = find_magnetic_symmetry(afm)

    # The finder reports the COMPLETE axial magnetic space group spglib's vector search finds
    # (not an index-2 subgroup), so the operation count and BNS number are mutually consistent
    # -- the moment-reversing operations are what let a single Fe representative regenerate both
    # the up and down sublattice sites under expansion.
    assert len(result.symops) == _spglib_operation_count(afm)
    _assert_round_trips(afm)


def test_near_collinear_moments_are_projected_not_forwarded() -> None:
    # A tiny in-plane component is real relaxed-DFT noise; forwarding it to spglib 2.7.0 can
    # segfault (an uncatchable process crash) or trip a mirror-stabilizer inconsistency at
    # expansion. The finder must project onto z (dropping the sub-tolerance noise), so a
    # near-collinear input yields exactly the same result as the perfectly collinear one.
    lattice = [[3, 0, 0], [0, 3, 0], [0, 0, 5]]
    sites = [[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]]
    noisy = UnitcellStructure(
        lattice, sites, _fe(), ("Fe", "Fe"),
        site_moments=CartesianSiteMoments([[5e-4, 0, 3], [0, -5e-4, -3]]),
    )
    clean = UnitcellStructure(
        lattice, sites, _fe(), ("Fe", "Fe"),
        site_moments=CartesianSiteMoments([[0, 0, 3], [0, 0, -3]]),
    )
    noisy_result = find_magnetic_symmetry(noisy)

    assert len(noisy_result.symops) > 1
    # The stored representative moments are projected onto z (no in-plane component survives).
    for row in noisy_result.listed_site_moments.cartesian_moments.to_floats():
        assert row[0] == 0 and row[1] == 0
    # Projecting onto z makes the noisy input equivalent to the clean one: same operation set.
    clean_result = find_magnetic_symmetry(clean)
    assert {(op.to_xyz(), tr) for op, tr in noisy_result.symops} == {
        (op.to_xyz(), tr) for op, tr in clean_result.symops
    }
    # And the clean input still round-trips exactly (moments matched).
    _assert_round_trips(clean)


def test_non_collinear_moment_is_rejected() -> None:
    structure = UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        [[0, 0, 0]],
        _fe(),
        ("Fe",),
        site_moments=CartesianSiteMoments([[1, 0, 3]]),
    )
    with pytest.raises(ValueError, match="collinear moments along z"):
        find_magnetic_symmetry(structure)


def test_missing_spglib_is_import_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    structure = UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        [[0, 0, 0]],
        _fe(),
        ("Fe",),
        site_moments=CartesianSiteMoments([[0, 0, 3]]),
    )
    monkeypatch.setitem(sys.modules, "spglib", None)
    with pytest.raises(ImportError, match="requires spglib"):
        find_magnetic_symmetry(structure)
