"""Tests for the explicit, lossy approximate CIF-writing opt-in.

CIF is exact by default: a cell with no exact CIF representation (an irrational or
orientation-losing basis) is refused unless the caller passes ``approximate=True`` to
``httk.core.save``. These tests pin the default refusal and the opt-in round-trip.
"""

import fractions

import pytest
from httk.core import load, save

from httk.atomistic import Cell, UnitcellStructure, UnitcellStructureView
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.sites.sites import Sites

F = fractions.Fraction

# A sheared, arbitrarily-oriented rational basis: length b/c are sqrt-surds with no exact CIF
# syntax, so the six-parameter form cannot round-trip to this exact basis (the altermagnets case).
_SHEARED = [[3, 0, 0], [1, 3, 0], [1, 1, 3]]
_ORTHO = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]


def _structure(basis: list[list[int]], *, moments: bool = False) -> UnitcellStructure:
    kwargs = {"site_moments": CartesianSiteMoments([[0, 0, 2], [0, 0, -2]])} if moments else {}
    return UnitcellStructure(
        Cell(basis),
        Sites([[F(0), F(0), F(0)], [F(1, 2), F(1, 2), F(1, 2)]]),
        None,
        ["Fe", "O"],
        **kwargs,
    )


def test_surd_basis_refuses_without_the_flag(tmp_path):
    with pytest.raises(ValueError, match="approximate=True"):
        save(_structure(_SHEARED), tmp_path / "a.cif", format="cif")


def test_surd_basis_serializes_with_the_flag_and_round_trips(tmp_path):
    source = _structure(_SHEARED)
    path = tmp_path / "b.cif"
    save(source, path, format="cif", approximate=True)

    reloaded = UnitcellStructureView(load(str(path)))
    assert len(reloaded.sites) == len(UnitcellStructureView(source).sites)
    assert sorted(reloaded.species_at_sites) == sorted(UnitcellStructureView(source).species_at_sites)

    original_lengths = [float(x.to_float()) for x in source.cell.lengths]
    original_angles = [float(x) for x in source.cell.angles]
    reloaded_lengths = [float(x.to_float()) for x in reloaded.cell.lengths]
    reloaded_angles = [float(x) for x in reloaded.cell.angles]
    assert max(abs(a - b) for a, b in zip(original_lengths, reloaded_lengths, strict=True)) < 1e-9
    assert max(abs(a - b) for a, b in zip(original_angles, reloaded_angles, strict=True)) < 1e-6


def test_exact_structure_is_unchanged_by_the_flag(tmp_path):
    exact_default = tmp_path / "e.cif"
    exact_approx = tmp_path / "e2.cif"
    save(_structure(_ORTHO), exact_default, format="cif")
    save(_structure(_ORTHO), exact_approx, format="cif", approximate=True)
    assert exact_default.read_text() == exact_approx.read_text()


def test_moments_do_not_break_the_approximate_path(tmp_path):
    # The ordinary CIF writer carries no site moments; the approximate path must still succeed.
    path = tmp_path / "m.cif"
    save(_structure(_SHEARED, moments=True), path, format="cif", approximate=True)
    assert path.read_text().count("Fe") >= 1
