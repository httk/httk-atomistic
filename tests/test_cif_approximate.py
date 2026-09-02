"""Tests for the default lossy CIF-writing behaviour and its strict opt-out.

Saving into CIF renders whatever the format can hold: a cell with no exact CIF representation
(an irrational or orientation-losing basis) is written as rounded decimals by default. Callers
who need the exact-or-nothing guarantee pass ``approximate=False`` to ``httk.core.save``. These
tests pin the default round-trip and the strict refusal.
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


def test_surd_basis_refuses_under_strict_opt_out(tmp_path):
    with pytest.raises(ValueError, match="approximate=False"):
        save(_structure(_SHEARED), tmp_path / "a.cif", format="cif", approximate=False)


def test_surd_basis_serializes_by_default_and_round_trips(tmp_path):
    source = _structure(_SHEARED)
    path = tmp_path / "b.cif"
    save(source, path, format="cif")

    reloaded = UnitcellStructureView(load(str(path)))
    assert len(reloaded.sites) == len(UnitcellStructureView(source).sites)
    assert sorted(reloaded.species_at_sites) == sorted(UnitcellStructureView(source).species_at_sites)

    original_lengths = [float(x.to_float()) for x in source.cell.lengths]
    original_angles = [float(x) for x in source.cell.angles]
    reloaded_lengths = [float(x.to_float()) for x in reloaded.cell.lengths]
    reloaded_angles = [float(x) for x in reloaded.cell.angles]
    assert max(abs(a - b) for a, b in zip(original_lengths, reloaded_lengths, strict=True)) < 1e-9
    assert max(abs(a - b) for a, b in zip(original_angles, reloaded_angles, strict=True)) < 1e-6


def test_exact_structure_is_identical_under_both_modes(tmp_path):
    default_path = tmp_path / "e.cif"
    strict_path = tmp_path / "e2.cif"
    save(_structure(_ORTHO), default_path, format="cif")
    save(_structure(_ORTHO), strict_path, format="cif", approximate=False)
    assert default_path.read_text() == strict_path.read_text()


def test_moments_do_not_break_the_default_path(tmp_path):
    # The ordinary CIF writer carries no site moments; the default lossy path must still succeed.
    path = tmp_path / "m.cif"
    save(_structure(_SHEARED, moments=True), path, format="cif")
    assert path.read_text().count("Fe") >= 1
