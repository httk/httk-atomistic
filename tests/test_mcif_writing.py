"""Round-trip tests for the mCIF magnetic-structure writer."""

from fractions import Fraction
from pathlib import Path

import pytest
from httk.core import load, save

from httk.atomistic import SymopsStructure, UnitcellStructure
from httk.atomistic.mcif_structures import symops_structures_from_mcif
from httk.atomistic.models.moments.crystalaxis import CrystalAxisSiteMoments
from httk.atomistic.models.moments.crystalaxis_view import CrystalAxisSiteMomentsView

FIXTURES = Path(__file__).with_name("fixtures")

# The commensurate, non-modulated fixtures the writer must round-trip; the incommensurate
# fixtures (magnetic_ssg, magnetic_kvector) load as ModulatedStructure and are out of scope.
COMMENSURATE_FIXTURES = (
    "magnetic_cartesian.mcif",
    "magnetic_cartesian_hexagonal.mcif",
    "magnetic_centered.mcif",
    "cif2_lists.mcif",
)


def _crystalaxis_floats(structure: SymopsStructure) -> list[list[float]] | None:
    """Return the listed crystal-axis moments as floats, for a frame-independent comparison."""
    moments = structure.listed_site_moments
    if moments is None:
        return None
    crystalaxis = CrystalAxisSiteMomentsView(moments, cell=structure.cell).crystalaxis_moments
    return [[crystalaxis._element((row, column)).to_float() for column in range(3)] for row in range(len(moments))]


def _symop_set(structure: SymopsStructure) -> set[tuple[str, int]]:
    return {(operation.to_xyz(), time_reversal) for operation, time_reversal in structure.symops}


@pytest.mark.parametrize("name", COMMENSURATE_FIXTURES)
def test_mcif_writer_round_trips_commensurate_fixtures(name: str, tmp_path: Path) -> None:
    original = load(str(FIXTURES / name))
    assert isinstance(original, SymopsStructure)

    destination = tmp_path / "roundtrip.mcif"
    save(original, str(destination))
    reloaded = load(str(destination))
    assert isinstance(reloaded, SymopsStructure)

    assert [str(length) for length in original.cell.lengths] == [str(length) for length in reloaded.cell.lengths]
    assert [str(angle) for angle in original.cell.angles] == [str(angle) for angle in reloaded.cell.angles]
    assert original.listed_species_at_sites == reloaded.listed_species_at_sites
    assert original.listed_sites.reduced_coords == reloaded.listed_sites.reduced_coords
    assert _symop_set(original) == _symop_set(reloaded)
    assert original.bns_number == reloaded.bns_number
    assert original.bns_label == reloaded.bns_label

    original_moments = _crystalaxis_floats(original)
    reloaded_moments = _crystalaxis_floats(reloaded)
    assert (original_moments is None) == (reloaded_moments is None)
    if original_moments is not None and reloaded_moments is not None:
        assert len(original_moments) == len(reloaded_moments)
        for source_row, target_row in zip(original_moments, reloaded_moments):
            for source, target in zip(source_row, target_row):
                # A crystal-axis component with no exact CIF decimal (a Cartesian source moment
                # projected onto an oblique cell) round-trips only within its rounded precision.
                assert abs(source - target) < 1e-6


def test_mcif_writer_round_trips_two_same_element_sites(tmp_path: Path) -> None:
    # Two full-occupancy Fe sites both name themselves "Fe"; without unique moment labels the
    # mCIF reader rejects the duplicate _atom_site_moment.label and drops the whole block.
    cell = ("4", "4", "4", "90", "90", "90")
    moments = CrystalAxisSiteMoments(((3, 0, 0), (-3, 0, 0)), cell)
    structure = SymopsStructure(
        cell,
        [("0", "0", "0"), ("1/2", "1/2", "1/2")],
        [{"name": "Fe", "chemical_symbols": ("Fe",), "concentration": (1,)}],
        ["Fe", "Fe"],
        ["x,y,z,+1"],
        site_moments=moments,
    )

    destination = tmp_path / "two-fe.mcif"
    save(structure, str(destination))
    reloaded = load(str(destination))

    assert isinstance(reloaded, SymopsStructure)
    assert len(reloaded.listed_sites) == 2
    assert structure.listed_sites.reduced_coords == reloaded.listed_sites.reduced_coords
    original_moments = _crystalaxis_floats(structure)
    reloaded_moments = _crystalaxis_floats(reloaded)
    assert original_moments is not None and reloaded_moments is not None
    for source_row, target_row in zip(original_moments, reloaded_moments):
        for source, target in zip(source_row, target_row):
            assert abs(source - target) < 1e-6


def test_mcif_writer_accepts_exact_cell_with_approximate_false(tmp_path: Path) -> None:
    # An exactly representable cell must not crash on the strict flag (the flag must not be
    # forwarded into the low-level write_cif).
    structure = load(str(FIXTURES / "magnetic_centered.mcif"))
    destination = tmp_path / "strict.mcif"
    save(structure, str(destination), approximate=False)
    assert isinstance(load(str(destination)), SymopsStructure)


def test_mcif_writer_default_omits_exact_companions(tmp_path: Path) -> None:
    structure = load(str(FIXTURES / "magnetic_centered.mcif"))
    destination = tmp_path / "no-companions.mcif"
    save(structure, str(destination))
    assert "_httk_" not in destination.read_text(encoding="utf-8")


def test_mcif_writer_rejects_bare_unit_cell(tmp_path: Path) -> None:
    structure = UnitcellStructure(
        ("1", "1", "1", "90", "90", "90"),
        [("0", "0", "0")],
        [{"name": "Fe", "chemical_symbols": ("Fe",), "concentration": (1,)}],
        ["Fe"],
    )
    with pytest.raises((TypeError, ValueError), match="SymopsStructure"):
        save(structure, str(tmp_path / "bare.mcif"))


def test_mcif_writer_rejects_modulated_structure(tmp_path: Path) -> None:
    payload = load(str(FIXTURES / "magnetic_kvector.mcif"), raw=True)
    block = dict(payload["blocks"][0])
    block["incomm"] = {
        "mod_dim": 1,
        "structural_q": None,
        "magnetic_q": ((Fraction(1, 8), 0, Fraction(1, 3)),),
    }
    modulated = symops_structures_from_mcif({"format": "mcif", "blocks": [block]})[0]
    with pytest.raises((TypeError, ValueError), match="SymopsStructure"):
        save(modulated, str(tmp_path / "modulated.mcif"))
