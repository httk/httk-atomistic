"""Tests for neutral mCIF payloads adapting to native magnetic structures."""

from fractions import Fraction
from pathlib import Path

import pytest
from httk.core import load

from httk.atomistic import ModulatedStructure, SymopsStructure, UnitcellStructureView
from httk.atomistic.mcif_structures import symops_structures_from_mcif

FIXTURES = Path(__file__).with_name("fixtures")


def _moment_rows(moments):
    values = moments.crystalaxis_moments if moments.kind == "crystalaxis" else moments.cartesian_moments
    return tuple(tuple(values._element((row, column)) for column in range(3)) for row in range(len(moments)))


def test_centered_mcif_loads_and_expands_exactly() -> None:
    structure = load(str(FIXTURES / "magnetic_centered.mcif"))

    assert isinstance(structure, SymopsStructure)
    assert _moment_rows(UnitcellStructureView(structure).site_moments) == ((1, 0, 0), (-1, 0, 0))
    assert len(UnitcellStructureView(structure).sites) == 2
    assert structure.bns_number is None
    assert structure.bns_label is None


def test_cartesian_mcif_preserves_source_kind_and_exact_values() -> None:
    structure = load(str(FIXTURES / "magnetic_cartesian_hexagonal.mcif"))

    assert isinstance(structure, SymopsStructure)
    assert structure.site_moments.kind == "cartesian"
    assert _moment_rows(UnitcellStructureView(structure).site_moments) == ((1, 2, 3),)


def test_incommensurate_block_loads_as_modulated_structure() -> None:
    payload = load(str(FIXTURES / "magnetic_kvector.mcif"), raw=True)
    block = dict(payload["blocks"][0])
    block["incomm"] = {
        "mod_dim": 1,
        "structural_q": None,
        "magnetic_q": ((Fraction(1, 8), 0, Fraction(1, 3)),),
    }
    structure = symops_structures_from_mcif({"format": "mcif", "blocks": [block]})[0]

    assert isinstance(structure, ModulatedStructure)
    assert structure.mod_dim == 1
    assert structure.magnetic_q == ((Fraction(1, 8), 0, Fraction(1, 3)),)
    with pytest.raises(ValueError, match="cannot be represented"):
        _ = UnitcellStructureView(structure).cell


def test_mcif_without_symops_is_rejected() -> None:
    block = {
        "format": "mcif",
        "cell_parameters_exact": ("1", "1", "1", "90", "90", "90"),
        "positions_exact": [("0", "0", "0")],
        "symbols": ["Fe"],
        "labels": ["Fe1"],
        "symops_xyz": (),
    }

    with pytest.raises(ValueError, match="mcif block.*symops_xyz"):
        symops_structures_from_mcif(block)


def test_mcif_type_symbols_preserve_oxidation_states() -> None:
    payload = load(str(FIXTURES / "magnetic_centered.mcif"), raw=True)
    block = dict(payload["blocks"][0])
    block["symbols"] = ["Fe2+"]
    structure = symops_structures_from_mcif({"format": "mcif", "blocks": [block]})[0]

    assert structure.species[0].chemical_symbols == ("Fe",)
    assert structure.species[0].charges == (2,)
