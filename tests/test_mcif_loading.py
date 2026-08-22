"""Tests for neutral mCIF payloads adapting to native magnetic structures."""

from fractions import Fraction
from pathlib import Path

import pytest
from httk.core import load

from httk.atomistic import ModulatedStructure, SymopsStructure, UnitcellStructureView
from httk.atomistic.mcif_structures import (
    _perfect_orbit_matching,
    _spatial_structure_from_mcif,
    symops_structures_from_mcif,
)

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


def test_mcif_atom_type_mass_reaches_the_species(tmp_path: Path) -> None:
    path = tmp_path / "mass.mcif"
    path.write_text(
        """data_mass
_cell_length_a 5
_cell_length_b 5
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_space_group_symop_magn_operation.xyz
'x,y,z,+1'
loop_
_atom_type_symbol
_atom_type_mass
Fe 55.8
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe1 Fe 0 0 0
""",
        encoding="utf-8",
    )

    structure = load(path)

    assert structure.species[0].mass == (55.8,)


def test_decimal_moments_are_converted_to_exact_rationals_before_expansion() -> None:
    block = {
        "format": "mcif",
        "cell_parameters_exact": ("1", "1", "1", "90", "90", "90"),
        "positions_exact": [("1/2", "1/2", "1/2")],
        "symbols": ["Gd"],
        "labels": ["Gd1"],
        "symops_xyz": ("x,y,z,+1", "-y,-x,-z,+1"),
        "moment_basis": "crystalaxis",
        "magmoms_exact": (("4.78", "-4.78", "0"),),
    }

    structure = symops_structures_from_mcif(block)[0]

    assert _moment_rows(UnitcellStructureView(structure).site_moments) == ((Fraction(239, 50), Fraction(-239, 50), 0),)


def test_mcif_preserves_component_resolution_esd_and_symmetry_form(tmp_path: Path) -> None:
    path = tmp_path / "moment-metadata.mcif"
    path.write_text(
        """data_moment_metadata
_cell_length_a 5
_cell_length_b 5
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_space_group_symop_magn_operation.xyz
'x,y,z,+1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe1 Fe 0 0 0
loop_
_atom_site_moment.label
_atom_site_moment.crystalaxis_x
_atom_site_moment.crystalaxis_y
_atom_site_moment.crystalaxis_z
_atom_site_moment.symmform
Fe1 -0.159(9) -0.319(18) 0.0 mx,2mx,0
""",
        encoding="utf-8",
    )

    structure = load(path)

    assert structure.moment_component_resolutions == ((Fraction(1, 1000), Fraction(1, 1000), None),)
    assert structure.moment_component_esds == ((Fraction(9, 1000), Fraction(18, 1000), None),)
    assert structure.moment_symmforms == ("mx,2mx,0",)
    assert structure.listed_site_moments.precision == Fraction(18, 1000)


def test_spatial_projection_deduplicates_symmetry_images_within_source_precision() -> None:
    block = {
        "format": "mcif",
        "cell_parameters_exact": ("5", "5", "5", "90", "90", "90"),
        "positions_exact": [("0", "0", "0")],
        "coordinate_precision": Fraction(1, 1_000_000),
        "symbols": ["Fe"],
        "labels": ["Fe1"],
        "symops_xyz": ("x+1/3,y,z,+1", "x+0.333333,y,z,+1"),
    }
    structure = symops_structures_from_mcif(block)[0]

    projected = _spatial_structure_from_mcif(structure)

    assert len(projected.sites) == 1


def test_spatial_disorder_projection_is_independent_of_source_row_order() -> None:
    block = {
        "format": "mcif",
        "cell_parameters_exact": ("1", "1", "1", "90", "90", "90"),
        "positions_exact": [("0.001", "0", "0"), ("0.002", "0", "0")],
        "coordinate_precision": Fraction(1, 1000),
        "occupancies_exact": ("0.5", "0.5"),
        "symbols": ["Fe", "Mn"],
        "labels": ["Fe1", "Mn1"],
        "symops_xyz": ("x,y,z,+1",),
    }
    forward = _spatial_structure_from_mcif(symops_structures_from_mcif(block)[0])
    reversed_block = {
        **block,
        "positions_exact": list(reversed(block["positions_exact"])),
        "occupancies_exact": tuple(reversed(block["occupancies_exact"])),
        "symbols": list(reversed(block["symbols"])),
        "labels": list(reversed(block["labels"])),
    }
    backward = _spatial_structure_from_mcif(symops_structures_from_mcif(reversed_block)[0])

    assert forward.sites.reduced_coords == backward.sites.reduced_coords
    assert forward.sites.reduced_coords.to_fractions() == [[Fraction(3, 2000), 0, 0]]
    assert forward.species == backward.species


def test_spatial_orbit_proximity_requires_a_bijective_matching() -> None:
    first = frozenset(((Fraction(0), 0, 0), (Fraction(1, 1000), 0, 0)))
    second = frozenset(((Fraction(0), 0, 0), (Fraction(1, 10), 0, 0)))

    matching = _perfect_orbit_matching(
        first,
        second,
        lambda left, right: abs(left[0] - right[0]) <= Fraction(1, 100),
    )

    assert matching is None


def test_mcif_repair_clamps_refined_occupancy_above_one(caplog: pytest.LogCaptureFixture) -> None:
    block = {
        "format": "mcif",
        "cell_parameters_exact": ("5", "5", "5", "90", "90", "90"),
        "positions_exact": [("0", "0", "0")],
        "occupancies_exact": ("1.013",),
        "occupancy_precisions": (Fraction(1, 1000),),
        "symbols": ["O"],
        "labels": ["O1"],
        "symops_xyz": ("x,y,z,+1",),
    }

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        symops_structures_from_mcif(block)
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        structure = symops_structures_from_mcif({**block, "repair": True})[0]
        wrapped = symops_structures_from_mcif({"format": "mcif", "blocks": [{**block, "repair": True}]})[0]

    assert structure.species[0].concentration == (1,)
    assert wrapped.species[0].concentration == (1,)
    assert any("clamped site 'O1' occupancy" in record.getMessage() for record in caplog.records)


def test_cell_parameter_lengths_remain_exact_during_moment_expansion() -> None:
    block = {
        "format": "mcif",
        "cell_parameters_exact": ("8.54420", "8.54420", "8.54420", "89.99", "89.99", "89.99"),
        "positions_exact": [("0", "0", "0")],
        "symbols": ["Fe"],
        "labels": ["Fe1"],
        "symops_xyz": ("x,y,z,+1", "z,x,y,+1"),
        "moment_basis": "crystalaxis",
        "magmoms_exact": (("1.915", "1.915", "1.915"),),
    }

    structure = symops_structures_from_mcif(block)[0]

    assert tuple(value._rational_fraction() for value in structure.cell.lengths) == (Fraction("8.54420"),) * 3
    assert _moment_rows(UnitcellStructureView(structure).site_moments) == (
        (Fraction("1.915"), Fraction("1.915"), Fraction("1.915")),
    )


def test_repaired_spatial_disorder_normalizes_occupancy_and_drops_partial_masses(
    caplog: pytest.LogCaptureFixture,
) -> None:
    block = {
        "format": "mcif",
        "cell_parameters_exact": ("5", "5", "5", "90", "90", "90"),
        "positions_exact": [("0", "0", "0"), ("0", "0", "0")],
        "occupancies_exact": ("0.6002", "0.4001"),
        "occupancy_precisions": (Fraction(1, 10_000), Fraction(1, 10_000)),
        "symbols": ["Fe", "Mn"],
        "labels": ["Fe1", "Mn1"],
        "masses": [55.845, None],
        "symops_xyz": ("x,y,z,+1",),
    }
    structure = symops_structures_from_mcif(block)[0]

    with pytest.raises(ValueError, match="masses for only some constituents"):
        _spatial_structure_from_mcif(structure)
    with caplog.at_level("WARNING", logger="httk.atomistic.cif_structures"):
        projected = _spatial_structure_from_mcif(structure, repair=True)

    species = projected.species[0]
    assert sum(species.concentration) == 1
    assert species.mass is None
    messages = [record.getMessage() for record in caplog.records]
    assert any("omitted partially declared constituent masses" in message for message in messages)
    assert any("normalized co-located-site occupancies" in message for message in messages)


def test_repaired_spatial_disorder_rejects_gross_overoccupancy() -> None:
    block = {
        "format": "mcif",
        "cell_parameters_exact": ("5", "5", "5", "90", "90", "90"),
        "positions_exact": [("0", "0", "0"), ("0", "0", "0")],
        "occupancies_exact": ("1", "1"),
        "symbols": ["Fe", "Mn"],
        "labels": ["Fe1", "Mn1"],
        "symops_xyz": ("x,y,z,+1",),
    }
    structure = symops_structures_from_mcif(block)[0]

    with pytest.raises(ValueError, match="occupancies sum to 2"):
        _spatial_structure_from_mcif(structure, repair=True)
