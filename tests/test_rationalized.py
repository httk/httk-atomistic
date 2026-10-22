from fractions import Fraction

import pytest
from httk.core import load, unwrap

from httk.atomistic import (
    CartesianSiteMoments,
    CartesianSites,
    Cell,
    CellView,
    CollinearSiteMoments,
    CrystalAxisSiteMoments,
    Sites,
    UnitcellStructure,
    UnitcellStructureView,
    same_crystal,
)
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.models.structure.unitcell import _snap


def _poscar(tmp_path, *rows: str) -> UnitcellStructure:
    path = tmp_path / "POSCAR"
    body = "\n".join(rows)
    path.write_text(f"x\n1.0\n3.0 0 0\n0 3.0 0\n0 0 3.0\nSi\n{len(rows)}\nDirect\n{body}\n", encoding="utf-8")
    return load(str(path))


def _coords(structure) -> list[list[Fraction]]:
    return structure.sites.reduced_coords.to_fractions()


def test_four_digit_coordinates_snap_to_thirds(tmp_path):
    structure = _poscar(tmp_path, "0.3333 0.6667 0.0")
    assert structure.coordinate_precision == Fraction(1, 10000)
    rational = structure.rationalized()
    assert _coords(rational) == [[Fraction(1, 3), Fraction(2, 3), 0]]
    assert rational.coordinate_precision == structure.coordinate_precision
    assert type(rational.sites) is Sites


def test_full_width_poscar_tokens_recover_non_terminating_rationals(tmp_path):
    # A POSCAR write renders 1/3 as 16 places; rationalized() recovers it, while a terminating
    # value written at full width has no smaller-denominator neighbour within 5e-17 and stays.
    structure = _poscar(tmp_path, "0.3333333333333333 0.1234560000000000 0.5000000000000000")
    assert _coords(structure.rationalized()) == [[Fraction(1, 3), Fraction("0.123456"), Fraction(1, 2)]]


def test_six_digit_coordinate_stays_exactly(tmp_path):
    # Six written digits give a 5e-7 tolerance; 0.123456 is simpler than 1189/9631 inside it.
    structure = _poscar(tmp_path, "0.123456 0.5 0.0")
    assert _coords(structure.rationalized())[0][0] == Fraction("0.123456")


@pytest.mark.parametrize(
    ("value", "tolerance", "expected"),
    [
        ("0.3333", Fraction(1, 20000), Fraction(1, 3)),
        ("0.6667", Fraction(1, 20000), Fraction(2, 3)),
        ("0.1429", Fraction(1, 20000), Fraction(1, 7)),  # rational simpler than the decimal 0.1429
        ("0.001001", Fraction(2, 10**9), Fraction(1, 999)),  # leading zeros are written digits too
        ("0.3549999999999969", Fraction(1, 10000), Fraction("0.355")),  # decimal beats 49/138
        ("5.5679999999999996", Fraction(1, 10000), Fraction("5.568")),
        ("0.123456", Fraction(1, 2_000_000), Fraction("0.123456")),
        ("0.3333333333333333", Fraction(1, 2 * 10**16), Fraction(1, 3)),
        ("0.5", Fraction(1, 10000), Fraction(1, 2)),
        ("-0.0000000000000003", Fraction(1, 10000), Fraction(0)),
    ],
)
def test_snap_takes_the_simpler_of_rational_and_decimal(value, tolerance, expected):
    assert _snap(Fraction(value), tolerance) == expected


def test_explicit_tolerance_overrides_the_precision(tmp_path):
    structure = _poscar(tmp_path, "0.3333333333333333 0.1234560000000000 0.5000000000000000")
    rational = structure.rationalized(coordinate_tolerance=Fraction(1, 500))
    assert _coords(rational)[0][1] == Fraction(1, 8)  # 0.125 lies within 2e-3 of 0.123456
    with pytest.raises(ValueError):
        structure.rationalized(coordinate_tolerance=0)


def test_without_precision_or_tolerance_nothing_changes():
    cell = Cell([["3.00001", 0, 0], [0, 3, 0], [0, 0, 3]])
    structure = UnitcellStructure(cell, Sites([["0.3333", 0, 0]]), species_at_sites=["Si"])
    rational = structure.rationalized()
    assert rational == structure
    assert rational.cell is structure.cell
    assert rational.sites.reduced_coords == structure.sites.reduced_coords


def test_cartesian_sites_come_back_as_plain_sites_when_only_the_cell_snaps():
    cell = Cell([["5.6401", 0, 0], [0, "5.64", 0], [0, 0, "5.64"]], precision=Fraction(1, 1000))
    sites = CartesianSites([["1.41", 0, 0]], cell.basis)
    structure = UnitcellStructure(cell, sites, species_at_sites=["Si"])
    rational = structure.rationalized(basis_tolerance="0.001")
    assert rational.cell.basis == Cell([[Fraction(282, 50), 0, 0], [0, "5.64", 0], [0, 0, "5.64"]]).basis
    assert type(unwrap(rational.sites)) is not CartesianSites
    assert rational.sites.reduced_coords == structure.sites.reduced_coords


def test_hexagonal_parameter_cell_snaps_lengths_and_keeps_exact_angles():
    cell = CellView(["3.00001", "3.00001", "5.00001", 90, 90, 120])
    coords = Sites([["0.3333", "0.6667", "0.25"]], precision=Fraction(1, 10000))
    structure = UnitcellStructure(cell, coords, species_at_sites=["Si"])
    rational = structure.rationalized(basis_tolerance=Fraction(1, 10000))
    assert type(rational.cell._backend) is CellParams
    assert rational.cell.lengths == (3, 3, 5)
    assert rational.cell.angles == (90, 90, 120)
    assert rational.cell.periodicity == structure.cell.periodicity  # CellParams cells are always 3D
    assert rational.cell.basis == CellParams((3, 3, 5, 90, 90, 120)).basis
    assert not rational.cell.basis.is_rational  # sqrt(3) kept, not rounded away
    assert rational.basis_precision is None
    sites = Sites([[Fraction(1, 3), Fraction(2, 3), Fraction(1, 4)]])
    assert same_crystal(rational, UnitcellStructure(CellView([3, 3, 5, 90, 90, 120]), sites, species_at_sites=["Si"]))


def test_cif_unit_cell_view_rationalizes_through_its_parameters(tmp_path):
    path = tmp_path / "x.cif"
    path.write_text(
        "data_x\n_cell_length_a 3.2100\n_cell_length_b 3.2100\n_cell_length_c 5.2000\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 120\n"
        "_space_group_IT_number 1\nloop_\n_space_group_symop_operation_xyz\n'x,y,z'\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n"
        "_atom_site_fract_z\nSi Si 0.3333 0.6667 0.2500\n",
        encoding="utf-8",
    )
    structure = UnitcellStructureView(load(path))
    rational = structure.rationalized()
    assert _coords(rational) == [[Fraction(1, 3), Fraction(2, 3), Fraction(1, 4)]]
    assert rational.cell.angles[2] == 120
    assert rational.cell.basis == structure.cell.basis  # 3.21 has no nearby smaller denominator
    assert same_crystal(rational, rational.rationalized())  # idempotent


def test_rational_basis_entries_snap_within_the_length_tolerance():
    cell = Cell([["1.50001", 0, "0.00002"], [0, 1, 0], [0, 0, 1]], scale=2, precision=Fraction(1, 1000))
    structure = UnitcellStructure(cell, Sites([[0, 0, 0]]), species_at_sites=["Si"])
    rational = structure.rationalized()
    assert rational.cell.unscaled_basis == Cell([[Fraction(3, 2), 0, 0], [0, 1, 0], [0, 0, 1]]).basis
    assert rational.cell.scale == 2
    assert rational.basis_precision == Fraction(1, 1000)
    # A 1e-5 length tolerance (5e-6 per unscaled entry) keeps the 0.00002 entry exactly.
    tight = structure.rationalized(basis_tolerance=Fraction(1, 100_000))
    assert tight.cell.unscaled_basis._element((0, 2)) == Fraction(1, 50_000)


def test_irrational_basis_not_built_from_parameters_is_unchanged():
    # A plain Cell holding a hexagonal surd basis: entry-wise snapping could corrupt sqrt(3).
    irrational = Cell(CellParams(("1.00001", "1.00001", 2, 90, 90, 120)).basis, precision=Fraction(1, 1000))
    structure = UnitcellStructure(irrational, Sites([[0, 0, 0]]), species_at_sites=["Si"])
    assert structure.rationalized().cell is structure.cell


def test_metadata_is_carried_and_ids_are_not():
    cell = Cell([[3, 0, 0], [0, 3, 0], [0, 0, 3]], precision=Fraction(1, 100), periodicity=(True, True, False))
    structure = UnitcellStructure(
        cell,
        Sites([["0.3333", 0, 0]], precision=Fraction(1, 10000)),
        species_at_sites=["Si"],
        site_moments=CollinearSiteMoments([2]),
        chemical_formula_descriptive="Si",
        optimization_type="local",
        immutable_id="abc",
        charge=1,
    )
    rational = structure.rationalized()
    assert rational.periodicity == (True, True, False)
    assert rational.site_moments is structure.site_moments
    assert rational.chemical_formula_descriptive == "Si"
    assert rational.optimization_type == "local"
    assert rational.charge == 1
    assert rational.species == structure.species
    assert rational.basis_precision == Fraction(1, 100)
    assert rational.immutable_id is None


def test_crystal_axis_moments_keep_their_cartesian_vectors():
    # b leans by 1e-5 along x; rationalizing to the identity cell must not rotate the moment.
    cell = Cell([[1, 0, 0], ["0.00001", 1, 0], [0, 0, 1]], precision=Fraction(1, 1000))
    moments = CrystalAxisSiteMoments([[0, 2, 0]], cell)
    structure = UnitcellStructure(cell, Sites([[0, 0, 0]]), species_at_sites=["Fe"], site_moments=moments)
    rational = structure.rationalized()
    assert rational.cell.basis == Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).basis
    assert isinstance(rational.site_moments, CrystalAxisSiteMoments)
    assert rational.site_moments.cell == rational.cell
    assert rational.site_moments.cartesian_moments == moments.cartesian_moments

    cartesian = CartesianSiteMoments([[0, 2, 0]])
    plain = UnitcellStructure(cell, Sites([[0, 0, 0]]), species_at_sites=["Fe"], site_moments=cartesian)
    assert plain.rationalized().site_moments is cartesian
