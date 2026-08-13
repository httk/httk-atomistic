"""Unit tests for 2D, 1D and 0D periodicity on Cell, Sites and UnitcellStructure.

The model under test: the basis is a *coordinate frame*, not a container. A row flagged
non-periodic is not a lattice vector, only a statement of what a fractional coordinate means
along that direction, so coordinates there are unbounded and are never wrapped.
"""

import pytest

from httk.atomistic import Cell, UnitcellStructure
from httk.atomistic.models.species.species import Species

NA = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
CUBE = [[3, 0, 0], [0, 3, 0], [0, 0, 5]]


def _structure(periodicity=None, coords=((0, 0, 0),)) -> UnitcellStructure:
    return UnitcellStructure(Cell(CUBE, periodicity=periodicity), list(coords), [NA], ["Na"] * len(coords))


# --- the attribute ---


def test_a_cell_is_fully_periodic_unless_told_otherwise() -> None:
    """The default is not a guess: every cell in httk was a crystal before this existed."""
    assert Cell(CUBE).periodicity == (True, True, True)
    assert Cell(CUBE).nperiodic_dimensions == 3
    assert Cell(CUBE, periodicity=None).periodicity == (True, True, True)


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ((True, True, False), (True, True, False)),
        ([1, 1, 0], (True, True, False)),  # OPTIMADE's own dimension_types spelling
        ((1, 0, 1), (True, False, True)),
        ([0, 0, 0], (False, False, False)),
    ],
)
def test_periodicity_accepts_the_usual_spellings(spelling, expected) -> None:
    assert Cell(CUBE, periodicity=spelling).periodicity == expected


@pytest.mark.parametrize("count, flags", [(3, (1, 1, 1)), (2, (1, 0, 1)), (1, (0, 1, 0)), (0, (0, 0, 0))])
def test_nperiodic_dimensions_counts_the_periodic_directions(count, flags) -> None:
    assert Cell(CUBE, periodicity=flags).nperiodic_dimensions == count
    assert _structure(flags).nperiodic_dimensions == count


@pytest.mark.parametrize("bad", [(1, 1), (1, 1, 1, 1), "abc", 5, ()])
def test_periodicity_must_be_exactly_three_flags(bad) -> None:
    with pytest.raises(ValueError) as excinfo:
        Cell(CUBE, periodicity=bad)
    assert "three flags" in str(excinfo.value)


# --- the plumbing ---
#
# Silent loss is this plumbing's failure mode: a view that forgets to carry the value
# produces no error, it just quietly reports a crystal. So each reconstruction site gets
# its own test rather than one representative test.


def test_periodicity_survives_the_class_view() -> None:
    from httk.atomistic.models.cell.view import CellView

    cell = Cell(CUBE, periodicity=(1, 1, 0))
    assert CellView(cell).periodicity == (True, True, False)


def test_periodicity_survives_the_structure_view() -> None:
    from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView

    assert UnitcellStructureView(_structure((1, 1, 0))).cell.periodicity == (True, True, False)


def test_periodicity_survives_the_numeric_view() -> None:
    from httk.atomistic.models.cell.numeric_view import CellNumericView

    pytest.importorskip("numpy")
    view = CellNumericView(Cell(CUBE, periodicity=(1, 0, 0)))
    assert view.periodicity == (True, False, False)
    assert view.nperiodic_dimensions == 1


def test_periodicity_survives_the_numeric_presentation() -> None:
    pytest.importorskip("numpy")
    assert Cell(CUBE, periodicity=(0, 0, 0)).numeric().periodicity == (False, False, False)


def test_a_backend_that_knows_no_periodicity_reports_a_crystal() -> None:
    """CellParams and PlainCell have no source for it, and a crystal is the honest default.

    Six lattice parameters cannot express periodicity, so the concrete `CellAPI.periodicity`
    default is what such a backend inherits.
    """
    from httk.atomistic.models.cell.params import CellParams
    from httk.atomistic.models.cell.plain import PlainCell

    assert CellParams([3, 3, 5, 90, 90, 90]).periodicity == (True, True, True)
    assert PlainCell(CUBE).periodicity == (True, True, True)


def test_structure_reads_periodicity_through_its_cell() -> None:
    """UnitcellStructure gains no constructor argument; the value rides inside the Cell."""
    structure = _structure((1, 1, 0))
    assert structure.periodicity == structure.cell.periodicity == (True, True, False)


# --- identity ---


def test_periodicity_takes_part_in_cell_equality() -> None:
    """Unlike precision, this is not provenance: it says which rows are lattice vectors."""
    assert Cell(CUBE) == Cell(CUBE)
    assert Cell(CUBE) != Cell(CUBE, periodicity=(1, 1, 0))
    assert Cell(CUBE, periodicity=(1, 1, 0)) == Cell(CUBE, periodicity=(1, 1, 0))
    assert Cell(CUBE, periodicity=(1, 1, 0)) != Cell(CUBE, periodicity=(1, 0, 1))


def test_periodicity_takes_part_in_structure_equality_and_same_crystal() -> None:
    from httk.atomistic import same_crystal

    bulk, slab = _structure((1, 1, 1)), _structure((1, 1, 0))
    assert bulk != slab
    assert not same_crystal(bulk, slab)


def test_repr_mentions_periodicity_only_when_it_is_not_a_crystal() -> None:
    assert "periodicity" not in repr(Cell(CUBE))
    assert "periodicity=(True, True, False)" in repr(Cell(CUBE, periodicity=(1, 1, 0)))


# --- wrapping ---


def test_a_non_periodic_direction_is_never_wrapped() -> None:
    """The bug this prevents: an atom at the top of a slab compared equal to one at the bottom.

    Along a lattice direction, `0.05` and `1.05` are the same site written as different
    translates. Along a frame direction they are two different places, a whole frame vector
    apart, and calling them the same silently loses a real difference.
    """
    from httk.atomistic import same_crystal

    low, high = ((0, 0, "1/20"),), ((0, 0, "21/20"),)
    assert same_crystal(_structure((1, 1, 1), low), _structure((1, 1, 1), high))
    assert not same_crystal(_structure((1, 1, 0), low), _structure((1, 1, 0), high))
    assert not same_crystal(_structure((0, 0, 0), low), _structure((0, 0, 0), high))


def test_a_periodic_direction_is_still_wrapped() -> None:
    from httk.atomistic import same_crystal

    assert same_crystal(
        _structure((1, 1, 0), ((0, 0, "1/20"),)),
        _structure((1, 1, 0), ((1, 1, "1/20"),)),  # a and b are lattice directions
    )


def test_the_tolerance_cap_does_not_fold_a_non_periodic_direction() -> None:
    """Folding it would report two well-separated atoms as close neighbours.

    Two sites at z=0.05 and z=0.95 of a 10 A frame vector are 9 A apart. Reduced as if the
    direction were periodic they look 1 A apart, which tightens the derived tolerance far
    below what the data justifies.
    """
    from httk.atomistic.models.structure.unitcell_view import UnitcellStructureView
    from httk.atomistic.symmetry.recognition import _half_minimum_separation

    pair = ((0, 0, "1/20"), (0, 0, "19/20"))
    tall = [[3, 0, 0], [0, 3, 0], [0, 0, 10]]

    def half_separation(periodicity):
        cell = Cell(tall, periodicity=periodicity)
        structure = UnitcellStructure(cell, list(pair), [NA], ["Na", "Na"])
        return _half_minimum_separation(UnitcellStructureView(structure))

    assert half_separation((1, 1, 1)) == pytest.approx(0.5)  # folded: 1 A apart
    assert half_separation((1, 1, 0)) == pytest.approx(4.5)  # as written: 9 A apart


def test_wrapping_a_non_periodic_direction_stays_exact() -> None:
    from httk.atomistic.symmetry._periodic_wrap import wrap_periodic

    wrapped = wrap_periodic([["1/3", "4/3", "4/3"]], (True, True, False))
    assert [str(value) for value in wrapped.to_fractions()[0]] == ["1/3", "1/3", "4/3"]


# --- geometry ---


@pytest.mark.parametrize("periodicity", [(1, 1, 0), (1, 0, 1), (0, 1, 0), (0, 0, 0)])
def test_volume_refuses_a_cell_that_is_not_fully_periodic(periodicity) -> None:
    """A determinant mixing lattice vectors with frame vectors is not a volume.

    It changes when a frame vector is rescaled, although nothing about the material did,
    and any density derived from it would inherit that.
    """
    with pytest.raises(ValueError) as excinfo:
        Cell(CUBE, periodicity=periodicity).volume  # noqa: B018
    message = str(excinfo.value)
    assert "fully 3D-periodic" in message
    assert "periodic_measure" in message


def test_volume_still_works_for_a_crystal() -> None:
    assert Cell(CUBE).volume.to_float() == pytest.approx(45.0)


@pytest.mark.parametrize(
    "periodicity, expected",
    [
        ((1, 1, 1), 45.0),  # 3 * 3 * 5, a volume
        ((1, 1, 0), 9.0),  # 3 * 3, an area
        ((1, 0, 1), 15.0),  # 3 * 5, an area
        ((0, 0, 1), 5.0),  # a length
        ((0, 0, 0), 1.0),  # no repeating unit at all: the empty product
    ],
)
def test_periodic_measure_is_the_size_of_the_repeating_unit(periodicity, expected) -> None:
    measure = Cell(CUBE, periodicity=periodicity).periodic_measure
    assert measure.to_float() == pytest.approx(expected)


def test_periodic_measure_matches_volume_for_a_crystal() -> None:
    cell = Cell(CUBE)
    assert cell.periodic_measure == cell.volume


def test_a_two_dimensional_measure_stays_exact_in_the_surd_field() -> None:
    """The hexagonal area is (9/2)*sqrt(3), not a float approximation of it."""
    import fractions

    from httk.core import SurdVector

    from httk.atomistic import CellParams

    basis = CellParams((3, 3, 5, 90, 90, 120)).basis
    slab = Cell(basis, periodicity=(1, 1, 0))
    assert slab.periodic_measure == SurdVector.from_radicand_map({3: fractions.Fraction(9, 2)})


@pytest.mark.parametrize(
    "basis",
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 0]],  # a zero row
        [[1, 0, 0], [0, 1, 0], [1, 1, 0]],  # linearly dependent
        [[1, 0, 0], [1, 0, 0], [0, 0, 1]],  # a repeated row
    ],
)
def test_a_degenerate_basis_is_rejected_at_construction(basis) -> None:
    """Previously accepted, returning volume 0 and only failing much later in `angles`.

    A non-periodic direction still needs a real frame vector — a unit one, not a zero one —
    or fractional coordinates along it mean nothing and the basis cannot be inverted.
    """
    with pytest.raises(ValueError) as excinfo:
        Cell(basis)
    assert "non-degenerate" in str(excinfo.value)

    with pytest.raises(ValueError):
        Cell(basis, periodicity=(1, 1, 0))


def test_lengths_and_angles_stay_available_whatever_the_periodicity() -> None:
    """They are honest geometry of whatever the three vectors are; only the reading changes."""
    slab = Cell(CUBE, periodicity=(1, 1, 0))
    assert [length.to_float() for length in slab.lengths] == pytest.approx([3.0, 3.0, 5.0])
    assert slab.angles == Cell(CUBE).angles
    assert slab.metric() == Cell(CUBE).metric()


# --- symmetry is refused ---
#
# Without these the failures would be silent rather than loud: an orbit generated across a
# non-periodic direction quietly deletes atoms, and a distance folded across one makes an
# unsymmetric structure pass as symmetric.


def _slab() -> UnitcellStructure:
    return _structure((1, 1, 0))


def test_asu_structure_refuses_a_reduced_periodicity_cell() -> None:
    from httk.atomistic import ASUStructure, WyckoffSite

    site = WyckoffSite(wyckoff="a", free_params=(), species="Na")
    with pytest.raises(ValueError, match="fully 3D-periodic"):
        ASUStructure(_slab().cell, 221, [site], [NA])


def test_recognize_asu_refuses_before_it_searches() -> None:
    from httk.atomistic import recognize_asu

    with pytest.raises(ValueError, match="fully 3D-periodic"):
        recognize_asu(_slab())


def test_recognize_asu_refuses_even_when_the_setting_is_supplied() -> None:
    from httk.atomistic import Spacegroup, recognize_asu

    with pytest.raises(ValueError, match="fully 3D-periodic"):
        recognize_asu(_slab(), setting=Spacegroup.standard(221))


def test_the_asu_view_refuses_too() -> None:
    """It has no guard of its own; it delegates to `recognize_asu`, which does."""
    from httk.atomistic.models.structure.asu_view import ASUStructureView

    view = ASUStructureView(_slab())
    with pytest.raises(ValueError, match="fully 3D-periodic"):
        _ = view.wyckoff_sites


def test_supercell_construction_refuses_a_slab() -> None:
    """Repeating a slab in its own plane is sensible; this operation is not that one.

    The transformation mixes all three rows and the coordinates are wrapped in all three
    directions, so it would generate images along a direction with no lattice translation.
    """
    from httk.atomistic import build_supercell

    with pytest.raises(ValueError, match="fully 3D-periodic"):
        build_supercell(_slab(), [[2, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_the_refusal_says_why_and_what_still_works() -> None:
    from httk.atomistic import recognize_asu

    with pytest.raises(ValueError) as excinfo:
        recognize_asu(_structure((0, 0, 0)))
    message = str(excinfo.value)
    assert "0 of 3 directions" in message
    assert "layer or rod group" in message


def test_a_crystal_is_still_recognised_normally() -> None:
    """The guards must not disturb the overwhelmingly common case."""
    from httk.atomistic import recognize_asu, same_crystal

    cell = Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    structure = UnitcellStructure(cell, [[0, 0, 0], ["1/2", "1/2", "1/2"]], [NA], ["Na", "Na"])
    asu = recognize_asu(structure)
    assert same_crystal(asu, structure)


# --- representing a molecule ---


def test_a_molecule_is_the_identity_frame_with_nothing_periodic() -> None:
    """The whole point of the frame model: no box, no padding, no wrapping.

    With a unit frame the fractional coordinates simply *are* the angstrom coordinates, and
    a negative one stays negative instead of being folded to the far side of a cell.
    """
    hydrogen = Species(name="H", chemical_symbols=("H",), concentration=(1.0,))
    oxygen = Species(name="O", chemical_symbols=("O",), concentration=(1.0,))
    water = [[0, 0, 0], ["3/4", "3/5", 0], ["-3/4", "3/5", 0]]
    molecule = UnitcellStructure(
        Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]], periodicity=(0, 0, 0)),
        water,
        [oxygen, hydrogen],
        ["O", "H", "H"],
    )

    assert molecule.nperiodic_dimensions == 0
    assert molecule.sites.reduced_coords.to_floats() == molecule.cartesian_sites().to_floats()
    assert molecule.cartesian_sites().to_floats()[2][0] == pytest.approx(-0.75)
    assert molecule.cell.periodic_measure.to_float() == pytest.approx(1.0)


# --- serving it over OPTIMADE ---


def _record(periodicity):
    from httk.atomistic import StructureEntryProvider

    (record,) = list(StructureEntryProvider({"x": _structure(periodicity)}).records("structures"))
    return record


@pytest.mark.parametrize(
    "periodicity, nperiodic, dimension_types",
    [
        ((1, 1, 1), 3, [1, 1, 1]),
        ((1, 1, 0), 2, [1, 1, 0]),
        ((1, 0, 1), 2, [1, 0, 1]),
        ((0, 0, 1), 1, [0, 0, 1]),
        ((0, 0, 0), 0, [0, 0, 0]),
    ],
)
def test_the_served_periodicity_is_the_real_one(periodicity, nperiodic, dimension_types) -> None:
    """It used to be a hardcoded [1, 1, 1] with a comment admitting the lie."""
    record = _record(periodicity)
    assert record["nperiodic_dimensions"] == nperiodic
    assert record["dimension_types"] == dimension_types


def test_dimension_types_is_ordered_by_lattice_vector_not_by_cartesian_axis() -> None:
    """OPTIMADE is explicit that entry i refers to lattice vector i, as Cell.periodicity is."""
    assert _record((0, 1, 0))["dimension_types"] == [0, 1, 0]


def test_a_reduced_periodicity_structure_serves_no_space_group() -> None:
    """OPTIMADE requires this, and it falls out rather than being enforced separately.

    `space_group_symbol_hall` and `space_group_it_number` MUST be null unless
    `nperiodic_dimensions` is 3, and `space_group_symmetry_operations_xyz` MUST be null when
    it is 0. Symmetry is only ever served for an `ASUStructure`, and one of those cannot
    hold a reduced-periodicity cell, so there is no combination that could violate it.
    """
    for periodicity in ((1, 1, 0), (0, 0, 1), (0, 0, 0)):
        record = _record(periodicity)
        for name, value in record.items():
            if (
                name.startswith("space_group") and name != "space_group_symmetry_operations_xyz"
            ) or name == "wyckoff_positions":
                assert value is None, f"{name} served for {periodicity}"
        assert record["space_group_symmetry_operations_xyz"] == (["x,y,z"] if any(periodicity) else None)


def test_a_crystal_still_serves_its_space_group() -> None:
    from httk.atomistic import StructureEntryProvider, recognize_asu

    cell = Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    structure = UnitcellStructure(cell, [[0, 0, 0], ["1/2", "1/2", "1/2"]], [NA], ["Na", "Na"])
    asu = recognize_asu(structure)
    (record,) = list(StructureEntryProvider({"x": asu}).records("structures"))
    assert record["nperiodic_dimensions"] == 3
    assert record["space_group_it_number"] == 229
    assert record["site_coordinate_span"] == "asymmetric_unit"


def test_site_coordinate_span_is_selected_by_representation() -> None:
    assert _record((1, 1, 1))["site_coordinate_span"] == "unit_cell"
    assert _record((1, 1, 0))["site_coordinate_span"] == "unit_cell"
    assert _record((0, 0, 1))["site_coordinate_span"] == "unit_cell"
    assert _record((0, 0, 0))["site_coordinate_span"] == "unit_cell"


def test_lattice_vectors_are_always_three_whatever_the_periodicity() -> None:
    """OPTIMADE requires all three regardless; the non-periodic ones are frame vectors."""
    record = _record((0, 0, 0))
    assert record["lattice_vectors"] == [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]]


# --- marking a loaded structure as a slab ---


def test_the_documented_remarking_recipe_preserves_everything_else() -> None:
    """Neither CIF nor POSCAR can state periodicity, so this is the only way to build a slab.

    There is deliberately no `with_periodicity()` helper, which makes it worth pinning that
    the by-hand recipe in `docs/periodicity.md` really does keep the exact scale factoring
    and both recorded precisions.
    """
    from httk.atomistic import Sites

    loaded = UnitcellStructure(
        Cell([[2, 0, 0], [0, 2, 0], [0, 0, 20]], scale=3, precision="1/10000"),
        Sites([[0, 0, "2/5"]], precision="1/100000"),
        [NA],
        ["Na"],
    )
    assert loaded.periodicity == (True, True, True)

    slab = UnitcellStructure(
        Cell(
            loaded.cell.unscaled_basis,
            loaded.cell.scale,
            loaded.cell.precision,
            (True, True, False),
        ),
        loaded.sites,
        loaded.species,
        loaded.species_at_sites,
    )

    assert slab.periodicity == (True, True, False)
    assert slab.cell.basis == loaded.cell.basis
    assert slab.cell.scale == loaded.cell.scale  # the exact factoring survives
    assert slab.basis_precision == loaded.basis_precision
    assert slab.coordinate_precision == loaded.coordinate_precision
    # 2D area of the 6 x 6 periodic plane (2 x 2 scaled by 3).
    assert slab.cell.periodic_measure.to_float() == pytest.approx(36.0)
