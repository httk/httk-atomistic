"""Unit tests for 2D, 1D and 0D periodicity on Cell, Sites and Structure.

The model under test: the basis is a *coordinate frame*, not a container. A row flagged
non-periodic is not a lattice vector, only a statement of what a fractional coordinate means
along that direction, so coordinates there are unbounded and are never wrapped.
"""

import pytest

from httk.atomistic import Cell, Structure
from httk.atomistic.species import Species

NA = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
CUBE = [[3, 0, 0], [0, 3, 0], [0, 0, 5]]


def _structure(periodicity=None, coords=((0, 0, 0),)) -> Structure:
    return Structure(Cell(CUBE, periodicity=periodicity), list(coords), [NA], ["Na"] * len(coords))


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
    from httk.atomistic.cell_class_view import CellClassView

    cell = Cell(CUBE, periodicity=(1, 1, 0))
    assert CellClassView(cell).periodicity == (True, True, False)


def test_periodicity_survives_the_structure_view() -> None:
    from httk.atomistic.structure_simple_view import StructureSimpleView

    assert StructureSimpleView(_structure((1, 1, 0))).cell.periodicity == (True, True, False)


def test_periodicity_survives_the_numeric_view() -> None:
    from httk.atomistic.cell_numeric_view import CellNumericView

    pytest.importorskip("numpy")
    view = CellNumericView(Cell(CUBE, periodicity=(1, 0, 0)))
    assert view.periodicity == (True, False, False)
    assert view.nperiodic_dimensions == 1


def test_periodicity_survives_the_numeric_presentation() -> None:
    pytest.importorskip("numpy")
    assert Cell(CUBE, periodicity=(0, 0, 0)).numeric().periodicity == (False, False, False)


def test_a_backend_that_knows_no_periodicity_reports_a_crystal() -> None:
    """CellParams and CellPrimitive have no source for it, and a crystal is the honest default.

    Six lattice parameters cannot express periodicity, so the concrete `CellAPI.periodicity`
    default is what such a backend inherits.
    """
    from httk.atomistic.cell_params import CellParams
    from httk.atomistic.cell_primitive import CellPrimitive

    assert CellParams([3, 3, 5, 90, 90, 90]).periodicity == (True, True, True)
    assert CellPrimitive(CUBE).periodicity == (True, True, True)


def test_structure_reads_periodicity_through_its_cell() -> None:
    """Structure gains no constructor argument; the value rides inside the Cell."""
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
    from httk.atomistic.asu_recognition import _half_minimum_separation
    from httk.atomistic.structure_simple_view import StructureSimpleView

    pair = ((0, 0, "1/20"), (0, 0, "19/20"))
    tall = [[3, 0, 0], [0, 3, 0], [0, 0, 10]]

    def half_separation(periodicity):
        cell = Cell(tall, periodicity=periodicity)
        structure = Structure(cell, list(pair), [NA], ["Na", "Na"])
        return _half_minimum_separation(StructureSimpleView(structure))

    assert half_separation((1, 1, 1)) == pytest.approx(0.5)  # folded: 1 A apart
    assert half_separation((1, 1, 0)) == pytest.approx(4.5)  # as written: 9 A apart


def test_wrapping_a_non_periodic_direction_stays_exact() -> None:
    from httk.atomistic._periodic_wrap import wrap_periodic

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
        Cell(CUBE, periodicity=periodicity).volume
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
