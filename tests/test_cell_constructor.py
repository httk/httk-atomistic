"""Cell() accepts any cell-like value: a 3x3 basis, six parameters, or another cell."""

import fractions

import pytest

from httk.atomistic import Cell, CellParams, CellView


def test_cell_from_six_parameters_is_the_exact_params_basis() -> None:
    cell = Cell([3, 3, 5, 90, 90, 120])
    assert cell.basis == CellParams((3, 3, 5, 90, 90, 120)).basis
    assert 3 in cell.basis.radicands
    assert cell.volume == CellView([3, 3, 5, 90, 90, 120]).volume
    assert cell.scale == 1 and cell.precision is None and cell.periodicity == (True, True, True)


def test_cell_adopts_another_cell_and_explicit_arguments_win() -> None:
    source = Cell(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scale=4,
        precision=fractions.Fraction(1, 1000),
        periodicity=(True, True, False),
    )
    adopted = Cell(source)
    assert adopted == source
    assert (adopted.scale, adopted.precision, adopted.periodicity) == (
        source.scale,
        source.precision,
        source.periodicity,
    )
    assert Cell(CellParams((3, 3, 5, 90, 90, 120))).basis == Cell([3, 3, 5, 90, 90, 120]).basis
    assert Cell(CellView([3, 3, 5, 90, 90, 120])).basis == Cell([3, 3, 5, 90, 90, 120]).basis
    overridden = Cell(source, periodicity=(True, True, True))
    assert overridden.periodicity == (True, True, True) and overridden.scale == source.scale


def test_cell_still_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError):
        Cell([1, 2, 3])
    with pytest.raises(ValueError):
        Cell([3, 3, 5, 90, 90, 0])
