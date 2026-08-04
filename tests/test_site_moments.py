import fractions

import pytest
from httk.core import FracVector, SurdVector

from httk.atomistic import (
    CartesianSiteMoments,
    CartesianSiteMomentsView,
    Cell,
    CellParams,
    CollinearSiteMoments,
    CrystalAxisSiteMoments,
    CrystalAxisSiteMomentsView,
)

F = fractions.Fraction


def _hexagonal_cell() -> Cell:
    return Cell(CellParams((3, 3, 4, 90, 90, 120)).basis)


def _orthorhombic_cell() -> Cell:
    return Cell([[2, 0, 0], [0, 3, 0], [0, 0, 4]])


def test_site_moments_backends_validate_shapes_and_normalize_precision() -> None:
    assert CartesianSiteMoments([[1, 2, 3]], 1e-4).precision == F(1, 10000)
    assert CrystalAxisSiteMoments([[1, 2, 3]], _orthorhombic_cell(), "1/1000").precision == F(1, 1000)
    assert CollinearSiteMoments([1, 2], F(1, 100)).precision == F(1, 100)
    assert isinstance(CartesianSiteMoments([[1, 2, 3]]).cartesian_moments, SurdVector)
    assert isinstance(CollinearSiteMoments([1, 2]).collinear_moments, FracVector)

    with pytest.raises(ValueError, match="Nx3"):
        CartesianSiteMoments([[1, 2]])
    with pytest.raises(ValueError, match="Nx3"):
        CrystalAxisSiteMoments([1, 2, 3], _orthorhombic_cell())
    with pytest.raises(ValueError, match="N-dimensional"):
        CollinearSiteMoments([[1, 2, 3]])


def test_hexagonal_crystalaxis_cartesian_conversion_round_trips_exactly() -> None:
    cell = _hexagonal_cell()
    crystal = CrystalAxisSiteMoments([[1, 2, 3]], cell)
    cartesian = crystal.cartesian_moments

    assert cartesian._element((0, 0)) == SurdVector.create(0)._as_scalar()
    assert cartesian._element((0, 1)) == SurdVector.sqrt_of(3)
    assert cartesian._element((0, 2)) == SurdVector.create(3)._as_scalar()
    assert (
        CrystalAxisSiteMomentsView(
            CartesianSiteMomentsView(
                crystal,
            )
        ).crystalaxis_moments
        == crystal.crystalaxis_moments
    )

    cartesian_view = CartesianSiteMomentsView(crystal)
    assert cartesian_view.cartesian_moments == cartesian
    assert CrystalAxisSiteMomentsView(cartesian_view, cell=cell).crystalaxis_moments == crystal.crystalaxis_moments


def test_orthorhombic_unit_axes_are_the_expected_cartesian_directions() -> None:
    moments = CrystalAxisSiteMoments([[1, 2, 3]], _orthorhombic_cell())
    assert moments.cartesian_moments == SurdVector.create([[1, 2, 3]])


def test_collinear_moments_have_no_fabricated_cartesian_axis() -> None:
    moments = CollinearSiteMoments([1, -2])
    with pytest.raises(ValueError, match="carries no axis"):
        _ = moments.cartesian_moments
    with pytest.raises(ValueError, match="carries no axis"):
        _ = CartesianSiteMomentsView(moments).cartesian_moments
    with pytest.raises(ValueError, match="carries no axis"):
        _ = CrystalAxisSiteMomentsView(moments).crystalaxis_moments


def test_crystalaxis_view_cell_hints_are_checked_eagerly() -> None:
    cell = _orthorhombic_cell()
    moments = CartesianSiteMoments([[1, 2, 3]])
    with pytest.raises(ValueError, match="requires a cell hint"):
        CrystalAxisSiteMomentsView(moments)
    with pytest.raises(ValueError, match="conflicts"):
        CrystalAxisSiteMomentsView(CrystalAxisSiteMoments([[1, 2, 3]], cell), cell=_hexagonal_cell())

    view = CrystalAxisSiteMomentsView(moments, cell=cell)
    assert view.crystalaxis_moments == SurdVector.create([[1, 2, 3]])
    assert view.cell == cell


def test_moment_equality_excludes_precision() -> None:
    cartesian = CartesianSiteMoments([[1, 2, 3]], F(1, 10))
    assert cartesian == CartesianSiteMoments([[1, 2, 3]], F(1, 100))
    assert hash(cartesian) == hash(CartesianSiteMoments([[1, 2, 3]]))
    assert CollinearSiteMoments([1], F(1, 10)) == CollinearSiteMoments([1], F(1, 100))
    cell = _orthorhombic_cell()
    assert CrystalAxisSiteMoments([[1, 2, 3]], cell, F(1, 10)) == CrystalAxisSiteMoments([[1, 2, 3]], cell, F(1, 100))
