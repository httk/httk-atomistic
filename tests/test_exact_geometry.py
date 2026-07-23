"""
Exactness tests for the migrated numeric model: SurdVector cell matrices, FracVector reduced
coordinates, exact lengths/angles/volume, exact Cartesian positions, and VectorLike inputs.
"""

import fractions
import math

import pytest
from httk.core import FracVector, SurdVector

from httk.atomistic import Cell, CellParams, Sites, Structure

F = fractions.Fraction


def _hexagonal(a: int, c: int, alpha: int = 90, beta: int = 90, gamma: int = 120) -> Cell:
    return Cell(CellParams((a, a, c, alpha, beta, gamma)).basis)


# --------------------------------------------------------------- exact params -> matrix -> params


def test_hexagonal_end_to_end_is_exact() -> None:
    params = CellParams((3, 3, 5, 90, 90, 120))
    matrix = params.basis
    # The hexagonal matrix carries a genuine sqrt(3).
    assert 3 in matrix.radicands
    cell = Cell(matrix)
    # Angles come back exactly through the reverse-Niven table; volume is (45/2)*sqrt(3).
    assert cell.angles == (F(90), F(90), F(120))
    assert cell.volume == SurdVector.from_radicand_map({3: F(45, 2)})
    assert cell.lengths == (SurdVector.create(3), SurdVector.create(3), SurdVector.create(5))
    # params -> matrix -> params round-trips exactly.
    assert params.params == (F(3), F(3), F(5), F(90), F(90), F(120))


def test_rhombohedral_60_exact_reconstruction() -> None:
    cell = Cell(CellParams((1, 1, 1, 60, 60, 60)).basis)
    assert cell.angles == (F(60), F(60), F(60))
    assert cell.lengths == (SurdVector.create(1), SurdVector.create(1), SurdVector.create(1))
    # Rhombohedral volume with a=1: sqrt(1 - 3/4 + 2/8) = sqrt(1/2).
    assert cell.volume == SurdVector.sqrt_of(F(1, 2))


def test_non_niven_angle_falls_back_deterministically() -> None:
    first = CellParams((1, 1, 1, 73, 73, 73)).basis
    second = CellParams((1, 1, 1, 73, 73, 73)).basis
    # The deterministic rational fallback is byte-identical across calls.
    assert first == second
    # ...and matches the float reference reconstruction to a tight tolerance.
    ca = math.cos(math.radians(73))
    sg = math.sin(math.radians(73))
    cy = (ca - ca * ca) / sg
    floats = first.to_floats()
    assert floats[1][0] == pytest.approx(ca, abs=1e-9)
    assert floats[1][1] == pytest.approx(sg, abs=1e-9)
    assert floats[2][1] == pytest.approx(cy, abs=1e-9)


# --------------------------------------------------------------- scale factoring


def test_scale_factors_out_exactly() -> None:
    sqrt3_over_2 = (SurdVector.sqrt_of(3) / 2)._as_scalar()
    zero = SurdVector.create(0)._as_scalar()
    one = SurdVector.create(1)._as_scalar()
    unscaled = SurdVector._from_scalar_grid(
        [
            [one, zero, zero],
            [SurdVector.create(F(-1, 2))._as_scalar(), sqrt3_over_2, zero],
            [zero, zero, SurdVector.create(F(5, 3))._as_scalar()],
        ],
        (3, 3),
    )
    scaled = Cell(unscaled, scale=3)
    absolute = Cell(scaled.basis)  # a plain matrix, scale == 1
    assert scaled.basis == absolute.basis
    assert scaled.scale == SurdVector.create(3)
    # Volume scales as scale**3; angles are scale-independent.
    unit = Cell(unscaled)
    assert scaled.volume == (unit.volume * SurdVector.create(27))._as_scalar()
    assert scaled.angles == unit.angles


# --------------------------------------------------------------- exact Cartesian positions


def test_exact_cartesian_and_metric_bond_length() -> None:
    cell = _hexagonal(3, 5)
    structure = Structure(
        cell=cell,
        sites=[[F(0), F(0), F(0)], [F(1, 3), F(1, 3), F(0)]],
        species=[{"name": "Fe", "chemical_symbols": ["Fe"], "concentration": [1.0]}],
        species_at_sites=["Fe"] * 0 + ["Fe", "Fe"],
    )
    cartesian = structure.cartesian_sites()
    assert cartesian.dim == (2, 3)
    # The exact Cartesian positions keep the sqrt(3).
    assert 3 in cartesian.radicands

    # Bond squared length two ways: via the exact Cartesian difference, and via the rational metric.
    diff = FracVector.create([F(1, 3), F(1, 3), F(0)])
    cart_diff = SurdVector.create(diff) * cell.basis
    lsq_cartesian = cart_diff.lengthsqr()
    metric = cell.metric()
    lsq_metric = (SurdVector.create(diff) * metric).dot(SurdVector.create(diff))
    assert lsq_cartesian == lsq_metric
    assert lsq_cartesian.is_rational  # magnitudes are rational-exact even when positions are surds


# --------------------------------------------------------------- VectorLike inputs


def test_vectorlike_inputs_fraction_string_fracvector() -> None:
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # Fraction leaves.
    frac_cell = Cell([[F(1, 2), 0, 0], [0, F(3, 2), 0], [0, 0, 1]])
    assert frac_cell.basis == SurdVector.create([[F(1, 2), 0, 0], [0, F(3, 2), 0], [0, 0, 1]])
    # Rational strings.
    str_sites = Sites([["1/3", "1/3", "1/3"]])
    assert str_sites.reduced_coords == FracVector.create([[F(1, 3), F(1, 3), F(1, 3)]])
    # FracVector directly.
    fv_cell = Cell(FracVector.create(identity))
    assert fv_cell.basis == SurdVector.create(identity)
    # A Structure accepts these forms.
    structure = Structure(
        cell=[["1/3", 0, 0], [0, 1, 0], [0, 0, 1]],
        sites=FracVector.create([[F(1, 4), F(1, 4), F(1, 4)]]),
        species=[{"name": "Si", "chemical_symbols": ["Si"], "concentration": [1.0]}],
        species_at_sites=["Si"],
    )
    assert structure.cell.basis.coefficient(1)[0][0] == FracVector.create(F(1, 3))
    assert structure.sites.reduced_coords == FracVector.create([[F(1, 4), F(1, 4), F(1, 4)]])


def test_vectorlike_numpy_input() -> None:
    numpy = pytest.importorskip("numpy")
    cell = Cell(numpy.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]))
    assert cell.basis == SurdVector.create([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    sites = Sites(numpy.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]))
    assert sites.reduced_coords == FracVector.create([[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]])
