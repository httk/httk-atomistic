"""Behavioral tests for the float-presentation accessors added on top of the exact API.

These check that the accessors agree with the existing exact routes to floating-point
precision, not that any particular class carries them.
"""

import fractions

import pytest

numpy = pytest.importorskip("numpy")

import httk.core.vectors as vectors_pkg
from httk.core import FracVector

from httk.atomistic import (
    ASUStructure,
    CartesianSiteMoments,
    Cell,
    CellParams,
    CollinearSiteMoments,
    CrystalAxisSiteMoments,
    FundamentalDomainStructure,
    NumericUnitcellStructureView,
    Sites,
    Species,
    SymopsStructure,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
)
from httk.atomistic.models._vector_guards import matmul_floats
from httk.atomistic.models.cell.numeric import NumericCell
from httk.atomistic.models.sites.numeric import NumericSites
from httk.atomistic.models.sites.plain import PlainSites
from httk.atomistic.models.sites.view import SitesView
from httk.atomistic.models.structure.backend import StructureBackend

F = fractions.Fraction


def _species() -> tuple[Species, Species]:
    return Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))


def _unitcell() -> UnitcellStructure:
    return UnitcellStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        [[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]],
        _species(),
        ("Na", "Cl"),
    )


def _domain(record_type: type[FundamentalDomainStructure] = FundamentalDomainStructure) -> FundamentalDomainStructure:
    return record_type(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        225,
        (
            WyckoffSite("a", FracVector(()), "Na"),
            WyckoffSite("b", FracVector(()), "Cl"),
        ),
        _species(),
    )


def _symops() -> SymopsStructure:
    cell = Cell(CellParams((3, 3, 5, 90, 90, 120)).basis)
    return SymopsStructure(
        cell,
        [[F(1, 4), 0, 0]],
        (Species("Fe", ("Fe",), (1,)),),
        ("Fe",),
        ("x,y,z", "-x,-y,-z"),
    )


_STRUCTURES = (
    pytest.param(_unitcell, id="unitcell"),
    pytest.param(_domain, id="fundamental_domain"),
    pytest.param(lambda: _domain(ASUStructure), id="asu"),
    pytest.param(_symops, id="symops"),
)


# ------------------------------------------------------------------ float pipeline agrees with the exact route


@pytest.mark.parametrize("make_structure", _STRUCTURES)
def test_cartesian_site_positions_matches_exact_cartesian_sites(make_structure: object) -> None:
    structure = make_structure()
    got = structure.cartesian_site_positions
    want = structure.cartesian_sites().to_floats()
    assert len(got) == len(want)
    assert got == pytest.approx(numpy.array(want), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("make_structure", _STRUCTURES)
def test_lattice_vectors_matches_exact_cell_basis(make_structure: object) -> None:
    structure = make_structure()
    got = structure.lattice_vectors
    want = structure.cell.basis.to_floats()
    assert got == pytest.approx(numpy.array(want), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("make_structure", _STRUCTURES)
def test_fractional_site_positions_matches_the_class_own_exact_route(make_structure: object) -> None:
    structure = make_structure()
    got = structure.fractional_site_positions
    # SymopsStructure has no representative/expanded split: reduced_coords is the same set
    # cartesian_sites() and fractional_site_positions both come from. Domain structures
    # (FundamentalDomainStructure/ASUStructure) deliberately keep .sites expanded while
    # fractional_site_positions stays representative, so compare against the representative
    # reduced coordinates each class actually derives it from, not against .sites.
    if isinstance(structure, FundamentalDomainStructure):
        want = structure._representative_sites().reduced_coords.to_floats()
    else:
        want = structure.sites.reduced_coords.to_floats()
    assert got == pytest.approx(numpy.array(want), rel=1e-12, abs=1e-12)


# ------------------------------------------------------------------ matmul_floats helper


def test_matmul_floats_empty_rows_return_empty_list() -> None:
    assert matmul_floats([], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) == []


def test_matmul_floats_numpy_and_pure_python_paths_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    matrix = Cell(CellParams((3, 3, 5, 90, 90, 120)).basis).basis.to_floats()
    rows = [[0.0, 0.0, 0.0], [F(1, 3), F(1, 3), F(0)]]
    rows = [[float(value) for value in row] for row in rows]

    with_numpy = matmul_floats(rows, matrix)

    monkeypatch.setattr(vectors_pkg, "_numpy_available", False)
    without_numpy = matmul_floats(rows, matrix)

    assert with_numpy == pytest.approx(numpy.array(without_numpy), rel=1e-12, abs=1e-12)
    assert matmul_floats([], matrix) == []


# ------------------------------------------------------------------ cartesian_site_moments_floats


def test_cartesian_site_moments_floats_is_none_without_moments() -> None:
    assert _unitcell().cartesian_site_moments_floats() is None


@pytest.mark.parametrize(
    "moments",
    (
        CartesianSiteMoments([[1, 2, 3], [-1, 0, 1]]),
        CrystalAxisSiteMoments([[1, 2, 3], [-1, 0, 1]], _unitcell().cell),
    ),
)
def test_cartesian_site_moments_floats_matches_exact_route(
    moments: CartesianSiteMoments | CrystalAxisSiteMoments,
) -> None:
    structure = UnitcellStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        [[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]],
        _species(),
        ("Na", "Cl"),
        site_moments=moments,
    )
    got = structure.cartesian_site_moments_floats()
    want = moments.cartesian_moments.to_floats()
    assert got == pytest.approx(numpy.array(want), rel=1e-12, abs=1e-12)


class _MismatchedMomentsBackend(StructureBackend):
    """A raw backend reporting more moment rows than sites, bypassing constructor validation."""

    def __init__(self) -> None:
        pass

    @property
    def cell(self) -> Cell:
        return Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]])

    @property
    def sites(self) -> Sites:
        return Sites([[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]])

    @property
    def species(self) -> tuple[Species, ...]:
        return _species()

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return ("Na", "Cl")

    @property
    def site_moments(self) -> CartesianSiteMoments:
        return CartesianSiteMoments([[1, 2, 3], [-1, 0, 1], [0, 0, 1]])


def test_cartesian_site_moments_floats_validates_length_via_the_exact_fill() -> None:
    # A view over a backend whose moments never went through UnitcellStructure's constructor
    # check: cartesian_site_moments_floats() must still catch the mismatch, not silently return
    # the ungated backend rows.
    view = UnitcellStructureView(_MismatchedMomentsBackend())
    with pytest.raises(ValueError):
        _ = view.site_moments
    with pytest.raises(ValueError):
        view.cartesian_site_moments_floats()


def test_cartesian_site_moments_floats_is_none_for_collinear() -> None:
    structure = UnitcellStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        [[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]],
        _species(),
        ("Na", "Cl"),
        site_moments=CollinearSiteMoments([1, -1]),
    )
    assert structure.cartesian_site_moments_floats() is None


# ------------------------------------------------------------------ empty structure


def test_empty_structure_has_no_cartesian_rows() -> None:
    structure = UnitcellStructure(Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), [], [], [])
    assert structure.cartesian_site_positions == []
    assert NumericUnitcellStructureView(structure).cartesian_sites().shape == (0, 3)


# ------------------------------------------------------------------ SitesView.num_sites without a forced decode


def test_sites_view_len_does_not_force_the_exact_coordinate_decode() -> None:
    backend = PlainSites([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    view = SitesView(backend)
    assert "_reduced_coords" not in view.__dict__
    assert len(view) == 2
    assert "_reduced_coords" not in view.__dict__


# ------------------------------------------------------------------ base-class float64 ndarrays


def test_numeric_cell_basis_and_sites_reduced_coords_are_base_class_float64() -> None:
    structure = _unitcell()

    basis = NumericCell(structure.cell).basis
    assert type(basis) is numpy.ndarray
    assert basis.dtype == numpy.float64
    assert basis.tolist() == pytest.approx(numpy.array(structure.cell.basis.to_floats()), rel=1e-12, abs=1e-12)

    reduced = NumericSites(structure.sites).reduced_coords
    assert type(reduced) is numpy.ndarray
    assert reduced.dtype == numpy.float64
    assert reduced.tolist() == pytest.approx(
        numpy.array(structure.sites.reduced_coords.to_floats()), rel=1e-12, abs=1e-12
    )

    cartesian = NumericUnitcellStructureView(structure).cartesian_sites()
    assert type(cartesian) is numpy.ndarray
    assert cartesian.dtype == numpy.float64
    assert cartesian.tolist() == pytest.approx(
        numpy.array(structure.cartesian_sites().to_floats()), rel=1e-12, abs=1e-12
    )
