"""
Tests for the numeric layer: the float-owning NumericCell/NumericSites/NumericUnitcellStructure
values, their lossless numeric views, and the numeric structure backend adapter.
"""

import datetime
import fractions
import math
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import pytest

numpy = pytest.importorskip("numpy")

import httk.core.vectors as vectors_pkg
from httk.core import FracVector, unwrap

from httk.atomistic import (
    Assembly,
    ASUStructure,
    BareProtostructureView,
    BarePrototypeView,
    CartesianSiteMoments,
    Cell,
    CellParams,
    CollinearSiteMoments,
    CrystalAxisSiteMoments,
    NumericCell,
    NumericSiteMoments,
    NumericSites,
    NumericUnitcellStructure,
    NumericUnitcellStructureView,
    Sites,
    Species,
    StructuretypeView,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
)
from httk.atomistic.models.cell.numeric_view import CellNumericView
from httk.atomistic.models.sites.numeric_view import SitesNumericView
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.numeric import _NumericStructureBackend

F = fractions.Fraction
CUBE = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]


def _hexagonal_structure() -> UnitcellStructure:
    cell = Cell(CellParams((3, 3, 5, 90, 90, 120)).basis)
    return UnitcellStructure(
        cell=cell,
        sites=[[F(0), F(0), F(0)], [F(1, 3), F(1, 3), F(0)]],
        species=[{"name": "Mg", "chemical_symbols": ["Mg"], "concentration": [1.0]}],
        species_at_sites=["Mg", "Mg"],
    )


def _disordered_structure(**kwargs: Any) -> UnitcellStructure:
    return UnitcellStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        [[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]],
        species=[
            Species("A", ("Fe", "Ni"), (F(1, 3), F(2, 3)), charges=(F(1, 3), None)),
            Species("O", ("O",), (1,)),
        ],
        species_at_sites=["A", "O"],
        charge=F(1, 3),
        **kwargs,
    )


def _numeric() -> NumericUnitcellStructure:
    return NumericUnitcellStructure(
        NumericCell(CUBE, precision=1e-3, periodicity=(True, True, False)),
        NumericSites([[0.0, 0.0, 0.0], [0.5, 0.5, 0.1]], precision=1e-4),
        species_at_sites=[Species("Na", ("Na",), (1,), mass=(22.99,)), Species("Cl", ("Cl",), (1,))],
        charge=-1.5,
        molecular=False,
        chemical_formula_descriptive="NaCl",
        chemical_formula_hill="ClNa",
        optimization_type="local",
        immutable_id="abc",
        last_modified=datetime.datetime(2026, 11, 5, tzinfo=datetime.UTC),
    )


# ------------------------------------------------------------------ values


def test_values_round_trip_through_repr() -> None:
    namespace = {
        "NumericCell": NumericCell,
        "NumericSites": NumericSites,
        "NumericSiteMoments": NumericSiteMoments,
        "NumericUnitcellStructure": NumericUnitcellStructure,
        "Species": Species,
        "Fraction": Fraction,
        "datetime": datetime,
        "numpy": numpy,
    }
    structure = _numeric()
    magnetic = NumericUnitcellStructure(
        CUBE, [[0, 0, 0], [0.5, 0.5, 0.5]], species_at_sites=["Fe", "Fe"], site_moments=[1.5, -1.5]
    )
    for value in (
        structure,
        structure.cell,
        structure.sites,
        NumericUnitcellStructure(_disordered_structure()),
        magnetic,
        NumericSiteMoments([[0.1, 0, -2]], precision=0.01),
        NumericSiteMoments(()),
        NumericSiteMoments(numpy.empty((0, 3))),
    ):
        copy = eval(repr(value), namespace)
        assert copy == value
        assert hash(copy) == hash(value)
    assert eval(repr(structure), namespace).last_modified == structure.last_modified


def test_values_normalise_negative_zero_and_reject_non_finite() -> None:
    cell = NumericCell([[4.0, -0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    assert cell == NumericCell(CUBE)
    assert hash(cell) == hash(NumericCell(CUBE))
    assert "-0.0" not in repr(cell)
    assert NumericSites([[-0.0, 0.0, 0.0]]) == NumericSites([[0.0, 0.0, 0.0]])
    zero_charge = NumericUnitcellStructure(CUBE, [[0, 0, 0]], species_at_sites=["Na"], charge=-0.0)
    assert repr(zero_charge.charge) == "0.0"
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            NumericCell([[bad, 0, 0], [0, 4, 0], [0, 0, 4]])
        with pytest.raises(ValueError, match="finite"):
            NumericSites([[0, bad, 0]])
        with pytest.raises(ValueError, match="finite"):
            NumericUnitcellStructure(CUBE, [[0, 0, 0]], species_at_sites=["Na"], charge=bad)
    with pytest.raises(ValueError, match="non-degenerate"):
        NumericCell([[1, 0, 0], [2, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="3x3"):
        NumericCell([[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="strictly positive"):
        NumericSites([[0, 0, 0]], precision=0)
    assert len(NumericSites(numpy.zeros((0, 3)))) == 0
    for wrong_width in (numpy.zeros((2, 0)), [[0, 0], [0, 0]]):
        with pytest.raises(ValueError, match="Nx3"):
            NumericSites(wrong_width)


def test_value_arrays_are_read_only_float64() -> None:
    structure = _numeric()
    for array in (
        structure.cell.basis,
        structure.cell.lengths,
        structure.cell.angles,
        structure.cell.metric(),
        structure.sites.reduced_coords,
        structure.sites[0],
        next(iter(structure.sites)),
    ):
        assert type(array) is numpy.ndarray
        assert array.dtype == numpy.float64
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 1.0
    assert NumericSites([]).reduced_coords.shape == (0, 3)


def test_numeric_cell_geometry() -> None:
    cell = NumericCell(Cell(CellParams((3, 3, 5, 90, 90, 120)).basis))
    assert cell.lengths.tolist() == pytest.approx([3.0, 3.0, 5.0])
    assert cell.angles.tolist() == pytest.approx([90.0, 90.0, 120.0])
    assert cell.volume == pytest.approx(3 * 3 * 5 * 3**0.5 / 2)
    assert numpy.allclose(cell.metric(), cell.basis @ cell.basis.T)
    assert not hasattr(cell, "scale") and not hasattr(cell, "exact")
    slab = NumericCell(CUBE, periodicity=(1, 1, 0))
    assert slab.nperiodic_dimensions == 2
    with pytest.raises(ValueError, match="fully 3D-periodic"):
        _ = slab.volume


def test_structure_value_fields_and_semantics() -> None:
    structure = _numeric()
    assert structure.cell.precision == 1e-3
    assert structure.sites.precision == 1e-4
    assert structure.species_at_sites == ("Na", "Cl")
    assert structure.nsites == 2
    assert structure.charge == -1.5
    assert structure.elements == ("Cl", "Na")
    assert structure.nelements == 2
    assert structure.elements_ratios == (F(1, 2), F(1, 2))
    assert structure.chemical_formula_reduced == "ClNa"
    assert structure.chemical_formula_anonymous == "AB"
    assert structure.structure_features == ()
    assert structure.lattice_vectors == CUBE
    assert structure.fractional_site_positions == [[0.0, 0.0, 0.0], [0.5, 0.5, 0.1]]
    assert numpy.allclose(structure.cartesian_site_positions, [[0.0, 0.0, 0.0], [2.0, 2.0, 0.4]])
    assert len(structure.id) == 64  # content identity; see test_numeric_structure_record.py
    assert structure.type == "structures"
    assert not isinstance(structure, StructureBackend)
    # Identity metadata does not take part in equality.
    other = NumericUnitcellStructure(
        structure.cell,
        structure.sites,
        structure.species,
        structure.species_at_sites,
        charge=-1.5,
        chemical_formula_descriptive="NaCl",
        chemical_formula_hill="ClNa",
        optimization_type="local",
    )
    assert other == structure
    assert NumericUnitcellStructure(other, immutable_id="x").immutable_id == "x"
    with pytest.raises(ValueError, match="unknown species"):
        NumericUnitcellStructure(CUBE, [[0, 0, 0]], [Species("Na", ("Na",), (1,))], ["K"])
    with pytest.raises(ValueError, match="same length"):
        NumericUnitcellStructure(CUBE, [[0, 0, 0]], species_at_sites=["Na", "Na"])
    with pytest.raises(ValueError, match="chemical_formula_hill"):
        NumericUnitcellStructure(CUBE, [[0, 0, 0]], species_at_sites=["Na"], chemical_formula_hill="Cl")


def test_lossy_conversion_quantises_species_through_float() -> None:
    exact = _disordered_structure()
    numeric = NumericUnitcellStructure(exact)
    third = Fraction(3333333333333333, 10**16)
    assert numeric.species[0].concentration == (third, Fraction(6666666666666666, 10**16))
    assert numeric.species[0].charges == (third, None)
    assert numeric.charge == 1 / 3
    assert numeric.cell == NumericCell(exact.cell)
    assert numeric.sites == NumericSites(exact.sites)
    # Quantisation is idempotent, so a float round trip is the identity.
    assert NumericUnitcellStructure(numeric) == numeric
    assert NumericUnitcellStructure(UnitcellStructureView(numeric)) == numeric
    assert NumericUnitcellStructure(exact.numeric()) == numeric
    assert "disorder" in numeric.structure_features
    assert NumericUnitcellStructure(_disordered_structure(immutable_id="src")).immutable_id is None
    with pytest.raises(TypeError, match="takes its content from the source"):
        NumericUnitcellStructure(exact, charge=1.0)


def test_conversion_refuses_content_a_numeric_structure_cannot_carry() -> None:
    moments = UnitcellStructure(
        CUBE,
        [[0, 0, 0]],
        species_at_sites=[Species("Fe", ("Fe",), (1,))],
        site_moments=CartesianSiteMoments([[0, 0, 1]]),
    )
    assert NumericUnitcellStructure(moments).site_moments == NumericSiteMoments([[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="assemblies"):
        NumericUnitcellStructure(
            UnitcellStructure(
                CUBE,
                [[0, 0, 0], [0, 0, F(1, 2)]],
                species_at_sites=[Species("Fe", ("Fe",), (1,)), Species("Ni", ("Ni",), (1,))],
                assemblies=(Assembly(((0,), (1,)), (F(1, 2), F(1, 2))),),
            )
        )


# ------------------------------------------------------------------ site moments


def test_numeric_site_moments_values() -> None:
    collinear = NumericSiteMoments([1, -0.0, 2.5])
    assert collinear.kind == "collinear"
    assert collinear.moments.tolist() == [1.0, 0.0, 2.5]
    assert math.copysign(1.0, collinear.moments[1]) == 1.0
    assert not collinear.moments.flags.writeable
    assert len(collinear) == 3
    with pytest.raises(ValueError, match="no axis"):
        collinear.cartesian_moments  # noqa: B018 - the access itself raises
    cartesian = NumericSiteMoments(numpy.array([[0.0, -0.0, 1.0]]), precision=numpy.float64(0.01))
    assert cartesian.kind == "cartesian"
    assert cartesian.cartesian_moments.shape == (1, 3)
    assert not cartesian.cartesian_moments.flags.writeable
    assert type(cartesian.precision) is float
    assert NumericSiteMoments(()).kind == "collinear"
    empty = NumericSiteMoments(numpy.zeros((0, 3)))
    assert empty.kind == "cartesian"
    assert empty.moments.shape == (0, 3)
    assert len(empty) == 0
    # Precision is provenance, not content; the kind is content.
    assert cartesian == NumericSiteMoments([[0, 0, 1]])
    assert hash(cartesian) == hash(NumericSiteMoments([[0, 0, 1]]))
    assert NumericSiteMoments([1.0]) != NumericSiteMoments([[1.0, 0.0, 0.0]])
    assert NumericSiteMoments(cartesian).precision == 0.01
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError, match="finite"):
            NumericSiteMoments([bad])
        with pytest.raises(ValueError, match="finite"):
            NumericSiteMoments([[0, bad, 0]])
    for shape in ([[1.0, 2.0]], [[[1.0, 2.0, 3.0]]], 1.0):
        with pytest.raises(ValueError, match="1-D or Nx3"):
            NumericSiteMoments(shape)
    with pytest.raises(ValueError, match="strictly positive"):
        NumericSiteMoments([1.0], precision=0)


def _magnetic_hexagonal(moments: Any) -> UnitcellStructure:
    structure = _hexagonal_structure()
    return UnitcellStructure(
        structure.cell, structure.sites, structure.species, structure.species_at_sites, site_moments=moments
    )


@pytest.mark.parametrize("kind", ("collinear", "cartesian", "crystalaxis"))
def test_conversion_carries_exact_moments_as_floats(kind: str) -> None:
    cell = _hexagonal_structure().cell
    rows = [[1, 2, 3], [-1, 0, F(1, 3)]]
    moments = {
        "collinear": CollinearSiteMoments([F(1, 3), -1], precision=F(1, 10)),
        "cartesian": CartesianSiteMoments(rows, precision=F(1, 10)),
        "crystalaxis": CrystalAxisSiteMoments(rows, cell, precision=F(1, 10)),
    }[kind]
    numeric = NumericUnitcellStructure(_magnetic_hexagonal(moments))
    converted = numeric.site_moments
    assert converted is not None
    assert converted == NumericSiteMoments(moments)
    assert converted.precision == 0.1
    assert NumericSiteMoments(moments, precision=0.5).precision == 0.5
    assert "_httk_magnetism" in numeric.structure_features
    exact = UnitcellStructureView(numeric)
    assert exact.site_moments is not None
    if kind == "collinear":
        assert converted.kind == "collinear"
        assert converted.moments.tolist() == [1 / 3, -1.0]
        assert exact.site_moments.kind == "collinear"
        assert exact.cartesian_site_moments_floats() is None
    else:
        # Crystal-axis moments land Cartesian: the hexagonal frame is not the Cartesian one.
        expected = moments.cartesian_moments.to_floats()
        assert converted.kind == "cartesian"
        assert converted.cartesian_moments.tolist() == expected
        assert (expected == [[1.0, 2.0, 3.0], [-1.0, 0.0, 1 / 3]]) is (kind == "cartesian")
        assert exact.site_moments.kind == "cartesian"
        assert exact.cartesian_site_moments_floats() == expected
    assert exact.site_moments.precision == F(1, 10)
    assert NumericUnitcellStructure(exact) == numeric
    assert NumericUnitcellStructure(numeric) == numeric


def test_structure_site_moments_keyword() -> None:
    sites = [[0, 0, 0], [0.5, 0.5, 0.5]]
    structure = NumericUnitcellStructure(
        CUBE, sites, species_at_sites=["Fe", "Fe"], site_moments=[[0, 0, 1], [0, 0, -1]]
    )
    assert structure.site_moments == NumericSiteMoments([[0, 0, 1], [0, 0, -1]])
    plain = NumericUnitcellStructure(CUBE, sites, species_at_sites=["Fe", "Fe"])
    assert plain.site_moments is None
    assert structure != plain
    assert structure.id != plain.id
    assert "_httk_magnetism" not in plain.structure_features
    with pytest.raises(ValueError, match="same length"):
        NumericUnitcellStructure(CUBE, sites, species_at_sites=["Fe", "Fe"], site_moments=[1.0])
    with pytest.raises(ValueError, match="without sites must be None"):
        NumericUnitcellStructure(CUBE, [], species_at_sites=[], site_moments=())
    with pytest.raises(TypeError, match="takes its content from the source"):
        NumericUnitcellStructure(_hexagonal_structure(), site_moments=[1.0, 1.0])


# ------------------------------------------------------------------ exact interface via the adapter


def test_unitcell_structure_view_presents_the_exact_interface() -> None:
    numeric = _numeric()
    exact = UnitcellStructureView(numeric)
    assert isinstance(exact._backend, _NumericStructureBackend)
    assert unwrap(exact) is numeric
    # Floats embed exactly as written: 0.1 is 1/10, and the precisions likewise.
    assert exact.sites.reduced_coords.to_fractions()[1] == [F(1, 2), F(1, 2), F(1, 10)]
    assert exact.basis_precision == F(1, 1000)
    assert exact.coordinate_precision == F(1, 10000)
    assert exact.periodicity == (True, True, False)
    assert exact.charge == F(-3, 2)
    assert exact.chemical_formula_hill == "ClNa"
    assert exact.optimization_type == "local"
    assert exact.immutable_id == "abc"
    assert isinstance(exact.unview(), UnitcellStructure)
    assert NumericUnitcellStructure(exact) == numeric
    with pytest.raises(TypeError):
        StructureBackend._select_backend(numeric, kind="unitcell")
    assert isinstance(StructureBackend._select_backend(numeric, kind="numeric"), _NumericStructureBackend)


def test_adapter_quartet_precedence() -> None:
    structure = _hexagonal_structure()
    # 1. an exact UnitcellStructure on the object wins.
    view = NumericUnitcellStructureView(structure)
    backend = StructureBackend._select_backend(view)
    assert isinstance(backend, _NumericStructureBackend)
    assert backend.cell is structure.cell
    assert backend.unwrap() is view
    # 2. else the numeric views' exact components.
    cell = Cell(CellParams((3, 3, 5, 90, 90, 120)).basis)
    sites = Sites([[F(1, 3), F(2, 3), 0]])
    foreign = SimpleNamespace(cell=cell.numeric(), sites=sites.numeric(), species=(), species_at_sites=("Mg",))
    foreign.species = structure.species
    backend = StructureBackend._select_backend(foreign)
    assert backend.cell is cell
    assert backend.sites is sites
    # 3. else the floats, embedded exactly as written.
    plain = SimpleNamespace(
        cell=NumericCell(CUBE), sites=NumericSites([[1 / 3, 0, 0]]), species=structure.species, species_at_sites=("Mg",)
    )
    backend = StructureBackend._select_backend(plain)
    assert backend.sites.reduced_coords.to_fractions() == [[Fraction(3333333333333333, 10**16), 0, 0]]


def test_recognition_probes_adopt_numeric_structures() -> None:
    pytest.importorskip("spglib")
    exact = UnitcellStructure(
        CUBE,
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        species_at_sites=[Species("Cs", ("Cs",), (1,)), Species("Cl", ("Cl",), (1,))],
    )
    numeric = NumericUnitcellStructure(exact)
    assert BareProtostructureView(numeric) == BareProtostructureView(exact)
    assert BarePrototypeView(numeric) == BarePrototypeView(exact)
    assert StructuretypeView(numeric).species_at_sites == StructuretypeView(exact).species_at_sites
    from httk.atomistic.entries.structures import _as_structure

    assert NumericUnitcellStructure(_as_structure(numeric)) == numeric


# ------------------------------------------------------------------ views


def test_numeric_unitcell_structure_view_of_structure_and_triple() -> None:
    structure = _hexagonal_structure()
    view = NumericUnitcellStructureView(structure)
    assert isinstance(view, NumericUnitcellStructureView)
    assert not isinstance(view, UnitcellStructure)
    assert unwrap(view) is structure  # unwrap returns the raw original
    assert NumericUnitcellStructureView(view) is view  # rewrap idempotence
    assert view.exact == structure
    assert UnitcellStructureView(view) == structure
    assert isinstance(view.cell, CellNumericView)
    assert isinstance(view.sites, SitesNumericView)
    assert view.species is structure.species
    with pytest.raises(TypeError, match="interface-only"):
        view.unview()

    triple = (CUBE, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], [11, 17])
    from_triple = NumericUnitcellStructureView(triple)
    assert unwrap(from_triple) is triple
    assert from_triple.cell.basis.tolist() == CUBE


def test_numeric_views_are_lossless_presentations() -> None:
    structure = _hexagonal_structure()
    numeric = structure.numeric()
    exact_cell = structure.cell
    for got, want in [
        (numeric.cell.basis, exact_cell.basis.to_floats()),
        (numeric.cell.metric(), exact_cell.metric().to_floats()),
        (numeric.sites.reduced_coords, structure.sites.reduced_coords.to_floats()),
    ]:
        assert type(got) is numpy.ndarray
        assert got.dtype == numpy.float64
        assert numpy.allclose(got, want, rtol=1e-12, atol=1e-12)
    assert numeric.cartesian_sites().tolist() == pytest.approx(
        numpy.array(structure.cartesian_sites().to_floats()), rel=1e-12, abs=1e-12
    )
    assert numeric.cell.angles.tolist() == [90.0, 90.0, 120.0]  # rendered from the exact angles
    assert type(numeric.cell.volume) is float

    cell = Cell([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]], precision="0.001")
    view = cell.numeric()
    assert isinstance(view, CellNumericView) and isinstance(view, NumericCell)
    assert view.exact is cell
    assert unwrap(view) is cell
    assert CellNumericView(view) is view
    assert view == NumericCell(cell)
    assert hash(view) == hash(NumericCell(cell))
    assert type(view.unview()) is NumericCell
    assert view.unview() == view
    assert repr(view) == repr(NumericCell(cell))

    sites = Sites([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], precision="0.0001")
    sites_view = sites.numeric()
    assert isinstance(sites_view, SitesNumericView) and isinstance(sites_view, NumericSites)
    assert sites_view.exact is sites
    assert SitesNumericView(sites_view) is sites_view
    assert type(sites_view.unview()) is NumericSites
    assert sites_view.unview() == NumericSites(sites)
    assert len(sites_view) == 2
    assert sites_view[1].tolist() == [0.5, 0.5, 0.5]

    params = (3.0, 3.0, 5.0, 90.0, 90.0, 120.0)
    from_params = CellNumericView(params)
    assert unwrap(from_params) is params
    assert from_params.basis.shape == (3, 3)


def test_numeric_asu_assemblies_use_expanded_indices() -> None:
    no_parameters = FracVector(())
    asu = ASUStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        221,
        [
            WyckoffSite("c", no_parameters, "C"),
            WyckoffSite("a", no_parameters, "Na"),
            WyckoffSite("b", no_parameters, "Cl"),
        ],
        [
            Species("C", ("C",), (1,)),
            Species("Na", ("Na",), (1,)),
            Species("Cl", ("Cl",), (1,)),
        ],
        assemblies=(Assembly(((1,), (2,)), (F(1, 2), F(1, 2))),),
    )

    exact = UnitcellStructureView(asu)
    numeric = NumericUnitcellStructureView(asu)
    assert numeric.assemblies == exact.assemblies
    assert numeric.assemblies is not None
    assert numeric.assemblies[0].sites_in_groups == ((3,), (4,))


# ------------------------------------------------------------------ numpy requirement


def test_construction_requires_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Report numpy unavailable via the core flag the helpers read; construction must fail fast.
    monkeypatch.setattr(vectors_pkg, "_numpy_available", False)
    structure = _hexagonal_structure()
    with pytest.raises(ImportError, match=r"httk-atomistic\[numpy\]"):
        NumericUnitcellStructureView(structure)
    with pytest.raises(ImportError, match=r"numpy"):
        NumericCell(structure.cell)
    with pytest.raises(ImportError, match=r"numpy"):
        NumericSites(structure.sites)
    with pytest.raises(ImportError, match=r"numpy"):
        NumericUnitcellStructure(structure)
    with pytest.raises(ImportError, match=r"numpy"):
        structure.numeric()
