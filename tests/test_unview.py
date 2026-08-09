"""Tests for unview across the atomistic view families."""

import fractions

import pytest
from httk.core import FracVector, View, unview

from httk.atomistic import (
    ASUStructure,
    ASUStructureView,
    Cell,
    CellParamsView,
    CellView,
    PlainSpeciesView,
    PlainStructureView,
    Sites,
    SitesView,
    Species,
    SpeciesView,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
)
from httk.atomistic.models.cell.params import CellParams
from httk.atomistic.models.cell.plain_view import PlainCellView
from httk.atomistic.models.formula.composition import Composition
from httk.atomistic.models.formula.formula import ChemicalFormula
from httk.atomistic.models.moments.cartesian import CartesianSiteMoments
from httk.atomistic.models.moments.cartesian_view import CartesianSiteMomentsView
from httk.atomistic.models.moments.crystalaxis import CrystalAxisSiteMoments
from httk.atomistic.models.moments.crystalaxis_view import CrystalAxisSiteMomentsView
from httk.atomistic.models.sites.plain_view import PlainSitesView

F = fractions.Fraction

CUBIC = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
NO_PARAMETERS = FracVector(())


def _cell() -> Cell:
    return Cell(CUBIC)


def _sites() -> Sites:
    return Sites([[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]])


def _species(name: str = "Na") -> Species:
    return Species(name=name, chemical_symbols=(name,), concentration=(1.0,))


def _structure() -> UnitcellStructure:
    return UnitcellStructure(_cell(), _sites(), (_species("Na"), _species("Cl")), ("Na", "Cl"))


# --- cell family ---


def test_cell_view_unview_reuses_a_cell_backend() -> None:
    cell = _cell()
    plain = unview(CellView(cell))
    assert plain is cell


def test_cell_view_unview_materializes_other_backends() -> None:
    view = CellView(CellParams((4.0, 4.0, 4.0, 90, 90, 90)))
    plain = unview(view)
    assert type(plain) is Cell
    assert plain == view


def test_plain_cell_view_and_params_view_unview_to_tuples() -> None:
    for view_cls in (PlainCellView, CellParamsView):
        view = view_cls(_cell())
        plain = unview(view)
        assert type(plain) is tuple
        assert plain == tuple(view)


# --- sites family ---


def test_sites_view_unview() -> None:
    sites = _sites()
    assert unview(SitesView(sites)) is sites
    plain = unview(SitesView([[0, 0, 0]]))
    assert type(plain) is Sites

    plain_tuple = unview(PlainSitesView(sites))
    assert type(plain_tuple) is tuple


# --- species family ---


def test_species_view_unview() -> None:
    species = _species()
    assert unview(SpeciesView(species)) is species
    converted = SpeciesView("Na")
    plain = unview(converted)
    assert type(plain) is Species
    assert plain == converted


def test_plain_species_view_unview_is_a_plain_dict() -> None:
    view = PlainSpeciesView(_species())
    plain = unview(view)
    assert type(plain) is dict
    assert plain == dict(view)


# --- structure family ---


def test_unitcell_structure_view_unview_reuses_the_backend() -> None:
    structure = _structure()
    assert unview(UnitcellStructureView(structure)) is structure


def test_structure_formula_and_composition_views_unview_to_plain_values() -> None:
    structure = _structure()

    assert type(unview(structure.formula)) is ChemicalFormula
    assert type(unview(structure.composition)) is Composition


def test_unitcell_structure_view_unview_with_view_metadata_materializes() -> None:
    structure = _structure()
    view = UnitcellStructureView(structure, immutable_id="mat-1")
    plain = unview(view)
    assert type(plain) is UnitcellStructure
    assert plain is not structure
    assert plain.immutable_id == "mat-1"
    assert plain == structure  # metadata takes no part in structure equality


def test_unitcell_structure_view_unview_expands_an_asu_backend() -> None:
    asu = ASUStructure(
        cell=CUBIC,
        spacegroup=225,
        wyckoff_sites=[WyckoffSite("a", NO_PARAMETERS, "Na"), WyckoffSite("b", NO_PARAMETERS, "Cl")],
        species=[_species("Na"), _species("Cl")],
    )
    plain = unview(UnitcellStructureView(asu))
    assert type(plain) is UnitcellStructure
    assert len(plain.sites) == 8


def test_asu_structure_view_unview() -> None:
    asu = ASUStructure(
        cell=CUBIC,
        spacegroup=225,
        wyckoff_sites=[WyckoffSite("a", NO_PARAMETERS, "Na")],
        species=[_species("Na")],
    )
    assert unview(ASUStructureView(asu)) is asu
    with_metadata = ASUStructureView(asu, immutable_id="mat-2")
    plain = unview(with_metadata)
    assert type(plain) is ASUStructure
    assert plain.immutable_id == "mat-2"


def test_plain_structure_view_unview() -> None:
    plain = unview(PlainStructureView(_structure()))
    assert type(plain) is tuple
    assert len(plain) == 3


def test_numeric_structure_view_is_interface_only() -> None:
    pytest.importorskip("numpy")
    from httk.atomistic import NumericUnitcellStructureView

    view = NumericUnitcellStructureView(_structure())
    with pytest.raises(TypeError, match="interface-only"):
        unview(view)


def test_ase_atoms_view_unview() -> None:
    ase = pytest.importorskip("ase")
    from httk.atomistic import ASEAtomsView

    view = ASEAtomsView(_structure())
    plain = unview(view)
    assert type(plain) is ase.Atoms
    assert not isinstance(plain, View)
    assert (plain.numbers == view.numbers).all()


def test_pymatgen_structure_view_unview() -> None:
    pytest.importorskip("pymatgen")
    from pymatgen.core import Structure

    from httk.atomistic import PymatgenStructureView

    source = Structure([[4, 0, 0], [0, 4, 0], [0, 0, 4]], ["Na"], [[0, 0, 0]])
    view = PymatgenStructureView(source)
    view.replace(0, "Na", label="replaced")
    plain = unview(view)
    assert type(plain) is Structure
    assert not isinstance(plain, View)
    assert plain.labels == ["replaced"]


# --- moments family ---


def test_cartesian_moments_view_unview() -> None:
    moments = CartesianSiteMoments([[0, 0, 1], [0, 0, -1]])
    assert unview(CartesianSiteMomentsView(moments)) is moments


def test_crystalaxis_moments_view_unview() -> None:
    cell = _cell()
    native = CrystalAxisSiteMoments([[0, 0, 1]], cell)
    assert unview(CrystalAxisSiteMomentsView(native)) is native

    converted = CrystalAxisSiteMomentsView(CartesianSiteMoments([[0, 0, 1]]), cell=cell)
    plain = unview(converted)
    assert type(plain) is CrystalAxisSiteMoments
    assert plain.crystalaxis_moments == converted.crystalaxis_moments
