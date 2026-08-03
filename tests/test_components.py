import fractions
import math

import pytest
from httk.core import SurdVector, unwrap
from httk.core.storage import project_storage_record

from httk.atomistic import (
    Cell,
    CellParams,
    CellParamsView,
    CellView,
    PlainCell,
    PlainSites,
    PlainSpecies,
    PlainSpeciesView,
    Sites,
    SitesView,
    Species,
    SpeciesView,
    UnitcellStructure,
)
from httk.atomistic.models.cell.backend import CellBackend
from httk.atomistic.models.cell.plain_view import PlainCellView
from httk.atomistic.models.sites.plain_view import PlainSitesView
from httk.atomistic.models.sites.backend import SitesBackend
from httk.atomistic.models.species.backend import SpeciesBackend
from httk.atomistic.storage.records import CellRecord, SitesRecord, SpeciesRecord

F = fractions.Fraction

ORTHO = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
# Hexagonal-ish cell: a = b in length, gamma = 120 degrees.
HEX = [[1.0, 0.0, 0.0], [-0.5, math.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 2.0]]

TOL = 1e-9


# --- Cell class ---


def test_cell_construction_and_validation() -> None:
    cell = Cell(ORTHO)
    # The exact matrix is a SurdVector; rational floats embed exactly and render back identically.
    assert cell.basis == SurdVector.create(ORTHO)
    assert cell.basis.to_floats() == [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
    with pytest.raises(ValueError):
        Cell([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        Cell([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_cell_derived_quantities_orthorhombic() -> None:
    cell = Cell(ORTHO)
    # Orthorhombic from exact (rational) data: lengths, angles and volume are all exact.
    assert cell.lengths == (
        SurdVector.create(2),
        SurdVector.create(3),
        SurdVector.create(4),
    )
    assert cell.angles == (F(90), F(90), F(90))
    assert cell.volume == SurdVector.create(24)


def test_cell_derived_quantities_hexagonal() -> None:
    cell = Cell(HEX)
    alpha, beta, gamma = cell.angles
    assert tuple(length.to_float() for length in cell.lengths) == pytest.approx((1.0, 1.0, 2.0), abs=TOL)
    # alpha between b,c; beta between a,c; gamma between a,b (120 for hexagonal). The HEX cell is
    # built from an irrational float (sqrt(3)/2), so gamma/volume are deterministic approximations.
    assert float(alpha) == pytest.approx(90.0, abs=TOL)
    assert float(beta) == pytest.approx(90.0, abs=TOL)
    assert float(gamma) == pytest.approx(120.0, abs=1e-6)
    assert cell.volume.to_float() == pytest.approx(math.sqrt(3.0), abs=TOL)


def test_cell_equality_and_repr() -> None:
    assert Cell(ORTHO) == Cell(ORTHO)
    assert Cell(ORTHO) != Cell(HEX)
    assert Cell(ORTHO) != object()
    assert "Cell(basis=" in repr(Cell(ORTHO))


# --- Cell dispatch and views ---


def test_cell_backend_dispatches_and_domain_adoption() -> None:
    assert isinstance(CellBackend.create(ORTHO), PlainCell)
    cell = Cell(ORTHO)
    assert isinstance(cell, CellBackend)
    assert CellView(cell)._backend is cell
    assert isinstance(CellBackend.create(ORTHO, kind="plain"), PlainCell)


def test_cell_backend_raises_for_malformed() -> None:
    with pytest.raises(TypeError):
        CellBackend.create([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(TypeError):
        CellBackend.create(12345)
    with pytest.raises(TypeError):
        CellBackend.create(Cell(ORTHO))


def test_cell_views_class_and_primitive() -> None:
    # Class view from a raw matrix (primitive backend).
    class_view = CellView(ORTHO)
    assert isinstance(class_view, Cell)
    assert class_view.basis.to_floats() == [
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0],
    ]
    assert class_view.volume == SurdVector.create(24)

    # Plain view from a Cell (class backend).
    primitive_view = PlainCellView(Cell(ORTHO))
    assert isinstance(primitive_view, tuple)
    assert tuple(primitive_view) == ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0))


def test_component_views_accept_storage_records_without_eager_exact_rebuild() -> None:
    cell_record = CellRecord(**project_storage_record(CellRecord, Cell(ORTHO)))
    cell_view = CellView(cell_record)
    assert "_unscaled_basis" not in cell_view.__dict__
    assert cell_view == Cell(ORTHO)
    assert unwrap(cell_view) is cell_record

    sites_record = SitesRecord(**project_storage_record(SitesRecord, Sites([[0, 0, 0]])))
    sites_view = SitesView(sites_record)
    assert "_reduced_coords" not in sites_view.__dict__
    assert sites_view == Sites([[0, 0, 0]])
    assert unwrap(sites_view) is sites_record

    species = Species("Fe", ("Fe",), (1,))
    species_record = SpeciesRecord(**SpeciesRecord.__httk_project__(species))
    species_view = SpeciesView(species_record)
    assert species_view == species
    assert unwrap(species_view) is species_record


def test_species_view_decodes_mixed_precision_record_sentinel() -> None:
    species = Species(
        "FeNi",
        ("Fe", "Ni"),
        (F(1, 2), F(1, 2)),
        concentration_precision=(None, F(1, 1000)),
    )
    record = SpeciesRecord(**SpeciesRecord.__httk_project__(species))

    assert SpeciesView(record).concentration_precision == (None, F(1, 1000))


def test_cell_view_rewrap_identity_and_unwrap() -> None:
    class_view = CellView(ORTHO)
    assert CellView(class_view) is class_view

    primitive_view = PlainCellView(Cell(ORTHO))
    assert PlainCellView(primitive_view) is primitive_view

    # unwrap returns the native raw object.
    assert unwrap(CellBackend.create(ORTHO)) is ORTHO
    cell = Cell(ORTHO)
    assert unwrap(cell) is cell
    assert unwrap(CellView(ORTHO)) is ORTHO


# --- Sites ---


def test_sites_construction_and_sequence_behavior() -> None:
    sites = Sites([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]])
    assert len(sites) == 3
    assert sites[1] == (0.5, 0.5, 0.5)
    assert list(sites) == [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)]
    assert sites.reduced_coords[0] == (0.0, 0.0, 0.0)
    assert sites.reduced_coords.to_floats() == [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
    ]
    with pytest.raises(ValueError):
        Sites([[0.0, 0.0]])


def test_sites_dispatch_and_views() -> None:
    raw = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    assert isinstance(SitesBackend.create(raw), PlainSites)
    sites = Sites(raw)
    assert isinstance(sites, SitesBackend)
    assert SitesView(sites)._backend is sites
    with pytest.raises(TypeError):
        SitesBackend.create([[0.0, 0.0]])

    class_view = SitesView(raw)
    assert isinstance(class_view, Sites)
    assert len(class_view) == 2

    primitive_view = PlainSitesView(Sites(raw))
    assert isinstance(primitive_view, tuple)
    assert tuple(primitive_view) == ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))

    assert SitesView(class_view) is class_view
    assert unwrap(SitesBackend.create(raw)) is raw


# --- Species ---


def test_species_construction_and_dispatch() -> None:
    species = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    optimade = {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]}

    assert isinstance(species, SpeciesBackend)
    assert SpeciesView(species)._backend is species
    assert isinstance(SpeciesBackend.create(optimade), PlainSpecies)
    assert isinstance(SpeciesBackend.create(optimade, kind="plain"), PlainSpecies)
    with pytest.raises(TypeError):
        SpeciesBackend.create("Fe", kind="plain")
    with pytest.raises(TypeError):
        SpeciesBackend.create(26, kind="plain")


def test_species_bare_symbol_and_atomic_number_inputs() -> None:
    for value in ("Fe", "X", "vacancy", 1, 118):
        symbol = value if isinstance(value, str) else ("H" if value == 1 else "Og")
        expected = Species(name=symbol, chemical_symbols=(symbol,), concentration=(1.0,))
        assert Species.create(value) == expected
        view = SpeciesView(value)
        assert view.name == symbol
        assert view.chemical_symbols == (symbol,)
        assert view.concentration == (1.0,)


def test_species_equality_and_hash_are_subclass_tolerant() -> None:
    plain = Species("Fe", ("Fe",), (1.0,))
    view = SpeciesView("Fe")

    assert view == plain
    assert plain == view
    assert hash(view) == hash(plain)
    assert view != Species("Fe", ("Fe",), (0.5,))
    assert view.__eq__(object()) is NotImplemented
    assert (view == object()) is False


def test_species_bare_input_validation() -> None:
    with pytest.raises(ValueError, match="chemical symbol"):
        Species.create("Zz")
    with pytest.raises(ValueError, match="atomic number"):
        Species.create(0)
    with pytest.raises(ValueError, match="atomic number"):
        Species.create(119)
    with pytest.raises(ValueError, match="bool"):
        Species.create(True)
    with pytest.raises(ValueError, match="bool"):
        SpeciesView(False)


def test_species_backend_raises_for_malformed() -> None:
    with pytest.raises(TypeError):
        SpeciesBackend.create(12345)
    with pytest.raises(TypeError):
        SpeciesBackend.create({"name": "Na"})  # missing required keys
    with pytest.raises(TypeError):
        SpeciesBackend.create({"name": 5, "chemical_symbols": ["Na"], "concentration": [1.0]})


def test_species_dict_to_class_roundtrip_with_optional_fields() -> None:
    optimade = {
        "name": "CH3",
        "chemical_symbols": ["C"],
        "concentration": [1.0],
        "mass": [12.011],
        "original_name": "methyl",
        "attached": ["H"],
        "nattached": [3],
    }
    # dict -> class (a genuine, fully validated Species)
    species_view = SpeciesView(optimade)
    assert isinstance(species_view, Species)
    assert species_view.name == "CH3"
    assert species_view.chemical_symbols == ("C",)
    assert species_view.concentration == (1.0,)
    assert species_view.mass == (12.011,)
    assert species_view.original_name == "methyl"
    assert species_view.attached == ("H",)
    assert species_view.nattached == (3,)

    # class -> dict, omitting None optional fields, with plain lists.
    species = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    payload = PlainSpeciesView(species)
    assert isinstance(payload, dict)
    assert payload == {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]}
    assert isinstance(payload["chemical_symbols"], list)
    assert "mass" not in payload
    assert "attached" not in payload

    # class -> dict, carrying the optional fields when present.
    full = PlainSpeciesView(SpeciesView(optimade))
    assert full["mass"] == [12.011]
    assert full["attached"] == ["H"]
    assert full["nattached"] == [3]
    assert full["original_name"] == "methyl"


def test_plain_species_view_is_detached_mutable_dict() -> None:
    species = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    payload = PlainSpeciesView(species)
    payload["name"] = "changed"
    # Mutating the detached dict does not affect the backend / original species.
    assert species.name == "Na"


def test_species_view_rewrap_and_unwrap() -> None:
    species = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    class_view = SpeciesView(species)
    assert SpeciesView(class_view) is class_view
    assert unwrap(species) is species

    optimade = {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]}
    assert unwrap(SpeciesBackend.create(optimade)) is optimade

    primitive_view = PlainSpeciesView(species)
    assert PlainSpeciesView(primitive_view) is primitive_view


def test_species_view_applies_full_validation() -> None:
    # A dict that passes the conservative primitive check but is not a valid Species.
    bad = {"name": "bad", "chemical_symbols": ["Zz"], "concentration": [1.0]}
    assert isinstance(SpeciesBackend.create(bad), PlainSpecies)  # conservative check passes
    with pytest.raises(ValueError):
        SpeciesView(bad)  # full validation rejects the unknown symbol


def test_cell_params_backend_constructs_standard_matrix() -> None:
    cubic = CellBackend.create((4.0, 4.0, 4.0, 90.0, 90.0, 90.0))
    assert isinstance(cubic, CellParams)
    for i, row in enumerate(cubic.basis.to_floats()):
        for j, x in enumerate(row):
            assert x == pytest.approx(4.0 if i == j else 0.0, abs=1e-12)

    hexagonal = CellBackend.create((3.0, 3.0, 5.0, 90.0, 90.0, 120.0))
    matrix = hexagonal.basis.to_floats()
    assert tuple(matrix[0]) == pytest.approx((3.0, 0.0, 0.0), abs=1e-12)
    assert matrix[1][0] == pytest.approx(-1.5)
    assert matrix[1][1] == pytest.approx(3.0 * math.sqrt(3.0) / 2.0)
    assert tuple(matrix[2]) == pytest.approx((0.0, 0.0, 5.0), abs=1e-12)


def test_cell_params_dispatch_and_kind_overrides() -> None:
    assert isinstance(CellBackend.create([1.0, 2.0, 3.0, 80.0, 85.0, 95.0]), CellParams)
    assert isinstance(CellBackend.create([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), PlainCell)
    with pytest.raises(TypeError):
        CellBackend.create((1.0, 2.0, 3.0, 80.0, 85.0, 95.0), kind="plain")
    with pytest.raises(TypeError):
        CellBackend.create([[1, 0, 0], [0, 1, 0], [0, 0, 1]], kind="params")


def test_cell_params_validation_errors() -> None:
    with pytest.raises(ValueError):
        CellBackend.create((0.0, 1.0, 1.0, 90.0, 90.0, 90.0))
    with pytest.raises(ValueError):
        CellBackend.create((1.0, 1.0, 1.0, 190.0, 90.0, 90.0))
    # Angles that cannot close into a parallelepiped.
    with pytest.raises(ValueError):
        CellBackend.create((1.0, 1.0, 1.0, 10.0, 10.0, 170.0))


def test_cell_params_view_from_params_backend_is_verbatim() -> None:
    raw = (2.0, 3.0, 4.0, 90.0, 90.0, 90.0)
    view = CellParamsView(raw)
    assert tuple(view) == raw
    assert (view.a, view.b, view.c) == (2.0, 3.0, 4.0)
    assert (view.alpha, view.beta, view.gamma) == (90.0, 90.0, 90.0)
    assert unwrap(view) is raw


def test_cell_params_view_of_rotated_matrix_is_lossy_but_faithful() -> None:
    # The same cubic cell rotated 90 degrees around z: different matrix, same parameters.
    rotated = Cell([[0.0, 4.0, 0.0], [-4.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    params = CellParamsView(rotated)
    assert tuple(params) == pytest.approx((4.0, 4.0, 4.0, 90.0, 90.0, 90.0))
    reconstructed = Cell(CellParams(tuple(params)).basis)
    assert reconstructed.basis != rotated.basis
    assert reconstructed.volume.to_float() == pytest.approx(rotated.volume.to_float())
    assert [length.to_float() for length in reconstructed.lengths] == pytest.approx(
        [length.to_float() for length in rotated.lengths]
    )
    assert [float(angle) for angle in reconstructed.angles] == pytest.approx([float(angle) for angle in rotated.angles])


def test_structure_from_cell_params() -> None:
    structure = UnitcellStructure(
        cell=(4.0, 4.0, 4.0, 90.0, 90.0, 90.0),
        sites=[[0.0, 0.0, 0.0]],
        species=[{"name": "Fe", "chemical_symbols": ["Fe"], "concentration": [1.0]}],
        species_at_sites=["Fe"],
    )
    assert isinstance(structure.cell, Cell)
    assert structure.cell.volume == SurdVector.create(64)
    assert tuple(CellParamsView(structure.cell)) == pytest.approx((4.0, 4.0, 4.0, 90.0, 90.0, 90.0))
