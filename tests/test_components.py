import math

import pytest

from httk.atomistic import (
    Cell,
    CellBackend,
    CellClass,
    CellClassView,
    CellPrimitive,
    CellPrimitiveView,
    Sites,
    SitesBackend,
    SitesClass,
    SitesClassView,
    SitesPrimitive,
    SitesPrimitiveView,
    Species,
    SpeciesBackend,
    SpeciesClass,
    SpeciesClassView,
    SpeciesPrimitive,
    SpeciesPrimitiveView,
)
from httk.core import unwrap

ORTHO = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]
# Hexagonal-ish cell: a = b in length, gamma = 120 degrees.
HEX = [[1.0, 0.0, 0.0], [-0.5, math.sqrt(3.0) / 2.0, 0.0], [0.0, 0.0, 2.0]]

TOL = 1e-9


# --- Cell class ---


def test_cell_construction_and_validation() -> None:
    cell = Cell(ORTHO)
    assert cell.matrix == ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0))
    with pytest.raises(ValueError):
        Cell([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        Cell([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_cell_derived_quantities_orthorhombic() -> None:
    cell = Cell(ORTHO)
    lengths = cell.lengths
    angles = cell.angles
    assert lengths == pytest.approx((2.0, 3.0, 4.0), abs=TOL)
    assert angles == pytest.approx((90.0, 90.0, 90.0), abs=TOL)
    assert cell.volume == pytest.approx(24.0, abs=TOL)


def test_cell_derived_quantities_hexagonal() -> None:
    cell = Cell(HEX)
    lengths = cell.lengths
    alpha, beta, gamma = cell.angles
    assert lengths == pytest.approx((1.0, 1.0, 2.0), abs=TOL)
    # alpha between b,c; beta between a,c; gamma between a,b (120 for hexagonal).
    assert alpha == pytest.approx(90.0, abs=TOL)
    assert beta == pytest.approx(90.0, abs=TOL)
    assert gamma == pytest.approx(120.0, abs=TOL)
    assert cell.volume == pytest.approx(math.sqrt(3.0), abs=TOL)


def test_cell_equality_and_repr() -> None:
    assert Cell(ORTHO) == Cell(ORTHO)
    assert Cell(ORTHO) != Cell(HEX)
    assert Cell(ORTHO) != object()
    assert "Cell(matrix=" in repr(Cell(ORTHO))


# --- Cell dispatch and views ---


def test_cell_backend_dispatches_and_kind_overrides() -> None:
    assert isinstance(CellBackend.create(ORTHO), CellPrimitive)
    assert isinstance(CellBackend.create(Cell(ORTHO)), CellClass)
    assert isinstance(CellBackend.create(ORTHO, kind="primitive"), CellPrimitive)
    assert isinstance(CellBackend.create(Cell(ORTHO), kind="class"), CellClass)


def test_cell_backend_raises_for_malformed() -> None:
    with pytest.raises(TypeError):
        CellBackend.create([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(TypeError):
        CellBackend.create(12345)
    with pytest.raises(TypeError):
        CellBackend.create(Cell(ORTHO), kind="primitive")


def test_cell_views_class_and_primitive() -> None:
    # Class view from a raw matrix (primitive backend).
    class_view = CellClassView(ORTHO)
    assert isinstance(class_view, Cell)
    assert class_view.matrix == ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0))
    assert class_view.volume == pytest.approx(24.0, abs=TOL)

    # Primitive view from a Cell (class backend).
    primitive_view = CellPrimitiveView(Cell(ORTHO))
    assert isinstance(primitive_view, tuple)
    assert tuple(primitive_view) == ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0))


def test_cell_view_rewrap_identity_and_unwrap() -> None:
    class_view = CellClassView(ORTHO)
    assert CellClassView(class_view) is class_view

    primitive_view = CellPrimitiveView(Cell(ORTHO))
    assert CellPrimitiveView(primitive_view) is primitive_view

    # unwrap returns the native raw object.
    assert unwrap(CellBackend.create(ORTHO)) is ORTHO
    cell = Cell(ORTHO)
    assert unwrap(CellBackend.create(cell)) is cell


# --- Sites ---


def test_sites_construction_and_sequence_behavior() -> None:
    sites = Sites([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]])
    assert len(sites) == 3
    assert sites[1] == (0.5, 0.5, 0.5)
    assert list(sites) == [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)]
    assert sites.reduced_coords[0] == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        Sites([[0.0, 0.0]])


def test_sites_dispatch_and_views() -> None:
    raw = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    assert isinstance(SitesBackend.create(raw), SitesPrimitive)
    assert isinstance(SitesBackend.create(Sites(raw)), SitesClass)
    with pytest.raises(TypeError):
        SitesBackend.create([[0.0, 0.0]])

    class_view = SitesClassView(raw)
    assert isinstance(class_view, Sites)
    assert len(class_view) == 2

    primitive_view = SitesPrimitiveView(Sites(raw))
    assert isinstance(primitive_view, tuple)
    assert tuple(primitive_view) == ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))

    assert SitesClassView(class_view) is class_view
    assert unwrap(SitesBackend.create(raw)) is raw


# --- Species ---


def test_species_class_construction_and_dispatch() -> None:
    species = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    optimade = {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]}

    assert isinstance(SpeciesBackend.create(species), SpeciesClass)
    assert isinstance(SpeciesBackend.create(optimade), SpeciesPrimitive)
    assert isinstance(SpeciesBackend.create(species, kind="class"), SpeciesClass)
    assert isinstance(SpeciesBackend.create(optimade, kind="primitive"), SpeciesPrimitive)


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
    species_view = SpeciesClassView(optimade)
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
    payload = SpeciesPrimitiveView(species)
    assert isinstance(payload, dict)
    assert payload == {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]}
    assert isinstance(payload["chemical_symbols"], list)
    assert "mass" not in payload
    assert "attached" not in payload

    # class -> dict, carrying the optional fields when present.
    full = SpeciesPrimitiveView(SpeciesClassView(optimade))
    assert full["mass"] == [12.011]
    assert full["attached"] == ["H"]
    assert full["nattached"] == [3]
    assert full["original_name"] == "methyl"


def test_species_primitive_view_is_detached_mutable_dict() -> None:
    species = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    payload = SpeciesPrimitiveView(species)
    payload["name"] = "changed"
    # Mutating the detached dict does not affect the backend / original species.
    assert species.name == "Na"


def test_species_view_rewrap_and_unwrap() -> None:
    species = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    class_view = SpeciesClassView(species)
    assert SpeciesClassView(class_view) is class_view
    assert unwrap(SpeciesBackend.create(species)) is species

    optimade = {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]}
    assert unwrap(SpeciesBackend.create(optimade)) is optimade

    primitive_view = SpeciesPrimitiveView(species)
    assert SpeciesPrimitiveView(primitive_view) is primitive_view


def test_species_class_view_applies_full_validation() -> None:
    # A dict that passes the conservative primitive check but is not a valid Species.
    bad = {"name": "bad", "chemical_symbols": ["Zz"], "concentration": [1.0]}
    assert isinstance(SpeciesBackend.create(bad), SpeciesPrimitive)  # conservative check passes
    with pytest.raises(ValueError):
        SpeciesClassView(bad)  # full validation rejects the unknown symbol
