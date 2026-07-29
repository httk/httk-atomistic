import pytest
from httk.core import FracVector, unwrap

from httk.atomistic import (
    ASUSite,
    ASUStructure,
    Cell,
    CellClassView,
    CellNumericView,
    NumericUnitcellStructureView,
    Sites,
    SitesClassView,
    SitesNumericView,
    Species,
    Structure,
    StructureBackend,
    UnitcellStructureView,
)
from httk.atomistic.cell_primitive import CellPrimitive
from httk.atomistic.sites_primitive import SitesPrimitive


class ProbeCellPrimitive(CellPrimitive):
    unscaled_basis_calls = 0

    @property
    def unscaled_basis(self):
        self.unscaled_basis_calls += 1
        return super().unscaled_basis


class ProbeSitesPrimitive(SitesPrimitive):
    reduced_coords_calls = 0

    @property
    def reduced_coords(self):
        self.reduced_coords_calls += 1
        return super().reduced_coords


CUBE = [[2, 0, 0], [0, 3, 0], [0, 0, 4]]
COORDS = [[0, 0, 0], ["1/2", "1/2", "1/2"]]


class CountingStructureBackend(StructureBackend):
    def __init__(
        self, species_at_sites: tuple[str, ...] = ("Na", "Cl"), sites: Sites | None = None
    ) -> None:
        self.calls = {"cell": 0, "sites": 0, "species": 0, "species_at_sites": 0}
        self._cell = Cell(CUBE)
        self._sites = Sites(COORDS) if sites is None else sites
        self._species = (
            Species("Na", ("Na",), (1.0,)),
            Species("Cl", ("Cl",), (1.0,)),
        )
        self._species_at_sites = species_at_sites

    @property
    def cell(self) -> Cell:
        self.calls["cell"] += 1
        return self._cell

    @property
    def sites(self) -> Sites:
        self.calls["sites"] += 1
        return self._sites

    @property
    def species(self) -> tuple[Species, ...]:
        self.calls["species"] += 1
        return self._species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        self.calls["species_at_sites"] += 1
        return self._species_at_sites


def test_cell_class_view_fills_backend_call_groups_lazily() -> None:
    probe = ProbeCellPrimitive(CUBE)
    view = CellClassView(probe)

    assert probe.unscaled_basis_calls == 0
    assert unwrap(view) is probe.unwrap()
    assert probe.unscaled_basis_calls == 0
    _ = view.basis
    _ = view.basis
    assert probe.unscaled_basis_calls == 1
    _ = view.scale
    assert probe.unscaled_basis_calls == 1


def test_sites_class_view_fills_backend_call_groups_lazily() -> None:
    probe = ProbeSitesPrimitive(COORDS)
    precision_view = SitesClassView(probe)

    assert precision_view.precision is None
    assert probe.reduced_coords_calls == 0

    view = SitesClassView(probe)
    _ = view.reduced_coords
    _ = view.reduced_coords
    assert probe.reduced_coords_calls == 1


def test_numeric_views_defer_their_exact_presentation() -> None:
    pytest.importorskip("numpy")

    cell_probe = ProbeCellPrimitive(CUBE)
    cell_view = CellNumericView(cell_probe)
    assert "_cell" not in cell_view.__dict__
    assert cell_probe.unscaled_basis_calls == 0
    _ = cell_view.basis
    _ = cell_view.basis
    assert cell_probe.unscaled_basis_calls == 1

    sites_probe = ProbeSitesPrimitive(COORDS)
    sites_view = SitesNumericView(sites_probe)
    assert "_sites" not in sites_view.__dict__
    assert sites_probe.reduced_coords_calls == 0
    _ = sites_view.reduced_coords
    _ = sites_view.reduced_coords
    assert sites_probe.reduced_coords_calls == 1


def test_cell_class_view_defers_degenerate_basis_validation() -> None:
    probe = ProbeCellPrimitive([[1, 0, 0], [2, 0, 0], [0, 0, 1]])
    view = CellClassView(probe)

    assert "_unscaled_basis" not in view.__dict__
    with pytest.raises(ValueError, match="non-degenerate"):
        _ = view.basis
    with pytest.raises(ValueError, match="non-degenerate"):
        _ = view.basis
    assert "_unscaled_basis" not in view.__dict__


def test_unmaterialized_cell_class_view_round_trips_and_compares() -> None:
    probe = ProbeCellPrimitive(CUBE)
    view = CellClassView(probe)

    assert "_unscaled_basis" not in view.__dict__
    assert view == Cell(CUBE)


def test_unitcell_structure_view_fills_components_on_first_access() -> None:
    probe = CountingStructureBackend()
    view = UnitcellStructureView(probe)

    assert probe.calls == {"cell": 0, "sites": 0, "species": 0, "species_at_sites": 0}
    _ = view.cell
    assert probe.calls == {"cell": 1, "sites": 0, "species": 0, "species_at_sites": 0}
    _ = view.cell
    _ = view.species
    assert probe.calls == {"cell": 1, "sites": 0, "species": 1, "species_at_sites": 0}
    _ = view.species
    _ = view.sites
    _ = view.sites
    assert probe.calls == {"cell": 1, "sites": 1, "species": 1, "species_at_sites": 1}


def test_asu_unitcell_view_does_not_expand_until_sites() -> None:
    asu = ASUStructure(
        CUBE,
        225,
        [ASUSite("a", FracVector.create(()), "Na")],
        [Species("Na", ("Na",), (1.0,))],
    )
    view = UnitcellStructureView(asu)

    assert "_expansion" not in asu.__dict__
    _ = view.cell
    _ = view.species
    assert "_expansion" not in asu.__dict__
    _ = view.sites
    assert "_expansion" in asu.__dict__
    expansion = asu.__dict__["_expansion"]
    _ = view.sites
    assert asu.__dict__["_expansion"] is expansion


def test_numeric_unitcell_structure_view_defers_exact_structure() -> None:
    pytest.importorskip("numpy")
    view = NumericUnitcellStructureView(CountingStructureBackend())

    assert "_exact" not in view.__dict__
    _ = view.species
    _ = view.species_at_sites
    assert "_exact" not in view.__dict__
    _ = view.cell
    assert "_exact" in view.__dict__
    exact = view.exact
    assert view.exact is exact


def test_structure_view_defers_species_name_validation() -> None:
    bad = CountingStructureBackend(("Missing", "Cl"))
    view = UnitcellStructureView(bad)

    with pytest.raises(ValueError) as error:
        _ = view.species_at_sites
    assert str(error.value) == "Structure species_at_sites references unknown species name: 'Missing'"

    cell_only = UnitcellStructureView(CountingStructureBackend(("Missing", "Cl")))
    assert cell_only.cell.basis.to_floats() == [
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0],
    ]

    with pytest.raises(ValueError) as eager_error:
        Structure(CUBE, COORDS, bad._species, ("Missing", "Cl"))
    assert str(eager_error.value) == str(error.value)


def test_structure_validation_precedence_stays_eager_but_views_are_per_component() -> None:
    species = (
        Species("Na", ("Na",), (1.0,)),
        Species("Cl", ("Cl",), (1.0,)),
    )
    one_site = [[0, 0, 0]]
    with pytest.raises(ValueError) as eager_error:
        Structure(CUBE, one_site, species, ("Missing", "Na"))
    assert str(eager_error.value) == "Structure species_at_sites must have the same length as sites"

    bad_name_view = UnitcellStructureView(
        CountingStructureBackend(("Missing", "Na"), Sites(one_site))
    )
    with pytest.raises(ValueError) as membership_error:
        _ = bad_name_view.species_at_sites
    assert str(membership_error.value) == "Structure species_at_sites references unknown species name: 'Missing'"

    bad_length_view = UnitcellStructureView(
        CountingStructureBackend(("Na", "Na"), Sites(one_site))
    )
    with pytest.raises(ValueError) as length_error:
        _ = bad_length_view.sites
    assert str(length_error.value) == "Structure species_at_sites must have the same length as sites"
