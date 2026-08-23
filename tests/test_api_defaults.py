"""Regression tests for template-method defaults on atomistic APIs."""

from collections.abc import Iterator
from fractions import Fraction

import pytest
from httk.core import SurdVector
from httk.core.storage import project_storage_record
from test_optimade_structure import _complete_attributes, _semantic_resource

from httk.atomistic import (
    Cell,
    CellParams,
    FundamentalDomainStructure,
    Sites,
    Spacegroup,
    Species,
    StructureBackend,
    StructureEntryProvider,
    SymopsStructure,
    UnitcellStructure,
    WyckoffSite,
)
from httk.atomistic.models.cell.backend import CellBackend
from httk.atomistic.models.sites.backend import SitesBackend
from httk.atomistic.models.species.backend import SpeciesBackend
from httk.atomistic.models.trajectory.backend import TrajectoryBackend
from httk.atomistic.storage.records import CellRecord, SitesRecord, SpeciesRecord

CELL_PARAMS = (3, 4, 5, 90, 60, 120)


def test_cell_geometry_defaults_match_every_backend() -> None:
    params = CellParams(CELL_PARAMS)
    source = Cell(params.basis)
    record = CellRecord(**project_storage_record(CellRecord, source))
    backends = (
        CellBackend._select_backend(params.basis, kind="plain"),
        CellBackend._select_backend(CELL_PARAMS, kind="params"),
        CellBackend._select_backend(record, kind="record"),
        source,
    )

    expected = (source.lengths, source.angles, source.volume, source.metric())
    for backend in backends:
        assert (backend.lengths, backend.angles, backend.volume, backend.metric()) == expected


def test_cell_params_uses_native_non_orthogonal_values() -> None:
    params = CellParams(CELL_PARAMS)
    assert params.lengths == tuple(SurdVector(value)._as_scalar() for value in CELL_PARAMS[:3])
    assert params.angles == tuple(Fraction(value) for value in CELL_PARAMS[3:])


def _frame(x: int) -> UnitcellStructure:
    return UnitcellStructure(
        Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        Sites([[x, 0, 0]]),
        [Species.from_object("Si")],
        ["Si"],
    )


class FramesOnlyTrajectory(TrajectoryBackend):
    """Minimal backend whose generator function provides a fresh traversal per call."""

    def __init__(self) -> None:
        self._values = (_frame(0), _frame(1), _frame(2))

    def frames(self) -> Iterator[UnitcellStructure]:
        yield from self._values

    @property
    def species(self) -> tuple[Species, ...]:
        return self._values[0].species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return self._values[0].species_at_sites


def test_trajectory_streaming_defaults() -> None:
    trajectory = FramesOnlyTrajectory()
    assert trajectory.nframes == 3
    assert trajectory.frame(0) is trajectory._values[0]
    assert trajectory.frame(2) is trajectory._values[2]
    assert trajectory.frame(-1) is trajectory._values[2]
    assert trajectory.frame(-2) is trajectory._values[1]
    assert trajectory.nframes == 3  # The generator function restarts traversal for each call.
    with pytest.raises(TypeError, match="must be an integer"):
        trajectory.frame(True)
    with pytest.raises(TypeError, match="must be an integer"):
        trajectory.frame(1.0)  # type: ignore[arg-type]
    with pytest.raises(IndexError):
        trajectory.frame(-4)
    with pytest.raises(IndexError):
        trajectory.frame(3)


def test_sites_and_species_api_defaults_cover_plain_and_record_backends() -> None:
    sites = Sites([[0, 0, 0], [Fraction(1, 2), 0, 0]])
    sites_record = SitesRecord(**SitesRecord.__httk_project__(sites))
    plain_sites = SitesBackend._select_backend(sites.reduced_coords.to_floats(), kind="plain")
    record_sites = SitesBackend._select_backend(sites_record, kind="record")
    assert plain_sites.num_sites == record_sites.num_sites == 2

    ordered = {"name": "Si", "chemical_symbols": ["Si"], "concentration": [1]}
    plain_species = SpeciesBackend._select_backend(ordered, kind="plain")
    species_record = SpeciesRecord(**SpeciesRecord.__httk_project__(Species.from_object("Si")))
    record_species = SpeciesBackend._select_backend(species_record, kind="record")
    assert plain_species.is_ordered is True
    assert record_species.is_ordered is True

    fractional = SpeciesBackend._select_backend(
        {"name": "FeNi", "chemical_symbols": ["Fe", "Ni"], "concentration": [Fraction(1, 2), Fraction(1, 2)]},
        kind="plain",
    )
    assert fractional.is_ordered is False


class MinimalStructure(StructureBackend):
    """Only the four canonical structure accessors are required."""

    def __init__(self, cell: Cell, sites: Sites, species: tuple[Species, ...], species_at_sites: tuple[str, ...]):
        self._cell = cell
        self._sites = sites
        self._species = species
        self._species_at_sites = species_at_sites

    @property
    def cell(self) -> Cell:
        return self._cell

    @property
    def sites(self) -> Sites:
        return self._sites

    @property
    def species(self) -> tuple[Species, ...]:
        return self._species

    @property
    def species_at_sites(self) -> tuple[str, ...]:
        return self._species_at_sites


def test_structure_api_defaults_match_unitcell_for_minimal_backend() -> None:
    cell = Cell([[3, 0, 0], [0, 4, 0], [0, 0, 5]])
    sites = Sites([[0, 0, 0], [Fraction(1, 2), 0, 0]])
    species = (Species.from_object("Si"), Species.from_object("O"))
    names = ("Si", "O")
    minimal = MinimalStructure(cell, sites, species, names)
    unitcell = UnitcellStructure(cell, sites, species, names)

    assert minimal.composition.amounts == unitcell.composition.amounts
    assert minimal.composition.complete == unitcell.composition.complete
    for name in (
        "elements",
        "nelements",
        "elements_ratios",
        "chemical_formula_descriptive",
        "chemical_formula_reduced",
        "chemical_formula_hill",
        "chemical_formula_anonymous",
        "dimension_types",
        "nperiodic_dimensions",
        "nsites",
        "implicit_atoms",
        "structure_features",
    ):
        assert getattr(minimal, name) == getattr(unitcell, name)


def test_structure_api_defaults_null_incomplete_composition_fields() -> None:
    unknown = Species("unknown", ("X",), (1,))
    structure = MinimalStructure(
        Cell([[3, 0, 0], [0, 3, 0], [0, 0, 3]]),
        Sites([[0, 0, 0]]),
        (unknown,),
        ("unknown",),
    )

    assert structure.composition.complete is False
    assert structure.elements is None
    assert structure.nelements is None
    assert structure.elements_ratios is None
    assert structure.chemical_formula_reduced is None
    assert structure.chemical_formula_anonymous is None


@pytest.mark.parametrize("kind", ("unitcell", "fundamental", "symops"))
def test_nperiodic_dimensions_default_matches_cell(kind: str) -> None:
    cell = Cell([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    species = (Species.from_object("Si"),)
    if kind == "unitcell":
        structure = UnitcellStructure(cell, Sites([[0, 0, 0]]), species, ("Si",))
    elif kind == "fundamental":
        structure = FundamentalDomainStructure(cell, Spacegroup.standard(221), (WyckoffSite("a", (), "Si"),), species)
    else:
        structure = SymopsStructure(cell, Sites([[0, 0, 0]]), species, ("Si",), ("x,y,z",))

    assert structure.nperiodic_dimensions == structure.cell.nperiodic_dimensions


def test_optimade_source_null_composition_fields_stay_null_when_sites_are_complete() -> None:
    attributes = _complete_attributes()
    attributes.update(
        {
            "elements": None,
            "nelements": None,
            "chemical_formula_reduced": None,
            "chemical_formula_anonymous": None,
        }
    )
    backend = StructureBackend._select_backend(_semantic_resource(attributes))
    record = next(iter(StructureEntryProvider({"remote": backend}).records("structures")))

    assert [
        record[name] for name in ("elements", "nelements", "chemical_formula_reduced", "chemical_formula_anonymous")
    ] == [
        None,
        None,
        None,
        None,
    ]


def test_optimade_elements_ratios_remain_source_values_at_serving_boundary() -> None:
    attributes = _complete_attributes()
    attributes.update(
        {
            "elements_ratios": [0.3333, 0.6666],
            "chemical_formula_reduced": "ClNa2",
            "chemical_formula_hill": "ClNa2",
            "chemical_formula_anonymous": "A2B",
            "structure_features": ["implicit_atoms"],
        }
    )
    backend = StructureBackend._select_backend(_semantic_resource(attributes))
    assert backend.elements_ratios == (Fraction(3333, 10000), Fraction(3333, 5000))
    record = next(iter(StructureEntryProvider({"remote": backend}).records("structures")))
    assert record["elements_ratios"] == pytest.approx([0.3333, 0.6666])
