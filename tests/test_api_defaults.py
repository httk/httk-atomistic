"""Regression tests for template-method defaults on atomistic APIs."""

from collections.abc import Iterator
from fractions import Fraction

import pytest
from httk.core import SurdVector
from httk.core.storage import project_storage_record

from httk.atomistic import Cell, CellParams, Sites, Species, UnitcellStructure
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
        CellBackend.create(params.basis, kind="plain"),
        CellBackend.create(CELL_PARAMS, kind="params"),
        CellBackend.create(record, kind="record"),
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
        [Species.create("Si")],
        ["Si"],
    )


class FramesOnlyTrajectory(TrajectoryBackend):
    """Minimal backend exercising the streaming API defaults."""

    def __init__(self) -> None:
        self._values = (_frame(0), _frame(1), _frame(2))

    def frames(self) -> Iterator[UnitcellStructure]:
        return iter(self._values)

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
    with pytest.raises(IndexError):
        trajectory.frame(-1)
    with pytest.raises(IndexError):
        trajectory.frame(3)


def test_sites_and_species_api_defaults_cover_plain_and_record_backends() -> None:
    sites = Sites([[0, 0, 0], [Fraction(1, 2), 0, 0]])
    sites_record = SitesRecord(**SitesRecord.__httk_project__(sites))
    plain_sites = SitesBackend.create(sites.reduced_coords.to_floats(), kind="plain")
    record_sites = SitesBackend.create(sites_record, kind="record")
    assert plain_sites.num_sites == record_sites.num_sites == 2

    ordered = {"name": "Si", "chemical_symbols": ["Si"], "concentration": [1]}
    plain_species = SpeciesBackend.create(ordered, kind="plain")
    species_record = SpeciesRecord(**SpeciesRecord.__httk_project__(Species.create("Si")))
    record_species = SpeciesBackend.create(species_record, kind="record")
    assert plain_species.is_ordered is True
    assert record_species.is_ordered is True

    fractional = SpeciesBackend.create(
        {"name": "FeNi", "chemical_symbols": ["Fe", "Ni"], "concentration": [Fraction(1, 2), Fraction(1, 2)]},
        kind="plain",
    )
    assert fractional.is_ordered is False
