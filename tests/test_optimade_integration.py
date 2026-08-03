"""Integration test: serve httk-atomistic structures through httk-serve.

This exercises the full path from :class:`~httk.atomistic.StructureEntryProvider`
to an OPTIMADE query, using ``adapter_from_providers`` from *httk-serve*.

*httk-serve* is an optional peer distribution, not a dependency of
*httk-atomistic*, so this test is gated by ``pytest.importorskip``. In the
workspace it runs when httk-serve's source is on the path
(``PYTHONPATH=src:../httk-serve/src``); in this module's own CI (where
httk-serve is not installed) it is skipped.
"""

import pytest

pytest.importorskip("httk.serve.optimade")

from httk.serve.optimade import adapter_from_providers  # noqa: E402
from httk.serve.optimade.backend import execute_query  # noqa: E402
from httk.serve.optimade.filter import parse_optimade_filter  # noqa: E402

from httk.atomistic import UnitcellStructure, StructureEntryProvider  # noqa: E402
from httk.atomistic.models.species.species import Species  # noqa: E402


def _nacl(sid_cell: list[list[float]]) -> UnitcellStructure:
    na = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    cl = Species(name="Cl", chemical_symbols=("Cl",), concentration=(1.0,))
    return UnitcellStructure(sid_cell, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], [na, cl], ["Na", "Cl"])


def _single(cell: list[list[float]]) -> UnitcellStructure:
    si = Species(name="Si", chemical_symbols=("Si",), concentration=(1.0,))
    return UnitcellStructure(cell, [[0.0, 0.0, 0.0]], [si], ["Si"])


def _provider() -> StructureEntryProvider:
    return StructureEntryProvider(
        {
            "nacl": _nacl([[3.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),
            "si": _single([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
        }
    )


def test_adapter_from_provider_serves_structures() -> None:
    adapter = adapter_from_providers([_provider()])
    assert adapter.schema.all_entries == ("structures",)
    results = list(execute_query(adapter, ["structures"], ["id", "type", "nelements"], [], 100, 0))
    assert {r.values["id"] for r in results} == {"nacl", "si"}


def test_filtered_query_returns_expected_served_fields() -> None:
    adapter = adapter_from_providers([_provider()])
    response_fields = ["id", "type", "species_at_sites", "lattice_vectors", "cartesian_site_positions", "species"]
    results = list(
        execute_query(
            adapter,
            ["structures"],
            response_fields,
            [],
            100,
            0,
            parse_optimade_filter("nelements = 2"),
        )
    )
    assert len(results) == 1
    entry = results[0].values
    assert entry["id"] == "nacl"
    assert entry["species_at_sites"] == ["Na", "Cl"]
    assert entry["lattice_vectors"] == [[3.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    # cartesian = sum_k reduced[k] * cell.basis[k]; second site [0.5,0.5,0.5]:
    assert entry["cartesian_site_positions"] == [[0.0, 0.0, 0.0], [2.0, 1.5, 1.5]]
    assert {s["name"] for s in entry["species"]} == {"Na", "Cl"}


def test_elements_filter_selects_structure() -> None:
    adapter = adapter_from_providers([_provider()])
    results = list(
        execute_query(
            adapter,
            ["structures"],
            ["id"],
            [],
            100,
            0,
            parse_optimade_filter('elements HAS "Si"'),
        )
    )
    assert [r.values["id"] for r in results] == ["si"]


def _periodicity_provider() -> StructureEntryProvider:
    from httk.atomistic import Cell

    si = Species(name="Si", chemical_symbols=("Si",), concentration=(1.0,))
    basis = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]]

    def structure(periodicity: tuple[int, int, int]) -> UnitcellStructure:
        return UnitcellStructure(Cell(basis, periodicity=periodicity), [[0.0, 0.0, 0.0]], [si], ["Si"])

    return StructureEntryProvider(
        {
            "bulk": structure((1, 1, 1)),
            "slab": structure((1, 1, 0)),
            "wire": structure((0, 0, 1)),
            "molecule": structure((0, 0, 0)),
        }
    )


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("nperiodic_dimensions = 3", {"bulk"}),
        ("nperiodic_dimensions <= 2", {"slab", "wire", "molecule"}),
        ("nperiodic_dimensions = 0", {"molecule"}),
    ],
)
def test_periodicity_is_queryable(expression: str, expected: set[str]) -> None:
    """`nperiodic_dimensions` carries query-support 'all mandatory' in the standard.

    The middle case is OPTIMADE's own documented query example, and it used to return
    nothing at all because every structure claimed to be a 3D crystal.
    """
    adapter = adapter_from_providers([_periodicity_provider()])
    results = list(execute_query(adapter, ["structures"], ["id"], [], 100, 0, parse_optimade_filter(expression)))
    assert {r.values["id"] for r in results} == expected


def test_dimension_types_is_served_through_the_full_path() -> None:
    adapter = adapter_from_providers([_periodicity_provider()])
    results = list(
        execute_query(
            adapter,
            ["structures"],
            ["id", "dimension_types", "nperiodic_dimensions", "site_coordinate_span"],
            [],
            100,
            0,
        )
    )
    served = {r.values["id"]: r.values for r in results}
    assert served["slab"]["dimension_types"] == [1, 1, 0]
    assert served["slab"]["nperiodic_dimensions"] == 2
    assert served["slab"]["site_coordinate_span"] == "unit_cell"
    assert served["molecule"]["dimension_types"] == [0, 0, 0]
    assert served["molecule"]["site_coordinate_span"] == "unit_cell"
