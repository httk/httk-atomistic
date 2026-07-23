"""Integration test: serve httk-atomistic structures through httk-optimade.

This exercises the full path from :class:`~httk.atomistic.StructureEntryProvider`
to an OPTIMADE query, using ``adapter_from_providers`` from *httk-optimade*.

*httk-optimade* is an optional peer distribution, not a dependency of
*httk-atomistic*, so this test is gated by ``pytest.importorskip``. In the
workspace it runs when httk-optimade's source is on the path
(``PYTHONPATH=src:../httk-optimade/src``); in this module's own CI (where
httk-optimade is not installed) it is skipped.
"""

import pytest

pytest.importorskip("httk.optimade")

from httk.optimade import adapter_from_providers  # noqa: E402
from httk.optimade.backend import execute_query  # noqa: E402
from httk.optimade.filter import parse_optimade_filter  # noqa: E402

from httk.atomistic import Structure, StructureEntryProvider  # noqa: E402
from httk.atomistic.species import Species  # noqa: E402


def _nacl(sid_cell: list[list[float]]) -> Structure:
    na = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    cl = Species(name="Cl", chemical_symbols=("Cl",), concentration=(1.0,))
    return Structure(sid_cell, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], [na, cl], ["Na", "Cl"])


def _single(cell: list[list[float]]) -> Structure:
    si = Species(name="Si", chemical_symbols=("Si",), concentration=(1.0,))
    return Structure(cell, [[0.0, 0.0, 0.0]], [si], ["Si"])


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
