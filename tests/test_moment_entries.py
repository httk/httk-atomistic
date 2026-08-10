from pathlib import Path

import pytest

from httk.atomistic import (
    CartesianSiteMoments,
    CollinearSiteMoments,
    Species,
    StructureEntryProvider,
    UnitcellStructure,
)
from httk.atomistic.entries.moments import MOMENT_PROPERTY_KEYS, moment_definitions
from httk.atomistic.entries.structures import StructureEntry


def _structure(moments=None) -> UnitcellStructure:
    return UnitcellStructure(
        [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        [[0, 0, 0], [1, 1, 1]],
        (Species("Fe", ("Fe",), (1,)),),
        ("Fe", "Fe"),
        site_moments=moments,
    )


def _record(structure) -> dict:
    return next(iter(StructureEntryProvider({"x": structure}).records("structures")))


@pytest.mark.parametrize(
    ("moments", "served", "magnetic"),
    (
        (CartesianSiteMoments([[1, 2, 3], [-1, 0, 1]]), [[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], True),
        (CollinearSiteMoments([1, -1]), None, True),
        (None, None, False),
    ),
)
def test_structure_entry_serves_site_moments_and_feature(moments, served, magnetic) -> None:
    record = _record(_structure(moments))
    assert record["_httk_site_moments"] == served
    assert ("_httk_magnetism" in record["structure_features"]) is magnetic


def test_magnetic_mcif_serves_expanded_cartesian_moments() -> None:
    pytest.importorskip("httk.io")
    from httk.core import load

    structure = load(str(Path(__file__).with_name("fixtures") / "magnetic_centered.mcif"))
    record = _record(structure)
    assert record["_httk_site_moments"] == [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    assert "_httk_magnetism" in record["structure_features"]


def test_site_moment_definition_is_vendored_and_extended() -> None:
    definition = moment_definitions()["_httk_site_moments"]
    assert definition.as_optimade()["$id"] == "https://schemas.httk.org/defs/v0.1/properties/magnetism/site_moments"
    extended = StructureEntry.entry_type_definition()
    assert "_httk_site_moments" in extended.properties
    assert set(MOMENT_PROPERTY_KEYS) <= set(extended.properties)


def test_moment_record_validates_when_httk_store_is_available() -> None:
    pytest.importorskip("httk.store")
    from httk.store.validation import validate_record

    provider = StructureEntryProvider({"x": _structure(CartesianSiteMoments([[1, 2, 3], [-1, 0, 1]]))})
    keys = provider.property_keys("structures")
    row = next(iter(provider.records("structures")))
    validate_record(provider.entry_types()["structures"], {name: row[key] for name, key in keys.items()})
