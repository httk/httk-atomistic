"""Validate the structures provider's actual rows against its served schemas."""

from fractions import Fraction

import pytest
from httk.core import PropertyDefinition
from test_structure_record import _structure
from test_symmetry_entries import _rocksalt

from httk.atomistic import Cell, CellParams, StructureEntryProvider


def test_structure_provider_rows_validate_against_served_definition() -> None:
    pytest.importorskip("httk.data")
    from httk.data.validation import PropertyValidationError, validate_record

    energy = PropertyDefinition.from_simple("_httk_total_energy", description="E", fulltype="float")
    provider = StructureEntryProvider(
        {
            "rational": _structure(
                Cell(
                    [[4, 0, 0], [0, 5, 0], [0, 0, 6]],
                    precision=Fraction(1, 10000),
                    periodicity=(True, True, True),
                )
            ),
            "hexagonal-2d": _structure(
                Cell(CellParams((1, 1, 3, 90, 90, 120)).basis, periodicity=(True, True, False))
            ),
            "rocksalt": _rocksalt(),
        },
        extra_definitions={"_httk_total_energy": energy},
        properties={"rational": {"_httk_total_energy": -1.5}},
    )
    definition = provider.entry_types()["structures"]
    property_keys = provider.property_keys("structures")
    rows = list(provider.records("structures"))

    served = {row["__id"]: {name: row[key] for name, key in property_keys.items()} for row in rows}
    assert served["rational"]["_httk_coordinate_precision"] == 0.001
    assert served["rational"]["_httk_basis_precision"] == 0.0001
    assert served["rational"]["_httk_crystal_system"] is None
    assert served["rational"]["_httk_centring_type"] is None
    assert served["rational"]["_httk_total_energy"] == -1.5
    assert served["rocksalt"]["_httk_setting_it_nc"] == "225"
    assert set(served["rocksalt"]["wyckoff_positions"]) == {"a", "b"}
    assert served["hexagonal-2d"]["dimension_types"] == [1, 1, 0]
    assert served["hexagonal-2d"]["nperiodic_dimensions"] == 2
    assert all(isinstance(record["structure_features"], list) for record in served.values())

    failures: list[str] = []
    for entry_id, record in served.items():
        try:
            validate_record(definition, record)
        except PropertyValidationError as exc:
            failures.append(f"{entry_id}: {exc}")

    assert not failures, "\n".join(failures)
