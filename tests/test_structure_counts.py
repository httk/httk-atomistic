"""Served ``_httk_natoms`` and ``_httk_nformula_units`` reference counts."""

from collections import Counter
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import pytest
from httk.core import FracVector

from httk.atomistic import (
    ASUStructure,
    ChemicalComposition,
    Spacegroup,
    Species,
    StructureEntry,
    StructureEntryProvider,
    UnitcellStructure,
    WyckoffSite,
)
from httk.atomistic.entries.counts import count_properties, formula_unit_counts

_HALF = Fraction(1, 2)
_NA = [[0, 0, 0], [0, _HALF, _HALF], [_HALF, 0, _HALF], [_HALF, _HALF, 0]]
_CL = [[_HALF, _HALF, _HALF], [_HALF, 0, 0], [0, _HALF, 0], [0, 0, _HALF]]
_CELL = [[Fraction(563, 100), 0, 0], [0, Fraction(563, 100), 0], [0, 0, Fraction(563, 100)]]


def _nacl() -> UnitcellStructure:
    species = (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,)))
    return UnitcellStructure(_CELL, _NA + _CL, species, ["Na"] * 4 + ["Cl"] * 4)


def _disordered() -> UnitcellStructure:
    species = (Species("NaK", ("Na", "K"), (_HALF, _HALF)), Species("Cl", ("Cl",), (1,)))
    return UnitcellStructure(_CELL, _NA + _CL, species, ["NaK"] * 4 + ["Cl"] * 4)


def _vacancy() -> UnitcellStructure:
    species = (Species("Na", ("Na",), (1,)), Species("Va", ("vacancy",), (1,)))
    return UnitcellStructure(_CELL, _NA[:2], species, ["Na", "Va"])


def _nacl_asu() -> ASUStructure:
    return ASUStructure(
        _CELL,
        225,
        (WyckoffSite("a", FracVector(()), "Na"), WyckoffSite("b", FracVector(()), "Cl")),
        (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
    )


_SOURCES: dict[str, Any] = {
    "nacl": _nacl(),
    "disordered": _disordered(),
    "vacancy": _vacancy(),
    "asu": _nacl_asu(),
    "empty": None,
}
_EXPECTED = {
    "nacl": (8, 4),
    "disordered": (None, None),
    "vacancy": (None, None),
    "asu": (8, 4),
    "empty": (None, None),
}


_HEX = [[4, 0, 0], [-2, 2 * Fraction(1732, 1000), 0], [0, 0, 12]]


def _bi_asu(*, transform: Any = None, chemical_composition: ChemicalComposition | None = None) -> ASUStructure:
    return ASUStructure(
        _HEX,
        166,
        (WyckoffSite("a", FracVector(()), "Bi"),),
        (Species("Bi", ("Bi",), (1,)),),
        transform=transform,
        chemical_composition=chemical_composition,
    )


def _live_and_stored(source: Any) -> tuple[tuple[object, object], tuple[object, object]]:
    pytest.importorskip("httk.store.backend.sql")
    from httk.store import Backend, EntryIdScheme, SqlStore
    from httk.store.backend.sql import stored_property_sql_plan

    from httk.atomistic import ASUStructureRecord

    (live,) = _counts(StructureEntryProvider({"x": source}).records("structures"), "__id").values()
    with Backend.sqlite() as database:
        store = SqlStore(
            database, entry_ids=EntryIdScheme("httk.test", "1"), entry_records={StructureEntry: ASUStructureRecord}
        )
        store.save(source)
        ((stored, _),) = Counter(
            (row["_httk_natoms"], row["_httk_nformula_units"])
            for row in stored_property_sql_plan(store, StructureEntry).records()
        ).items()
    return live, stored


def _counts(rows: Any, id_key: str) -> dict[str, tuple[object, object]]:
    return {row[id_key]: (row["_httk_natoms"], row["_httk_nformula_units"]) for row in rows}


def test_structure_entry_serves_counts_for_ordered_structures_only() -> None:
    provider = StructureEntryProvider(_SOURCES)
    assert _counts(provider.records("structures"), "__id") == _EXPECTED
    assert {"_httk_natoms", "_httk_nformula_units"} <= set(provider.property_keys("structures"))


def test_stored_backings_serve_the_same_counts_without_expansion() -> None:
    pytest.importorskip("httk.store.backend.sql")
    from httk.store import Backend, EntryIdScheme, SqlStore
    from httk.store.backend.sql import stored_property_sql_plan
    from httk.store.query.optimade_filters import FilterTranslationError

    from httk.atomistic import ASUStructureRecord, FundamentalDomainStructureRecord, UnitcellStructureRecord

    database = Backend.sqlite()
    with database:
        store = SqlStore(
            database,
            entry_ids=EntryIdScheme("httk.test", "1"),
            entry_records={
                StructureEntry: (UnitcellStructureRecord, FundamentalDomainStructureRecord, ASUStructureRecord)
            },
        )
        for source in _SOURCES.values():
            if source is not None:
                store.save(source)
        plan = stored_property_sql_plan(store, StructureEntry)
        served = Counter(
            (row["site_coordinate_span"], row["_httk_natoms"], row["_httk_nformula_units"]) for row in plan.records()
        )
        assert served == Counter({("asymmetric_unit", 8, 4): 1, ("unit_cell", 8, 4): 1, ("unit_cell", None, None): 2})
        # The definitions declare query-support none: filtering is not implemented.
        with pytest.raises(FilterTranslationError):
            plan.filter_searchers("_httk_natoms = 8")


def test_count_definitions_are_served_as_integers() -> None:
    properties = StructureEntry.entry_type_definition().properties
    for name in ("_httk_natoms", "_httk_nformula_units"):
        assert properties[name].as_optimade()["x-optimade-type"] == "integer"


def test_formula_unit_counts_relate_to_the_reduced_formula() -> None:
    assert formula_unit_counts(20, "FeO3Sm") == {"_httk_natoms": 20, "_httk_nformula_units": 4}
    assert formula_unit_counts(12, "Al2O3") == {"_httk_natoms": 12, "_httk_nformula_units": None}
    assert formula_unit_counts(None, "ClNa")["_httk_nformula_units"] is None
    assert formula_unit_counts(2, None) == {"_httk_natoms": 2, "_httk_nformula_units": None}


def test_unknown_structure_features_serve_null_counts() -> None:
    remote = SimpleNamespace(
        structure_features=None,
        species=(Species("Si", ("Si",), (1,)),),
        site_coordinate_span="unit_cell",
        nsites=1,
        chemical_formula_reduced="Si",
    )
    assert count_properties(remote) == {"_httk_natoms": None, "_httk_nformula_units": None}


def test_volume_changing_setting_transform_counts_the_own_cell() -> None:
    # 166 hexagonal axes: Wyckoff 3a; the rhombohedral own cell holds one of those three atoms.
    assert _live_and_stored(_bi_asu()) == ((3, 3), (3, 3))
    rhombohedral = _bi_asu(transform=Spacegroup.from_setting("166:R").transform_from_standard)
    assert _live_and_stored(rhombohedral) == ((1, 1), (1, 1))


def test_stored_domain_with_declared_composition_serves_null_counts() -> None:
    # Accepted ceiling: the stored domain path sums the normalized composition, which a
    # declared composition replaces, so it serves null rather than expanding at serve time.
    source = _bi_asu(chemical_composition=ChemicalComposition({"Bi": 3}, mode="full"))
    assert _live_and_stored(source) == ((3, 3), (None, None))
