"""Unit tests for StructureEntryProvider and its registration."""

import pytest

from httk.atomistic import Structure, StructureEntryProvider
from httk.atomistic.species import Species
from httk.core import PropertyDefinition


def _nacl_like() -> Structure:
    # A non-orthogonal cell (rows are the lattice vectors).
    cell = [[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    sites = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    na = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    cl = Species(name="Cl", chemical_symbols=("Cl",), concentration=(1.0,))
    return Structure(cell, sites, [na, cl], ["Na", "Cl"])


def _provider() -> StructureEntryProvider:
    return StructureEntryProvider({"s-1": _nacl_like()})


def test_entry_types_describe_structures() -> None:
    from httk.core import EntryTypeDefinition

    entry_types = _provider().entry_types()
    assert set(entry_types) == {"structures"}
    definition = entry_types["structures"]
    assert isinstance(definition, EntryTypeDefinition)
    properties = definition.properties
    # The vendored standard describes the full v1.3 property set (30), a superset
    # of the subset the provider serves:
    assert len(properties) == 30
    for name in ("id", "type", "elements", "nelements", "nsites", "species", "structure_features"):
        assert name in properties
    # Includes v1.3-native properties the provider does not serve:
    assert "wyckoff_positions" in properties
    assert "fractional_site_positions" in properties
    # nelements keeps its canonical v1.2 $id:
    assert (
        properties["nelements"].definition_id
        == "https://schemas.optimade.org/defs/v1.2/properties/optimade/structures/nelements"
    )


def test_property_keys_cover_id_and_type() -> None:
    property_keys = _provider().property_keys("structures")
    assert "id" in property_keys and "type" in property_keys
    # id is normalized under the '__id' record key:
    assert property_keys["id"] == "__id"


def test_records_keyed_by_property_keys() -> None:
    provider = _provider()
    property_keys = provider.property_keys("structures")
    (record,) = list(provider.records("structures"))
    for key in property_keys.values():
        assert key in record
    assert record["__id"] == "s-1"
    assert record["type"] == "structures"
    assert record["elements"] == ["Cl", "Na"]
    assert record["nelements"] == 2
    assert record["nsites"] == 2
    assert record["species_at_sites"] == ["Na", "Cl"]


def test_cartesian_positions_nonorthogonal_hand_computed() -> None:
    # cartesian = sum_k reduced[k] * cell.basis[k] (row-vector convention).
    # For reduced [0.5, 0.5, 0.5] against
    #   a=[2,0,0], b=[1,2,0], c=[0,0,3]:
    #   x = 0.5*2 + 0.5*1 + 0.5*0 = 1.5
    #   y = 0.5*0 + 0.5*2 + 0.5*0 = 1.0
    #   z = 0.5*0 + 0.5*0 + 0.5*3 = 1.5
    (record,) = list(_provider().records("structures"))
    assert record["lattice_vectors"] == [[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    assert record["cartesian_site_positions"] == [[0.0, 0.0, 0.0], [1.5, 1.0, 1.5]]


def test_species_are_optimade_dicts() -> None:
    (record,) = list(_provider().records("structures"))
    species = record["species"]
    assert {s["name"] for s in species} == {"Na", "Cl"}
    na = next(s for s in species if s["name"] == "Na")
    assert na["chemical_symbols"] == ["Na"]
    assert na["concentration"] == [1.0]


def test_structure_features_disorder() -> None:
    cell = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    mixed = Species(name="M", chemical_symbols=("Fe", "Ni"), concentration=(0.5, 0.5))
    structure = Structure(cell, [[0.0, 0.0, 0.0]], [mixed], ["M"])
    (record,) = list(StructureEntryProvider({"m": structure}).records("structures"))
    assert record["structure_features"] == ["disorder"]
    # elements collects the constituent chemical symbols:
    assert record["elements"] == ["Fe", "Ni"]
    assert record["nelements"] == 2


def test_structure_features_site_attachments() -> None:
    cell = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    ch3 = Species(name="CH3", chemical_symbols=("C",), concentration=(1.0,), attached=("H",), nattached=(3,))
    structure = Structure(cell, [[0.0, 0.0, 0.0]], [ch3], ["CH3"])
    (record,) = list(StructureEntryProvider({"c": structure}).records("structures"))
    assert record["structure_features"] == ["site_attachments"]


def test_structure_features_empty_for_ordered() -> None:
    (record,) = list(_provider().records("structures"))
    assert record["structure_features"] == []


def _smfeo3() -> Structure:
    # 4 Fe, 12 O, 4 Sm ordered sites (a fully ordered composition).
    cell = [[5.6, 0.0, 0.0], [0.0, 7.6, 0.0], [0.0, 0.0, 5.3]]
    sites = [[0.01 * i, 0.0, 0.0] for i in range(20)]
    fe = Species(name="Fe", chemical_symbols=("Fe",), concentration=(1.0,))
    o = Species(name="O", chemical_symbols=("O",), concentration=(1.0,))
    sm = Species(name="Sm", chemical_symbols=("Sm",), concentration=(1.0,))
    species_at_sites = ["Fe"] * 4 + ["O"] * 12 + ["Sm"] * 4
    return Structure(cell, sites, [fe, o, sm], species_at_sites)


def test_chemical_formula_and_ratios() -> None:
    (record,) = list(StructureEntryProvider({"x": _smfeo3()}).records("structures"))
    # gcd(4, 12, 4) = 4 -> Fe1 O3 Sm1, alphabetical.
    assert record["chemical_formula_reduced"] == "FeO3Sm"
    # reduced amounts [1, 3, 1] ordered descending -> A3, B, C.
    assert record["chemical_formula_anonymous"] == "A3BC"
    assert record["chemical_formula_descriptive"] == "FeO3Sm"
    assert record["elements_ratios"] == [0.2, 0.6, 0.2]
    assert record["nperiodic_dimensions"] == 3
    assert record["dimension_types"] == [1, 1, 1]


def test_null_structure_serves_null() -> None:
    provider = StructureEntryProvider({"empty": None})
    property_keys = provider.property_keys("structures")
    (record,) = list(provider.records("structures"))
    for key in property_keys.values():
        assert key in record
    assert record["__id"] == "empty"
    assert record["type"] == "structures"
    assert record["lattice_vectors"] is None
    assert record["nelements"] is None
    assert record["chemical_formula_reduced"] is None
    assert record["elements_ratios"] is None


def test_disordered_structure_has_null_composition() -> None:
    cell = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    mixed = Species(name="M", chemical_symbols=("Fe", "Ni"), concentration=(0.5, 0.5))
    structure = Structure(cell, [[0.0, 0.0, 0.0]], [mixed], ["M"])
    (record,) = list(StructureEntryProvider({"m": structure}).records("structures"))
    assert record["chemical_formula_reduced"] is None
    assert record["chemical_formula_anonymous"] is None
    assert record["elements_ratios"] is None


def test_extra_definitions_and_properties_merged() -> None:
    energy = PropertyDefinition.from_simple("_httk_total_energy", description="E", fulltype="float")
    provider = StructureEntryProvider(
        {"a": _nacl_like(), "b": _nacl_like()},
        extra_definitions={"_httk_total_energy": energy},
        properties={"a": {"_httk_total_energy": -1.5}},
    )
    definition = provider.entry_types()["structures"]
    assert "_httk_total_energy" in definition.properties
    assert "_httk_total_energy" in provider.property_keys("structures")
    records = {record["__id"]: record for record in provider.records("structures")}
    assert records["a"]["_httk_total_energy"] == -1.5
    assert records["b"]["_httk_total_energy"] is None  # absent for this entry -> null


def test_unknown_property_name_rejected() -> None:
    with pytest.raises(ValueError) as excinfo:
        StructureEntryProvider({"a": _nacl_like()}, properties={"a": {"_httk_missing": 1.0}})
    assert "_httk_missing" in str(excinfo.value)


def test_registration_discovered_via_httk_core() -> None:
    # Importing httk.core discovers httk.handlers.* packages, registering the
    # provider factory (httk.handlers.atomistic).
    import httk.core
    from httk.core._plugins import resolve_callable
    from httk.core.register import entry_providers

    assert "atomistic-structures" in httk.core.known_entry_providers()
    factory = resolve_callable(entry_providers.require("atomistic-structures").handler)
    provider = factory({"s-1": _nacl_like()})
    assert isinstance(provider, StructureEntryProvider)
    assert set(provider.entry_types()) == {"structures"}
