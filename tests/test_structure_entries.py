"""Unit tests for StructureEntry and its complete OPTIMADE projection."""

import datetime
from fractions import Fraction

import pytest
from httk.core import FracVector, PropertyDefinition

from httk.atomistic import (
    Assembly,
    ASUStructure,
    ASUStructureRecord,
    FundamentalDomainStructureRecord,
    StructureEntryProvider,
    UnitcellStructure,
    UnitcellStructureRecord,
    WyckoffSite,
)
from httk.atomistic.models.species.species import Species
from httk.atomistic.entries.structures import StructureEntry


def _nacl_like() -> UnitcellStructure:
    # A non-orthogonal cell (rows are the lattice vectors).
    cell = [[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    sites = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    na = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    cl = Species(name="Cl", chemical_symbols=("Cl",), concentration=(1.0,))
    return UnitcellStructure(cell, sites, [na, cl], ["Na", "Cl"])


def _provider() -> StructureEntryProvider:
    return StructureEntryProvider({"s-1": _nacl_like()})


def test_entry_types_describe_structures() -> None:
    from httk.core import EntryTypeDefinition

    entry_types = _provider().entry_types()
    assert set(entry_types) == {"structures"}
    definition = entry_types["structures"]
    assert isinstance(definition, EntryTypeDefinition)
    properties = definition.properties
    # The provider serves all 30 standard v1.3 properties, plus the six symmetry,
    # two precision, one charge, and one moment property under the _httk_ prefix.
    # 30 standard OPTIMADE properties, plus the six symmetry properties, two precision
    # properties httk serves under the _httk_ prefix, each described by a vendored
    # schemas.httk.org definition.
    assert len(properties) == 40
    for name in ("id", "type", "elements", "nelements", "nsites", "species", "structure_features"):
        assert name in properties
    # These formerly omitted v1.3-native properties are now served:
    assert "wyckoff_positions" in properties
    assert "fractional_site_positions" in properties
    # nelements keeps its canonical v1.2 $id:
    assert (
        properties["nelements"].definition_id
        == "https://schemas.optimade.org/defs/v1.2/properties/optimade/structures/nelements"
    )


def test_property_keys_cover_id_and_type() -> None:
    property_keys = _provider().property_keys("structures")
    assert len({name for name in property_keys if not name.startswith("_httk_")}) == 30
    assert set(_provider().entry_types()["structures"].properties) <= set(property_keys)
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
    assert record["immutable_id"] is None
    assert record["last_modified"] is None


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
    structure = UnitcellStructure(cell, [[0.0, 0.0, 0.0]], [mixed], ["M"])
    (record,) = list(StructureEntryProvider({"m": structure}).records("structures"))
    assert record["structure_features"] == ["disorder"]
    # elements collects the constituent chemical symbols:
    assert record["elements"] == ["Fe", "Ni"]
    assert record["nelements"] == 2


def test_structure_features_site_attachments() -> None:
    cell = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    ch3 = Species(name="CH3", chemical_symbols=("C",), concentration=(1.0,), attached=("H",), nattached=(3,))
    structure = UnitcellStructure(cell, [[0.0, 0.0, 0.0]], [ch3], ["CH3"])
    (record,) = list(StructureEntryProvider({"c": structure}).records("structures"))
    assert record["structure_features"] == ["site_attachments"]


def test_structure_features_empty_for_ordered() -> None:
    (record,) = list(_provider().records("structures"))
    assert record["structure_features"] == []


def test_unused_species_do_not_mark_structure_features() -> None:
    """Only species referenced by represented sites contribute OPTIMADE features."""
    cell = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    na = Species(name="Na", chemical_symbols=("Na",), concentration=(1,))
    unused_disordered = Species(name="X", chemical_symbols=("Fe", "Ni"), concentration=(0.5, 0.5))
    unused_attached = Species(name="CH3", chemical_symbols=("C",), concentration=(1,), attached=("H",), nattached=(3,))
    structure = UnitcellStructure(cell, [[0, 0, 0]], [na, unused_disordered, unused_attached], ["Na"])
    (record,) = list(StructureEntryProvider({"only-na": structure}).records("structures"))
    assert record["structure_features"] == []


def _smfeo3() -> UnitcellStructure:
    # 4 Fe, 12 O, 4 Sm ordered sites (a fully ordered composition).
    cell = [[5.6, 0.0, 0.0], [0.0, 7.6, 0.0], [0.0, 0.0, 5.3]]
    sites = [[0.01 * i, 0.0, 0.0] for i in range(20)]
    fe = Species(name="Fe", chemical_symbols=("Fe",), concentration=(1.0,))
    o = Species(name="O", chemical_symbols=("O",), concentration=(1.0,))
    sm = Species(name="Sm", chemical_symbols=("Sm",), concentration=(1.0,))
    species_at_sites = ["Fe"] * 4 + ["O"] * 12 + ["Sm"] * 4
    return UnitcellStructure(cell, sites, [fe, o, sm], species_at_sites)


def test_chemical_formula_and_ratios() -> None:
    (record,) = list(StructureEntryProvider({"x": _smfeo3()}).records("structures"))
    # gcd(4, 12, 4) = 4 -> Fe1 O3 Sm1, alphabetical.
    assert record["chemical_formula_reduced"] == "FeO3Sm"
    # reduced amounts [1, 3, 1] ordered descending -> A3, B, C.
    assert record["chemical_formula_anonymous"] == "A3BC"
    # Descriptive and Hill formulas are source annotations, not invented aliases.
    assert record["chemical_formula_descriptive"] is None
    assert record["chemical_formula_hill"] is None
    assert record["elements_ratios"] == [0.2, 0.6, 0.2]


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


def test_disordered_structure_uses_exact_expected_composition() -> None:
    cell = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    mixed = Species(name="M", chemical_symbols=("Fe", "Ni"), concentration=(0.5, 0.5))
    structure = UnitcellStructure(cell, [[0.0, 0.0, 0.0]], [mixed], ["M"])
    (record,) = list(StructureEntryProvider({"m": structure}).records("structures"))
    assert record["chemical_formula_reduced"] == "FeNi"
    assert record["chemical_formula_anonymous"] == "AB"
    assert record["elements_ratios"] == [0.5, 0.5]


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


@pytest.mark.parametrize("source", ["properties", "extra_definitions"])
def test_standard_properties_cannot_be_overridden(source: str) -> None:
    replacement = PropertyDefinition.from_simple("elements", description="not standard", fulltype="string")
    with pytest.raises(ValueError, match="may not override standard"):
        if source == "properties":
            StructureEntryProvider({"a": _nacl_like()}, properties={"a": {"elements": ["H"]}})
        else:
            StructureEntryProvider({"a": _nacl_like()}, extra_definitions={"elements": replacement})


def test_structure_entry_metadata_is_served_but_not_structural_equality() -> None:
    stamp = datetime.datetime(2026, 8, 1, 12, 30, tzinfo=datetime.UTC)
    structure = _nacl_like()
    left = UnitcellStructure(
        structure.cell,
        structure.sites,
        structure.species,
        structure.species_at_sites,
        immutable_id="stable-left",
        last_modified=stamp,
    )
    right = UnitcellStructure(
        structure.cell,
        structure.sites,
        structure.species,
        structure.species_at_sites,
        immutable_id="stable-right",
    )
    assert left == right
    assert left.id == right.id

    (record,) = list(StructureEntryProvider({"left": left}).records("structures"))
    assert record["__id"] == "left"
    assert record["type"] == "structures"
    assert record["immutable_id"] == "stable-left"
    assert record["last_modified"] == "2026-08-01T12:30:00+00:00"


def test_structure_entry_validation_and_mapping_identity() -> None:
    with pytest.raises(ValueError, match="timezone"):
        naive = datetime.datetime(2026, 8, 1, tzinfo=datetime.UTC).replace(tzinfo=None)
        structure = _nacl_like()
        UnitcellStructure(
            structure.cell, structure.sites, structure.species, structure.species_at_sites, last_modified=naive
        )
    with pytest.raises(TypeError, match="logical entry family"):
        StructureEntry()


def test_complete_standard_projection_and_assembly_null_semantics() -> None:
    cell = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
    na = Species(name="Na", chemical_symbols=("Na",), concentration=(1,))
    cl = Species(name="Cl", chemical_symbols=("Cl",), concentration=(1,))
    structure = UnitcellStructure(
        cell,
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        [na, cl],
        ["Na", "Cl"],
        assemblies=(Assembly(((0,), (1,)), (Fraction(1, 3), Fraction(2, 3))),),
        chemical_formula_descriptive="NaCl",
        chemical_formula_hill="Cl2Na",
        optimization_type="experimental",
    )
    provider = StructureEntryProvider({"x": structure})
    (record,) = list(provider.records("structures"))
    for name, key in provider.property_keys("structures").items():
        assert key in record, name
    assert record["assemblies"] == [{"sites_in_groups": [[0], [1]], "group_probabilities": [1 / 3, 2 / 3]}]
    assert record["chemical_formula_descriptive"] == "NaCl"
    assert record["chemical_formula_hill"] == "Cl2Na"
    assert record["optimization_type"] == "experimental"
    assert record["site_coordinate_span"] == "unit_cell"
    assert record["fractional_site_positions"] == [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]

    (plain,) = list(StructureEntryProvider({"x": _nacl_like()}).records("structures"))
    assert plain["assemblies"] is None
    empty_assemblies = UnitcellStructure(cell, [[0, 0, 0]], [na], ["Na"], assemblies=())
    (present_empty,) = list(StructureEntryProvider({"x": empty_assemblies}).records("structures"))
    assert present_empty["assemblies"] == []


def test_asymmetric_unit_is_projected_without_unit_cell_expansion() -> None:
    no_parameters = FracVector(())
    asu = ASUStructure(
        [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
        225,
        [WyckoffSite("a", no_parameters, "Na"), WyckoffSite("b", no_parameters, "Cl")],
        [
            Species(name="Na", chemical_symbols=("Na",), concentration=(1,)),
            Species(name="Cl", chemical_symbols=("Cl",), concentration=(1,)),
        ],
    )
    (record,) = list(StructureEntryProvider({"rocksalt": asu}).records("structures"))
    assert record["site_coordinate_span"] == "asymmetric_unit"
    assert record["nsites"] == 2
    assert len(record["fractional_site_positions"]) == 2
    assert record["species_at_sites"] == ["Na", "Cl"]
    assert record["wyckoff_positions"] == ["a", "b"]
    # Composition remains a unit-cell projection (four atoms in each orbit).
    assert record["chemical_formula_reduced"] == "ClNa"


def test_registration_discovered_via_httk_core() -> None:
    # Importing httk.core discovers the adapter and entry registration tiers.
    from httk.core._plugins import resolve_callable
    from httk.core.register import (
        entry_providers,
        known_entry_providers,
        known_format_adapters,
        resolve_entry_family,
        resolve_entry_record,
    )

    assert "atomistic-structures" in known_entry_providers()
    assert known_format_adapters()["cif"] == "atomistic-structures"
    assert resolve_entry_family("structures") is StructureEntry
    assert resolve_entry_record("atomistic-unitcell-structure") is UnitcellStructureRecord
    assert resolve_entry_record("atomistic-fundamental-domain-structure") is FundamentalDomainStructureRecord
    assert resolve_entry_record("atomistic-asu-structure") is ASUStructureRecord
    factory = resolve_callable(entry_providers.require("atomistic-structures").handler)
    provider = factory({"s-1": _nacl_like()})
    assert isinstance(provider, StructureEntryProvider)
    assert set(provider.entry_types()) == {"structures"}


# --- periodicity ---


def test_periodicity_is_served_whatever_the_composition() -> None:
    """It is not a composition property, and must not null out with the formula.

    The bug was that these two travelled with the composition block, which serves null for
    any structure that is not fully ordered — so a disordered alloy, or a structure with
    attached atoms, reported its periodicity as *unknown* purely because a whole-atom
    formula could not be written for it. The two are independent: a disordered alloy has a
    periodicity all the same.

    These structures are all built without saying otherwise, so all three are crystals.
    Serving a reduced periodicity is covered in `test_periodicity.py`.
    """
    cell = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    ordered = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    disordered = Species(name="X", chemical_symbols=("Na", "K"), concentration=(0.5, 0.5))
    attached = Species(name="C", chemical_symbols=("C",), concentration=(1.0,), attached=("H",), nattached=(1,))

    for label, species in (("ordered", ordered), ("disordered", disordered), ("attached", attached)):
        structure = UnitcellStructure(cell, [[0, 0, 0]], [species], [species.name])
        (record,) = list(StructureEntryProvider({"x": structure}).records("structures"))
        assert record["nperiodic_dimensions"] == 3, label
        assert record["dimension_types"] == [1, 1, 1], label

    # Exact partial occupations have an exact expected composition as well.
    (record,) = list(
        StructureEntryProvider({"x": UnitcellStructure(cell, [[0, 0, 0]], [disordered], ["X"])}).records("structures")
    )
    assert record["chemical_formula_reduced"] == "KNa"
    assert record["elements_ratios"] == [0.5, 0.5]


def test_an_entry_with_no_structure_still_serves_null_periodicity() -> None:
    """There is no structure to describe, so unknown is the honest answer there."""
    (record,) = list(StructureEntryProvider({"empty": None}).records("structures"))
    assert record["nperiodic_dimensions"] is None
    assert record["dimension_types"] is None
