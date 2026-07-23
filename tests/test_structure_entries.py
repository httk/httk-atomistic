"""Unit tests for StructureEntryProvider and its registration."""

from httk.atomistic import Structure, StructureEntryProvider
from httk.atomistic.species import Species


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
    entry_types = _provider().entry_types()
    assert set(entry_types) == {"structures"}
    properties = entry_types["structures"]["properties"]
    for name in ("id", "type", "elements", "nelements", "nsites", "species", "structure_features"):
        assert name in properties
        assert "fulltype" in properties[name]


def test_columns_cover_id_and_type() -> None:
    columns = _provider().columns("structures")
    assert "id" in columns and "type" in columns
    # id is normalized under the '__id' record column:
    assert columns["id"] == "__id"


def test_records_keyed_by_columns() -> None:
    provider = _provider()
    columns = provider.columns("structures")
    (record,) = list(provider.records("structures"))
    for column in columns.values():
        assert column in record
    assert record["__id"] == "s-1"
    assert record["type"] == "structures"
    assert record["elements"] == ["Cl", "Na"]
    assert record["nelements"] == 2
    assert record["nsites"] == 2
    assert record["species_at_sites"] == ["Na", "Cl"]


def test_cartesian_positions_nonorthogonal_hand_computed() -> None:
    # cartesian = sum_k reduced[k] * cell.matrix[k] (row-vector convention).
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
