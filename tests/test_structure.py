import pytest

from httk.atomistic import (
    Species,
    Structure,
    StructureBackend,
    StructurePrimitive,
    StructurePrimitiveView,
    StructureSimple,
    StructureSimpleView,
    atomic_number,
    symbol_of,
)
from httk.core import unwrap

CUBIC = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]


def nacl_structure() -> Structure:
    return Structure(
        basis=CUBIC,
        sites=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        species=[
            {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
            {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1.0]},
        ],
        species_at_sites=["Na", "Cl"],
    )


def nacl_triple() -> tuple[list[list[float]], list[list[float]], list[int]]:
    return (CUBIC, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], [11, 17])


# --- elements ---


def test_elements_roundtrip() -> None:
    assert atomic_number("H") == 1
    assert atomic_number("Og") == 118
    assert symbol_of(1) == "H"
    assert symbol_of(118) == "Og"
    for z in (1, 22, 79, 118):
        assert atomic_number(symbol_of(z)) == z


def test_elements_unknowns_raise() -> None:
    with pytest.raises(ValueError):
        atomic_number("X")
    with pytest.raises(ValueError):
        atomic_number("vacancy")
    with pytest.raises(ValueError):
        atomic_number("Zz")
    with pytest.raises(ValueError):
        symbol_of(0)
    with pytest.raises(ValueError):
        symbol_of(119)


# --- Species ---


def test_species_valid_and_create_from_dict() -> None:
    species = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    assert species.is_single_element
    assert Species.create(species) is species

    from_dict = Species.create({"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]})
    assert from_dict == species
    assert from_dict.chemical_symbols == ("Na",)
    assert from_dict.concentration == (1.0,)


def test_species_validation_errors() -> None:
    with pytest.raises(ValueError):
        Species(name="Na", chemical_symbols=("Na",), concentration=(1.0, 0.0))
    with pytest.raises(ValueError):
        Species(name="bad", chemical_symbols=("Zz",), concentration=(1.0,))
    with pytest.raises(ValueError):
        Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,), attached=("H",))
    with pytest.raises(ValueError):
        Species(
            name="Na",
            chemical_symbols=("Na",),
            concentration=(1.0,),
            attached=("H", "H"),
            nattached=(1,),
        )
    with pytest.raises(ValueError):
        Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,), mass=(1.0, 2.0))


def test_species_is_single_element_cases() -> None:
    pure = Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))
    assert pure.is_single_element

    alloy = Species(name="Ti", chemical_symbols=("Ti", "vacancy"), concentration=(0.9, 0.1))
    assert not alloy.is_single_element

    vacancy = Species(name="vac", chemical_symbols=("vacancy",), concentration=(1.0,))
    assert not vacancy.is_single_element

    attached = Species(
        name="CH3",
        chemical_symbols=("C",),
        concentration=(1.0,),
        attached=("H",),
        nattached=(3,),
    )
    assert not attached.is_single_element


# --- Structure ---


def test_structure_normalizes_and_exposes_quartet() -> None:
    structure = nacl_structure()
    assert structure.basis == ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))
    assert structure.sites == ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
    assert all(isinstance(s, Species) for s in structure.species)
    assert structure.species_at_sites == ("Na", "Cl")


def test_structure_shape_and_name_validation() -> None:
    with pytest.raises(ValueError):
        Structure(
            basis=[[1.0, 0.0], [0.0, 1.0]],
            sites=[[0.0, 0.0, 0.0]],
            species=[{"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]}],
            species_at_sites=["Na"],
        )
    with pytest.raises(ValueError):
        Structure(
            basis=CUBIC,
            sites=[[0.0, 0.0, 0.0]],
            species=[{"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]}],
            species_at_sites=["Na", "Cl"],
        )
    with pytest.raises(ValueError):
        Structure(
            basis=CUBIC,
            sites=[[0.0, 0.0, 0.0]],
            species=[{"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]}],
            species_at_sites=["Cl"],
        )
    with pytest.raises(ValueError):
        Structure(
            basis=CUBIC,
            sites=[[0.0, 0.0, 0.0]],
            species=[
                {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
                {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
            ],
            species_at_sites=["Na"],
        )


def test_structure_equality() -> None:
    assert nacl_structure() == nacl_structure()
    other = Structure(
        basis=CUBIC,
        sites=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        species=[
            {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
            {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1.0]},
        ],
        species_at_sites=["Na", "Na"],
    )
    assert nacl_structure() != other


# --- Dispatch ---


def test_backend_create_dispatches_and_kind_overrides() -> None:
    simple = StructureBackend.create(nacl_structure())
    assert isinstance(simple, StructureSimple)

    primitive = StructureBackend.create(nacl_triple())
    assert isinstance(primitive, StructurePrimitive)

    assert isinstance(StructureBackend.create(nacl_structure(), kind="simple"), StructureSimple)
    assert isinstance(StructureBackend.create(nacl_triple(), kind="primitive"), StructurePrimitive)


def test_backend_create_raises_for_malformed_triple() -> None:
    with pytest.raises(TypeError):
        StructureBackend.create([CUBIC, [[0.0, 0.0, 0.0]], [11, 17]])  # numbers/positions length mismatch
    with pytest.raises(TypeError):
        StructureBackend.create([[[1.0, 2.0]], [[0.0, 0.0, 0.0]], [1]])  # lattice not 3x3
    with pytest.raises(TypeError):
        StructureBackend.create(12345)
    with pytest.raises(TypeError):
        StructureBackend.create(nacl_structure(), kind="primitive")


# --- Views ---


def test_simple_view_from_primitive_derives_species() -> None:
    view = StructureSimpleView(nacl_triple())
    assert isinstance(view, Structure)
    assert view.basis == ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))
    assert view.species_at_sites == ("Na", "Cl")
    assert {s.name for s in view.species} == {"Na", "Cl"}
    assert all(s.is_single_element for s in view.species)


def test_simple_view_species_is_one_per_distinct_number() -> None:
    triple = (CUBIC, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], [11, 11])
    view = StructureSimpleView(triple)
    assert len(view.species) == 1
    assert view.species[0].name == "Na"
    assert view.species_at_sites == ("Na", "Na")


def test_primitive_view_from_structure_has_correct_numbers() -> None:
    lattice, positions, numbers = StructurePrimitiveView(nacl_structure())
    assert lattice == ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))
    assert positions == ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
    assert numbers == (11, 17)


def test_primitive_view_raises_for_non_single_element_species() -> None:
    alloy = Structure(
        basis=CUBIC,
        sites=[[0.0, 0.0, 0.0]],
        species=[{"name": "Ti", "chemical_symbols": ["Ti", "vacancy"], "concentration": [0.9, 0.1]}],
        species_at_sites=["Ti"],
    )
    with pytest.raises(TypeError):
        StructurePrimitiveView(alloy)


def test_view_rewrap_identity_and_shared_backend() -> None:
    backend = StructureBackend.create(nacl_triple())
    v1 = StructureSimpleView(backend)
    assert StructureSimpleView(v1) is v1

    v2 = StructureSimpleView(backend)
    assert v1._backend is backend
    assert v2._backend is backend

    pv = StructurePrimitiveView(nacl_structure())
    assert StructurePrimitiveView(pv) is pv


def test_unwrap_returns_native_raw_object() -> None:
    structure = nacl_structure()
    simple_view = StructureSimpleView(structure)
    # Simple view built from a Structure -> unwrap gives back a Structure
    assert isinstance(unwrap(simple_view), Structure)

    triple = nacl_triple()
    primitive_backend = StructureBackend.create(triple)
    assert unwrap(primitive_backend) is triple

    primitive_view = StructurePrimitiveView(structure)
    assert isinstance(unwrap(primitive_view), Structure)


# --- OPTIMADE example fidelity ---


def test_optimade_vacancy_and_attached_examples_survive_simple_but_not_primitive() -> None:
    structure = Structure(
        basis=CUBIC,
        sites=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        species=[
            {"name": "Ti", "chemical_symbols": ["Ti", "vacancy"], "concentration": [0.9, 0.1]},
            {
                "name": "CH3",
                "chemical_symbols": ["C"],
                "concentration": [1.0],
                "attached": ["H"],
                "nattached": [3],
            },
        ],
        species_at_sites=["Ti", "CH3"],
    )
    # Attached / vacancy species survive in the Simple representation.
    by_name = {s.name: s for s in structure.species}
    assert by_name["Ti"].chemical_symbols == ("Ti", "vacancy")
    assert by_name["CH3"].attached == ("H",)
    assert by_name["CH3"].nattached == (3,)

    # But they cannot be represented as a primitive structure.
    with pytest.raises(TypeError):
        StructurePrimitiveView(structure)
