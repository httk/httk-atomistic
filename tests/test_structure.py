import fractions

import pytest
from httk.core import unwrap

from httk.atomistic import (
    Cell,
    PlainStructure,
    PlainStructureView,
    Sites,
    Species,
    StructureBackend,
    UnitcellStructure,
    UnitcellStructureView,
    atomic_number,
    symbol_of,
)

CUBIC = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]


def nacl_structure() -> UnitcellStructure:
    return UnitcellStructure(
        cell=CUBIC,
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
    assert Species.from_object(species) is species

    from_dict = Species.from_object({"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]})
    assert from_dict == species
    assert from_dict.chemical_symbols == ("Na",)
    assert from_dict.concentration == (1.0,)


def test_species_validation_errors() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Species(name="Na", chemical_symbols=("Na",), concentration=(-1,))
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


def test_only_unrepresented_species_accept_aggregate_concentrations() -> None:
    aggregate = Species("H1", ("H",), (5,))
    structure = UnitcellStructure(CUBIC, [[0, 0, 0]], (Species("O", ("O",), (1,)), aggregate), ("O",))

    assert structure.implicit_atoms == ("H1",)
    with pytest.raises(ValueError, match="assigned to sites"):
        UnitcellStructure(CUBIC, [[0, 0, 0]], (aggregate,), ("H1",))


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


# --- UnitcellStructure ---


def test_structure_normalizes_and_exposes_quartet() -> None:
    structure = nacl_structure()
    assert isinstance(structure.cell, Cell)
    assert structure.cell.basis.to_floats() == [
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0],
    ]
    assert isinstance(structure.sites, Sites)
    assert structure.sites.reduced_coords.to_floats() == [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ]
    assert len(structure.sites) == 2
    assert all(isinstance(s, Species) for s in structure.species)
    assert structure.species_at_sites == ("Na", "Cl")


def test_structure_accepts_mixed_bare_and_species_inputs() -> None:
    structure = UnitcellStructure(
        cell=CUBIC,
        sites=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ],
        species=[Species("Pb", ("Pb",), (1.0,)), "Ti", 8],
        species_at_sites=["Pb", "Ti", "O", "O"],
    )
    assert tuple(species.name for species in structure.species) == ("Pb", "Ti", "O")
    assert structure.species_at_sites == ("Pb", "Ti", "O", "O")


def test_structure_accepts_bare_symbol_species_inputs() -> None:
    structure = UnitcellStructure(
        cell=CUBIC,
        sites=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.1, 0.1, 0.1],
        ],
        species=["Pb", "Ti", "O"],
        species_at_sites=["Pb", "Ti", "O", "O", "O"],
    )
    assert tuple(species.name for species in structure.species) == ("Pb", "Ti", "O")
    assert structure.species_at_sites == ("Pb", "Ti", "O", "O", "O")


def test_structure_accepts_atomic_number_species_inputs() -> None:
    structure = UnitcellStructure(
        cell=CUBIC,
        sites=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.1, 0.1, 0.1],
        ],
        species=[82, 22, 8],
        species_at_sites=["Pb", "Ti", "O", "O", "O"],
    )
    assert tuple(species.name for species in structure.species) == ("Pb", "Ti", "O")
    assert structure.species_at_sites == ("Pb", "Ti", "O", "O", "O")


def test_structure_infers_species_from_site_values_in_first_occurrence_order() -> None:
    oxygen = Species("O", ("O",), (1,))
    chlorine = {"name": "Cl", "chemical_symbols": ["Cl"], "concentration": [1]}
    structure = UnitcellStructure(
        cell=CUBIC,
        sites=[[0, 0, 0], ["1/2", 0, 0], [0, "1/2", 0], [0, 0, "1/2"], ["1/2", "1/2", "1/2"]],
        species_at_sites=[oxygen, 22, chlorine, 22, oxygen],
    )

    assert tuple(value.name for value in structure.species) == ("O", "Ti", "Cl")
    assert structure.species_at_sites == ("O", "Ti", "Cl", "Ti", "O")


def test_structure_rejects_conflicting_inferred_species_definitions() -> None:
    with pytest.raises(ValueError, match="conflicting definitions"):
        UnitcellStructure(
            cell=CUBIC,
            sites=[[0, 0, 0], ["1/2", "1/2", "1/2"]],
            species_at_sites=[
                Species("mixed", ("Na",), (1,)),
                Species("mixed", ("Cl",), (1,)),
            ],
        )


def test_structure_shape_and_name_validation() -> None:
    # A malformed cell is rejected by the Cell family (a 2x2 cannot be represented).
    with pytest.raises((ValueError, TypeError)):
        UnitcellStructure(
            cell=[[1.0, 0.0], [0.0, 1.0]],
            sites=[[0.0, 0.0, 0.0]],
            species=[{"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]}],
            species_at_sites=["Na"],
        )
    with pytest.raises(ValueError):
        UnitcellStructure(
            cell=CUBIC,
            sites=[[0.0, 0.0, 0.0]],
            species=[{"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]}],
            species_at_sites=["Na", "Cl"],
        )
    with pytest.raises(ValueError):
        UnitcellStructure(
            cell=CUBIC,
            sites=[[0.0, 0.0, 0.0]],
            species=[{"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]}],
            species_at_sites=["Cl"],
        )
    with pytest.raises(ValueError):
        UnitcellStructure(
            cell=CUBIC,
            sites=[[0.0, 0.0, 0.0]],
            species=[
                {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
                {"name": "Na", "chemical_symbols": ["Na"], "concentration": [1.0]},
            ],
            species_at_sites=["Na"],
        )


def test_structure_equality() -> None:
    assert nacl_structure() == nacl_structure()
    other = UnitcellStructure(
        cell=CUBIC,
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
    simple = nacl_structure()
    assert isinstance(simple, StructureBackend)
    assert UnitcellStructureView(simple)._backend is simple

    primitive = StructureBackend._select_backend(nacl_triple())
    assert isinstance(primitive, PlainStructure)

    assert isinstance(StructureBackend._select_backend(nacl_triple(), kind="plain"), PlainStructure)


def test_backend_create_raises_for_malformed_triple() -> None:
    with pytest.raises(TypeError):
        StructureBackend._select_backend([CUBIC, [[0.0, 0.0, 0.0]], [11, 17]])  # numbers/positions length mismatch
    with pytest.raises(TypeError):
        StructureBackend._select_backend([[[1.0, 2.0]], [[0.0, 0.0, 0.0]], [1]])  # lattice not 3x3
    with pytest.raises(TypeError):
        StructureBackend._select_backend(12345)
    with pytest.raises(TypeError):
        StructureBackend._select_backend(nacl_structure(), kind="plain")


# --- Views ---


def test_simple_view_from_primitive_derives_species() -> None:
    view = UnitcellStructureView(nacl_triple())
    assert isinstance(view, UnitcellStructure)
    assert view.cell.basis.to_floats() == [
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0],
    ]
    assert view.species_at_sites == ("Na", "Cl")
    assert {s.name for s in view.species} == {"Na", "Cl"}
    assert all(s.is_single_element for s in view.species)


def test_simple_view_species_is_one_per_distinct_number() -> None:
    triple = (CUBIC, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], [11, 11])
    view = UnitcellStructureView(triple)
    assert len(view.species) == 1
    assert view.species[0].name == "Na"
    assert view.species_at_sites == ("Na", "Na")


def test_primitive_view_from_structure_has_correct_numbers() -> None:
    lattice, positions, numbers = PlainStructureView(nacl_structure())
    assert lattice == ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))
    assert positions == ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
    assert numbers == (11, 17)


def test_primitive_view_raises_for_non_single_element_species() -> None:
    alloy = UnitcellStructure(
        cell=CUBIC,
        sites=[[0.0, 0.0, 0.0]],
        species=[
            {
                "name": "Ti",
                "chemical_symbols": ["Ti", "vacancy"],
                "concentration": [0.9, 0.1],
            }
        ],
        species_at_sites=["Ti"],
    )
    with pytest.raises(TypeError):
        PlainStructureView(alloy)


def test_primitive_view_rejects_every_unrepresentable_semantic_feature() -> None:
    from httk.atomistic import Assembly, Species
    from httk.atomistic.composition import ChemicalComposition

    element = Species("C", ("C",), (1,))
    partial = Species("C", ("C",), (fractions.Fraction(1, 2),))
    attached = Species("CH", ("C",), (1,), attached=("H",), nattached=(1,))
    implicit = UnitcellStructure(
        CUBIC,
        [[0, 0, 0]],
        [element],
        ["C"],
        chemical_composition=ChemicalComposition({"H": 2}, mode="implicit"),
    )
    cases = (
        UnitcellStructure(CUBIC, [[0, 0, 0]], [partial], ["C"]),
        UnitcellStructure(CUBIC, [[0, 0, 0]], [attached], ["CH"]),
        UnitcellStructure(CUBIC, [[0, 0, 0]], [element], ["C"], assemblies=(Assembly(((0,),), (1,)),)),
        implicit,
        UnitcellStructure(
            CUBIC,
            [[0, 0, 0]],
            [element],
            ["C"],
            chemical_composition=ChemicalComposition({"O": 1}, mode="full"),
        ),
    )
    for structure in cases:
        with pytest.raises(TypeError):
            PlainStructureView(structure)


def test_primitive_view_refuses_partially_occupied_single_symbol_species() -> None:
    partial = UnitcellStructure(
        cell=CUBIC,
        sites=[[0.0, 0.0, 0.0]],
        species=[Species("Na_half", ("Na",), (0.5,))],
        species_at_sites=["Na_half"],
    )
    assert not partial.species[0].is_single_element
    with pytest.raises(TypeError, match="single, unattached"):
        PlainStructureView(partial)


def test_view_rewrap_identity_and_shared_backend() -> None:
    backend = StructureBackend._select_backend(nacl_triple())
    v1 = UnitcellStructureView(backend)
    assert UnitcellStructureView(v1) is v1

    v2 = UnitcellStructureView(backend)
    assert v1._backend is backend
    assert v2._backend is backend

    pv = PlainStructureView(nacl_structure())
    assert PlainStructureView(pv) is pv


def test_unwrap_returns_native_raw_object() -> None:
    structure = nacl_structure()
    simple_view = UnitcellStructureView(structure)
    # Simple view built from a UnitcellStructure -> unwrap gives back a UnitcellStructure
    assert isinstance(unwrap(simple_view), UnitcellStructure)

    triple = nacl_triple()
    primitive_backend = StructureBackend._select_backend(triple)
    assert unwrap(primitive_backend) is triple

    primitive_view = PlainStructureView(structure)
    assert isinstance(unwrap(primitive_view), UnitcellStructure)


# --- OPTIMADE example fidelity ---


def test_optimade_vacancy_and_attached_examples_survive_simple_but_not_primitive() -> None:
    structure = UnitcellStructure(
        cell=CUBIC,
        sites=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        species=[
            {
                "name": "Ti",
                "chemical_symbols": ["Ti", "vacancy"],
                "concentration": [0.9, 0.1],
            },
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
    # Attached / vacancy species survive in the Unitcell representation.
    by_name = {s.name: s for s in structure.species}
    assert by_name["Ti"].chemical_symbols == ("Ti", "vacancy")
    assert by_name["CH3"].attached == ("H",)
    assert by_name["CH3"].nattached == (3,)

    # But they cannot be represented as a primitive structure.
    with pytest.raises(TypeError):
        PlainStructureView(structure)
