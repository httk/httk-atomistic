"""Tests for the optional ASE interoperability layer."""

import fractions
from typing import Any

import pytest

from httk.atomistic import (
    Assembly,
    CartesianSiteMoments,
    CartesianSiteMomentsView,
    CollinearSiteMoments,
    CrystalAxisSiteMoments,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
)
from httk.atomistic.composition import ChemicalComposition
from httk.atomistic.integrations.ase import ASEAtoms, ASEAtomsProtocol
from httk.atomistic.models.structure.backend import StructureBackend


class FakeAtoms:
    def get_cell(self) -> Any:
        return [[4, 0, 0], [0, 4, 0], [0, 0, 4]]

    def get_scaled_positions(self) -> Any:
        return [[0, 0, 0], [1 / 2, 1 / 2, 1 / 2], [1 / 4, 1 / 4, 1 / 4]]

    def get_atomic_numbers(self) -> Any:
        return [11, 17, 11]

    def get_pbc(self) -> Any:
        return [True, True, False]


class ChargedFakeAtoms(FakeAtoms):
    charge = 1


class PlainSpeciesFakeAtoms(FakeAtoms):
    species = ("Fe", "Fe")

    def get_scaled_positions(self) -> Any:
        return [[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]]

    def get_atomic_numbers(self) -> Any:
        return [26, 26]


def test_fake_atoms_protocol_and_exact_structure_conversion() -> None:
    fake = FakeAtoms()

    assert isinstance(fake, ASEAtomsProtocol)
    backend = StructureBackend.create(fake)
    assert isinstance(backend, ASEAtoms)
    assert backend.unwrap() is fake

    structure = UnitcellStructureView(fake)
    assert structure.cell.basis.to_floats() == [
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0],
    ]
    assert structure.sites.reduced_coords.to_floats() == [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
    ]
    assert tuple(species.name for species in structure.species) == ("Na", "Cl")
    assert structure.species_at_sites == ("Na", "Cl", "Na")
    assert structure.periodicity == (True, True, False)
    assert structure.site_moments is None
    assert tuple(species.charges for species in structure.species) == (None, None)


def test_ase_view_accepts_plain_string_source_species() -> None:
    pytest.importorskip("ase")
    from httk.atomistic import ASEAtomsView

    view = ASEAtomsView(PlainSpeciesFakeAtoms())

    assert view.get_atomic_numbers().tolist() == [26, 26]


def test_kind_hint_selects_or_rejects_ase_backend() -> None:
    fake = FakeAtoms()

    assert isinstance(StructureBackend.create(fake, kind="ase"), ASEAtoms)
    with pytest.raises(TypeError):
        StructureBackend.create(fake, kind="unitcell")


def test_ase_atoms_round_trip() -> None:
    pytest.importorskip("ase")
    from ase.build import fcc111

    from httk.atomistic import ASEAtomsView

    slab = fcc111("Al", size=(2, 2, 3), vacuum=10.0)
    structure = UnitcellStructureView(slab)
    assert structure.periodicity == (True, True, False)
    assert len(structure.sites) == len(slab)
    assert tuple(species.name for species in structure.species) == ("Al",)

    atoms = ASEAtomsView(structure)
    assert atoms.get_pbc().tolist() == [True, True, False]
    assert tuple(int(number) for number in atoms.get_atomic_numbers()) == tuple(
        int(number) for number in slab.get_atomic_numbers()
    )
    for actual, expected in zip(atoms.get_cell(), slab.get_cell()):
        assert tuple(float(value) for value in actual) == pytest.approx(tuple(float(value) for value in expected))
    for actual, expected in zip(atoms.get_scaled_positions(), slab.get_scaled_positions()):
        assert tuple(float(value) for value in actual) == pytest.approx(tuple(float(value) for value in expected))
    assert ASEAtomsView(atoms) is atoms


def test_ase_atoms_rejects_assemblies_and_implicit_atoms() -> None:
    pytest.importorskip("ase")
    from httk.atomistic import ASEAtomsView

    species = Species("C", ("C",), (1,))
    geometry = ([[4, 0, 0], [0, 4, 0], [0, 0, 4]], [[0, 0, 0]], [species], ["C"])
    assembled = UnitcellStructure(*geometry, assemblies=(Assembly(((0,),), (1,)),))
    implicit = UnitcellStructure(*geometry, chemical_composition=ChemicalComposition({"H": 2}, mode="implicit"))
    with pytest.raises(TypeError, match="assemblies"):
        ASEAtomsView(assembled)
    with pytest.raises(TypeError, match="declared chemical composition"):
        ASEAtomsView(implicit)
    full = UnitcellStructure(*geometry, chemical_composition=ChemicalComposition({"O": 1}, mode="full"))
    with pytest.raises(TypeError, match="declared chemical composition"):
        ASEAtomsView(full)


def test_ase_atoms_rejects_disorder_partial_occupancy_and_attachments() -> None:
    pytest.importorskip("ase")
    from httk.atomistic import ASEAtomsView

    species_values = (
        Species("mixed", ("C", "N"), (fractions.Fraction(1, 2), fractions.Fraction(1, 2))),
        Species("partial", ("C",), (fractions.Fraction(1, 2),)),
        Species("attached", ("C",), (1,), attached=("H",), nattached=(1,)),
    )
    for species in species_values:
        structure = UnitcellStructure(
            [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            [[0, 0, 0]],
            [species],
            [species.name],
        )
        with pytest.raises(TypeError, match="single, unattached"):
            ASEAtomsView(structure)


def test_ase_atoms_imports_moments_and_charges() -> None:
    ase = pytest.importorskip("ase")
    from httk.atomistic import CartesianSiteMoments, CollinearSiteMoments

    scalar = ase.Atoms("Fe2", cell=[[3, 0, 0], [0, 3, 0], [0, 0, 3]], pbc=True)
    scalar.set_initial_magnetic_moments([1.5, -2])
    scalar.set_initial_charges([2.5, 2.5])
    backend = ASEAtoms(scalar)
    assert isinstance(backend.site_moments, CollinearSiteMoments)
    assert backend.site_moments.collinear_moments.to_fractions() == [fractions.Fraction(3, 2), -2]
    assert tuple(species.charges for species in backend.species) == ((fractions.Fraction(5, 2),),)
    assert backend.species_at_sites[0] == backend.species_at_sites[1]

    vector = ase.Atoms("Fe2", cell=[[3, 0, 0], [0, 3, 0], [0, 0, 3]], pbc=True)
    vector.set_initial_magnetic_moments([[1, 2, 3], [0, 0, 0]])
    vector_backend = ASEAtoms(vector)
    assert isinstance(vector_backend.site_moments, CartesianSiteMoments)
    assert vector_backend.site_moments.cartesian_moments.to_floats() == [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]

    zeros = ase.Atoms("Fe2", cell=[[3, 0, 0], [0, 3, 0], [0, 0, 3]], pbc=True)
    zeros.set_initial_charges([0, 0])
    zero_backend = ASEAtoms(zeros)
    assert zero_backend.site_moments is None
    assert tuple(species.charges for species in zero_backend.species) == (None,)


def test_ase_atoms_imports_distinct_charges_for_same_symbol() -> None:
    ase = pytest.importorskip("ase")
    atoms = ase.Atoms("Fe2", cell=[[3, 0, 0], [0, 3, 0], [0, 0, 3]], pbc=True)
    atoms.set_initial_charges([2, 3])

    backend = ASEAtoms(atoms)

    assert len(backend.species) == 2
    assert tuple(species.name for species in backend.species) == ("Fe2+", "Fe3+")
    assert backend.species_at_sites[0] != backend.species_at_sites[1]
    assert {species.charges for species in backend.species} == {
        (fractions.Fraction(2),),
        (fractions.Fraction(3),),
    }


def test_ase_atoms_mixed_charge_zero_is_decorated_and_a_round_trip_fixpoint() -> None:
    ase = pytest.importorskip("ase")
    from httk.atomistic import ASEAtomsView

    atoms = ase.Atoms("Fe2", cell=[[3, 0, 0], [0, 3, 0], [0, 0, 3]], pbc=True)
    atoms.set_initial_charges([2.0, 0.0])

    first = ASEAtoms(atoms)
    second = ASEAtoms(ASEAtomsView(first))

    assert tuple(species.charges for species in first.species) == (
        (fractions.Fraction(2),),
        (fractions.Fraction(0),),
    )
    assert second.species_at_sites == first.species_at_sites
    assert tuple(species.charges for species in second.species) == tuple(species.charges for species in first.species)
    assert ASEAtomsView(first).get_initial_charges().tolist() == [2.0, 0.0]


def test_ase_atoms_round_trip_moments_and_charges() -> None:
    pytest.importorskip("ase")
    from httk.atomistic import ASEAtomsView, same_crystal

    species = (
        Species("Fe2+", ("Fe",), (1,), charges=(2,)),
        Species("Fe3+", ("Fe",), (1,), charges=(3,)),
    )
    source = UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        [[0, 0, 0], [fractions.Fraction(1, 2), fractions.Fraction(1, 2), fractions.Fraction(1, 2)]],
        species,
        ["Fe2+", "Fe3+"],
        site_moments=CollinearSiteMoments([1, -2]),
    )

    exported = ASEAtomsView(source)
    backend = ASEAtoms(exported)

    assert exported.get_initial_magnetic_moments().tolist() == [1.0, -2.0]
    assert exported.get_initial_charges().tolist() == [2.0, 3.0]
    assert same_crystal(source, UnitcellStructureView(backend))


def test_ase_atoms_export_handles_cartesian_and_crystalaxis_moments() -> None:
    pytest.importorskip("ase")
    from httk.atomistic import ASEAtomsView

    cell = [[3, 0, 0], [1, 3, 0], [0, 0, 3]]
    cartesian = CartesianSiteMoments([[1, 2, 3]])
    structure = UnitcellStructure(cell, [[0, 0, 0]], [Species.create("Fe")], ["Fe"], site_moments=cartesian)
    assert ASEAtomsView(structure).get_initial_magnetic_moments().tolist() == [[1.0, 2.0, 3.0]]

    crystalaxis = CrystalAxisSiteMoments([[1, 2, 3]], structure.cell)
    crystal_structure = UnitcellStructure(
        structure.cell, structure.sites, structure.species, structure.species_at_sites, site_moments=crystalaxis
    )
    expected = CartesianSiteMomentsView(crystalaxis).cartesian_moments.to_floats()
    assert ASEAtomsView(crystal_structure).get_initial_magnetic_moments().tolist() == expected


def test_ase_atoms_round_trip_full_cartesian_moments() -> None:
    pytest.importorskip("ase")
    from httk.atomistic import ASEAtomsView

    moments = CartesianSiteMoments([[1, 2, 3], [-4, 5, 6]])
    structure = UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        [[0, 0, 0], [fractions.Fraction(1, 2), fractions.Fraction(1, 2), fractions.Fraction(1, 2)]],
        [Species.create("Fe")],
        ["Fe", "Fe"],
        site_moments=moments,
    )

    round_tripped = UnitcellStructureView(ASEAtoms(ASEAtomsView(structure)))

    assert round_tripped.site_moments == moments


@pytest.mark.parametrize(
    ("field", "species", "kwargs"),
    [
        ("spins", Species("Fe", ("Fe",), (1,), spins=(1,)), {}),
        ("labels", Species("Fe", ("Fe",), (1,), labels=("site",)), {}),
        ("mass", Species("Fe", ("Fe",), (1,), mass=(56,)), {}),
    ],
)
def test_ase_atoms_export_rejects_unencoded_species_state(field: str, species: Species, kwargs: Any) -> None:
    pytest.importorskip("ase")
    from httk.atomistic import ASEAtomsView

    structure = UnitcellStructure([[3, 0, 0], [0, 3, 0], [0, 0, 3]], [[0, 0, 0]], [species], [species.name], **kwargs)
    with pytest.raises(ValueError, match=field):
        ASEAtomsView(structure)


def test_ase_atoms_export_maps_none_charge_to_ase_zero() -> None:
    pytest.importorskip("ase")
    from httk.atomistic import ASEAtomsView

    charged = Species("Fe+1", ("Fe",), (1,), charges=(1,))
    plain = Species.create("Fe")
    structure = UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]], [[0, 0, 0], [0.5, 0.5, 0.5]], [charged, plain], [charged.name, plain.name]
    )
    assert ASEAtomsView(structure).get_initial_charges().tolist() == [1.0, 0.0]


def test_ase_atoms_export_rejects_structure_charge() -> None:
    pytest.importorskip("ase")
    from httk.atomistic import ASEAtomsView

    structure = UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]], [[0, 0, 0]], [Species.create("Fe")], ["Fe"], charge=1
    )
    with pytest.raises(ValueError, match="charge"):
        ASEAtomsView(structure)


def test_ase_atoms_export_rejects_charge_on_unwrapped_source() -> None:
    pytest.importorskip("ase")
    from httk.atomistic import ASEAtomsView

    with pytest.raises(ValueError, match="charge"):
        ASEAtomsView(ChargedFakeAtoms())
