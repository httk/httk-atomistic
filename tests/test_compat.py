"""Tests for the optional ASE interoperability layer."""

from typing import Any

import pytest

from httk.atomistic import (
    ASEAtomsBackend,
    ASEAtomsProtocol,
    StructureBackend,
    UnitcellStructureView,
)


class FakeAtoms:
    def get_cell(self) -> Any:
        return [[4, 0, 0], [0, 4, 0], [0, 0, 4]]

    def get_scaled_positions(self) -> Any:
        return [[0, 0, 0], [1 / 2, 1 / 2, 1 / 2], [1 / 4, 1 / 4, 1 / 4]]

    def get_atomic_numbers(self) -> Any:
        return [11, 17, 11]

    def get_pbc(self) -> Any:
        return [True, True, False]


def test_fake_atoms_protocol_and_exact_structure_conversion() -> None:
    fake = FakeAtoms()

    assert isinstance(fake, ASEAtomsProtocol)
    backend = StructureBackend.create(fake)
    assert isinstance(backend, ASEAtomsBackend)
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


def test_kind_hint_selects_or_rejects_ase_backend() -> None:
    fake = FakeAtoms()

    assert isinstance(StructureBackend.create(fake, kind="ase"), ASEAtomsBackend)
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
        assert tuple(float(value) for value in actual) == pytest.approx(
            tuple(float(value) for value in expected)
        )
    for actual, expected in zip(
        atoms.get_scaled_positions(), slab.get_scaled_positions()
    ):
        assert tuple(float(value) for value in actual) == pytest.approx(
            tuple(float(value) for value in expected)
        )
    assert ASEAtomsView(atoms) is atoms
