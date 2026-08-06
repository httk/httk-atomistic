import json
from fractions import Fraction

import pytest
from httk.core import FracVector, SurdVector, load_entry_type_definition
from httk.core.optimade import OptimadeDocument, OptimadeResource, OptimadeSchemaSnapshot

from httk.atomistic import (
    ASUStructure,
    ASUStructureView,
    Cell,
    PlainStructureView,
    Spacegroup,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    build_supercell,
    conventional_cell,
    recognize_asu,
    same_crystal,
)
from httk.atomistic._writing import _cif_payload_from_structure, _poscar_payload_from_structure
from httk.atomistic.models.structure.backend import StructureBackend
from httk.atomistic.models.structure.optimade import OptimadeStructure


def _structure(charge: object = None) -> UnitcellStructure:
    return UnitcellStructure(
        Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        [[0, 0, 0]],
        [Species("Si", ("Si",), (1,))],
        ["Si"],
        charge=charge,
    )


def test_charge_is_exact_and_none_is_not_zero() -> None:
    assert _structure().charge is None
    assert _structure("0").charge == Fraction(0)
    assert _structure() != _structure(0)
    assert UnitcellStructureView(_structure("3/2")).charge == Fraction(3, 2)
    assert _structure("3/2").numeric().charge == 1.5
    exact = _structure(Fraction(1, 3))
    assert same_crystal(exact, StructureBackend.create(exact.numeric()))
    assert same_crystal(_structure(), _structure(0)) is False


def test_charge_survives_asu_recognition_and_expansion() -> None:
    asu = ASUStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        225,
        [WyckoffSite("a", FracVector.create(()), "Si")],
        [Species("Si", ("Si",), (1,))],
        charge="2",
    )
    expanded = UnitcellStructureView(asu)
    recognized = recognize_asu(expanded, setting=asu.setting())

    assert expanded.charge == Fraction(2)
    assert recognized.charge == Fraction(2)
    assert UnitcellStructureView(recognized).charge == Fraction(2)
    assert ASUStructureView(asu).charge == Fraction(2)


def test_supercell_scales_charge() -> None:
    result = build_supercell(_structure("3/2"), [[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert result.structure.charge == Fraction(3)
    assert build_supercell(_structure(), 2).structure.charge is None


def test_conventional_cell_scales_charge_by_exact_site_multiplier() -> None:
    zero = SurdVector.create(0)._as_scalar()
    two = SurdVector.create(2)._as_scalar()
    four = SurdVector.create(4)._as_scalar()
    minus_two = SurdVector.create(-2)._as_scalar()
    root_three = SurdVector.sqrt_of(3)
    rhombohedral_basis = SurdVector._from_scalar_grid(
        [
            [zero, -(root_three * Fraction(4, 3)), four],
            [two, root_three * Fraction(2, 3), four],
            [minus_two, root_three * Fraction(2, 3), four],
        ],
        (3, 3),
    )
    asu = ASUStructure(
        rhombohedral_basis,
        166,
        [WyckoffSite("a", FracVector.create(()), "Bi")],
        [Species("Bi", ("Bi",), (1,))],
        transform=Spacegroup.for_setting("166:R").transform_from_standard,
        charge="2",
    )

    result = conventional_cell(asu)

    assert result.multiplier == Fraction(3)
    assert result.structure.charge == Fraction(6)


def test_optimade_charge_is_read_from_private_attribute() -> None:
    info = OptimadeDocument.create(json.dumps({"data": {"properties": {}}}), "https://example.test/info")
    document = OptimadeDocument.create(
        json.dumps({"data": [{"id": "one", "type": "structures", "attributes": {"_httk_charge": 2.5}}]}),
        "https://example.test/structures",
    )
    resource = OptimadeResource(document, 0, OptimadeSchemaSnapshot("structures", info))

    assert OptimadeStructure(resource).charge == Fraction(5, 2)


def test_writers_reject_charge_after_optimade_conversion() -> None:
    schema = load_entry_type_definition(
        "https://schemas.optimade.org/defs/v1.3/entrytypes/optimade/structures"
    )
    names = {
        "lattice_vectors": "remote_lattice",
        "fractional_site_positions": "remote_fractional",
        "species": "remote_species",
        "species_at_sites": "remote_site_species",
        "dimension_types": "remote_dimensions",
    }
    properties = {
        remote: {"$id": schema.properties[name].definition_id} for name, remote in names.items()
    }
    attributes = {
        names["lattice_vectors"]: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        names["fractional_site_positions"]: [[0, 0, 0]],
        names["species"]: [
            {
                "name": "Si",
                "chemical_symbols": ["Si"],
                "concentration": [1],
                "_httk_charges": [1],
            }
        ],
        names["species_at_sites"]: ["Si"],
        names["dimension_types"]: [1, 1, 1],
        "_httk_charge": 1,
    }
    info = OptimadeDocument.create(json.dumps({"data": {"properties": properties}}), "https://example.test/info")
    document = OptimadeDocument.create(
        json.dumps({"data": [{"id": "charged", "type": "structures", "attributes": attributes}]}),
        "https://example.test/structures",
    )
    resource = OptimadeResource(document, 0, OptimadeSchemaSnapshot("structures", info))

    with pytest.raises(ValueError, match="charge"):
        _cif_payload_from_structure(resource)
    with pytest.raises(ValueError, match="charge"):
        _poscar_payload_from_structure(resource)


def test_fundamental_domain_equality_includes_charge() -> None:
    species = [Species("Si", ("Si",), (1,))]
    first = ASUStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        225,
        [WyckoffSite("a", FracVector.create(()), "Si")],
        species,
        charge=0,
    )
    second = ASUStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        225,
        [WyckoffSite("a", FracVector.create(()), "Si")],
        species,
        charge=1,
    )

    assert first != second


@pytest.mark.parametrize("field", ["charge", "decorations"])
def test_plain_and_file_writers_reject_unencoded_state(field: str) -> None:
    species = Species("Si", ("Si",), (1,), charges=(1,)) if field == "decorations" else Species("Si", ("Si",), (1,))
    structure = UnitcellStructureView(
        UnitcellStructure(
            Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            [[0, 0, 0]],
            [species],
            ["Si"],
            charge=1 if field == "charge" else None,
        )
    )
    message = "charge" if field == "charge" else "charges"

    with pytest.raises(ValueError, match=message):
        PlainStructureView(structure)
    with pytest.raises(ValueError, match=message):
        _cif_payload_from_structure(structure)
    with pytest.raises(ValueError, match=message):
        _poscar_payload_from_structure(structure)
