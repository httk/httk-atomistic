"""``_unitcell_record_from_structure`` builds the record of a structure with a chemical composition."""

from fractions import Fraction

from httk.core.storage import content_id

from httk.atomistic import UnitcellStructure
from httk.atomistic.composition import ChemicalComposition
from httk.atomistic.storage.records import ChemicalCompositionRecord, _unitcell_record_from_structure


def test_record_carries_the_chemical_composition():
    structure = UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        [[0, 0, 0]],
        species_at_sites=["Fe"],
        chemical_composition=ChemicalComposition({"Fe": 1, "O": Fraction(1, 2)}, "implicit", {"O": Fraction(1, 10)}),
    )

    record = _unitcell_record_from_structure(structure)

    assert isinstance(record.chemical_composition, ChemicalCompositionRecord)
    assert [(value.element, value.amount, value.precision) for value in record.chemical_composition.amounts] == [
        ("Fe", Fraction(1), None),
        ("O", Fraction(1, 2), Fraction(1, 10)),
    ]
    assert content_id(record) == content_id(structure)
