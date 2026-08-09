from fractions import Fraction

import pytest
from httk.core import FracVector

from httk.atomistic import Cell, CellParams, Species, UnitcellStructure
from httk.atomistic._writing import _poscar_padded_token, _poscar_payload_from_structure, _poscar_token

CELL = Cell([[5, 0, 0], [0, 5, 0], [0, 0, 5]])


def _structure() -> UnitcellStructure:
    return UnitcellStructure(
        CELL,
        [[0, 0, 0], [Fraction(1, 3), Fraction(1, 2), 0], [Fraction(2, 3), 0, 0]],
        [Species("Si", ("Si",), (1.0,)), Species("O", ("O",), (1.0,))],
        ["Si", "O", "Si"],
    )


def test_poscar_tokens_are_decimal_and_coordinates_are_padded() -> None:
    assert _poscar_token(Fraction(1, 2)) == "0.5"
    assert _poscar_token(Fraction(1, 3)) == "0.3333333333333333"
    payload = _poscar_payload_from_structure(_structure())
    assert payload["coords"][1] == ["0.6666666666666667", "0.0000000000000000", "0.0000000000000000"]
    assert all("/" not in token for row in payload["coords"] for token in row)


def test_poscar_renders_irrational_hexagonal_basis() -> None:
    structure = UnitcellStructure(
        Cell(CellParams((3, 3, 5, 90, 90, 120)).basis),
        [[0, 0, 0]],
        [Species("Si", ("Si",), (1,))],
        ["Si"],
    )
    payload = _poscar_payload_from_structure(structure)
    tokens = [token for row in payload["cell"] for token in row]
    assert "2.5980762113533160" in tokens
    assert payload["scale"] == "1"
    assert all("/" not in token for token in tokens)


def test_poscar_exponent_tokens_are_not_padded() -> None:
    value = Fraction(10**100, 3)
    token = _poscar_token(value)
    assert "e" in token.lower()
    assert _poscar_padded_token(value) == token


def test_poscar_groups_sites_by_first_symbol_appearance() -> None:
    payload = _poscar_payload_from_structure(_structure())
    assert payload["symbols"] == ["Si", "O"]
    assert payload["counts"] == [2, 1]
    assert payload["coords"] == [
        ["0.0000000000000000", "0.0000000000000000", "0.0000000000000000"],
        ["0.6666666666666667", "0.0000000000000000", "0.0000000000000000"],
        ["0.3333333333333333", "0.5000000000000000", "0.0000000000000000"],
    ]


def test_poscar_keeps_distinct_same_symbol_species_groups() -> None:
    structure = UnitcellStructure(
        CELL,
        [[0, 0, 0], [Fraction(1, 4), 0, 0], [Fraction(1, 2), 0, 0], [Fraction(3, 4), 0, 0]],
        [
            Species("Fe1", ("Fe",), (1.0,)),
            Species("Fe2", ("Fe",), (1.0,)),
            Species("O", ("O",), (1.0,)),
        ],
        ["Fe1", "O", "Fe2", "O"],
    )
    payload = _poscar_payload_from_structure(structure)
    assert payload["symbols"] == ["Fe", "O", "Fe"]
    assert payload["counts"] == [1, 2, 1]
    assert sum(payload["counts"]) == len(payload["coords"]) == 4
    assert len({tuple(row) for row in payload["coords"]}) == 4


@pytest.mark.parametrize(
    "species",
    [
        Species("mixed", ("Si", "Ge"), (Fraction(1, 2), Fraction(1, 2))),
        Species("partial", ("Si",), (Fraction(1, 2),)),
    ],
)
def test_poscar_refuses_disordered_or_partial_species(species: Species) -> None:
    structure = UnitcellStructure(CELL, [[0, 0, 0]], [species], [species.name])
    with pytest.raises(ValueError, match="disorder/partial occupancy"):
        _poscar_payload_from_structure(structure)


def test_poscar_refuses_empty_structure() -> None:
    with pytest.raises(ValueError, match="empty structure"):
        _poscar_payload_from_structure(UnitcellStructure(CELL, [], [], []))


def test_poscar_comment_contains_formula_and_id() -> None:
    structure = _structure()
    comment = _poscar_payload_from_structure(structure)["comment"]
    assert structure.chemical_formula_reduced in comment
    assert structure.id in comment


def test_poscar_expands_asymmetric_unit() -> None:
    from httk.atomistic import ASUStructure, WyckoffSite

    sodium = Species("Na", ("Na",), (1.0,))
    structure = ASUStructure(
        CELL,
        225,
        [WyckoffSite("a", FracVector(()), "Na")],
        [sodium],
    )
    payload = _poscar_payload_from_structure(structure)
    assert payload["symbols"] == ["Na"]
    assert payload["counts"] == [4]


def test_poscar_save_load_roundtrip_and_suffixes(tmp_path) -> None:
    pytest.importorskip("httk.io")
    from httk.core import has_writer_for, load, save

    from httk.atomistic import same_crystal

    assert has_writer_for("POSCAR")
    original = _structure()
    for destination in (tmp_path / "POSCAR", tmp_path / "structure.vasp", tmp_path / "POSCAR.bz2"):
        save(original, destination)
        recovered = load(destination)
        assert recovered.cell.scale == original.cell.scale
        assert recovered.cell.unscaled_basis == original.cell.unscaled_basis
        assert same_crystal(original, recovered)


def test_poscar_repeated_symbol_groups_survive_save_load(tmp_path) -> None:
    pytest.importorskip("httk.io")
    from httk.core import has_writer_for, load, save

    assert has_writer_for("POSCAR")
    original = UnitcellStructure(
        CELL,
        [[0, 0, 0], [Fraction(1, 4), 0, 0], [Fraction(1, 2), 0, 0], [Fraction(3, 4), 0, 0]],
        [
            Species("Fe1", ("Fe",), (1.0,)),
            Species("Fe2", ("Fe",), (1.0,)),
            Species("O", ("O",), (1.0,)),
        ],
        ["Fe1", "O", "Fe2", "O"],
    )
    destination = tmp_path / "POSCAR"
    save(original, destination)
    recovered = load(destination)
    assert len(recovered.species) == 3
    assert [recovered.species_at_sites.count(species.name) for species in recovered.species] == [1, 2, 1]
    original_payload = _poscar_payload_from_structure(original)
    recovered_payload = _poscar_payload_from_structure(recovered)
    assert recovered_payload["symbols"] == original_payload["symbols"] == ["Fe", "O", "Fe"]
    assert recovered_payload["counts"] == original_payload["counts"] == [1, 2, 1]
