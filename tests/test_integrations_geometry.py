"""Tests for the private neutral geometry payload shared by the simulation-code integrations."""

import io
from fractions import Fraction
from typing import Any

import pytest
from httk.core import SurdVector, exactmath

from httk.atomistic import Species, Trajectory, UnitcellStructure
from httk.atomistic._loading import _structure_from_poscar
from httk.atomistic.integrations._geometry import (
    _element_from_label,
    _geometry_from_structure,
    _number,
    _render,
    _structure_from_geometry,
    _trajectory_from_frames,
    _write_text,
)
from httk.atomistic.integrations.vasp.io import read_poscar

BOHR = "0.52917720859"
SPECIES = [{"name": "Na", "element": "Na", "mass": None}, {"name": "Cl", "element": "Cl", "mass": None}]


def _payload(**changes: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": "test",
        "cell": [["4.0", "0", "0"], ["0", "4.0", "0"], ["0", "0", "4.0"]],
        "cell_unit": "angstrom",
        "positions": [["0", "0", "0"], ["0.5", "0.5", "0.5"]],
        "positions_kind": "fractional",
        "species": SPECIES,
        "species_at_sites": ["Na", "Cl"],
    }
    payload.update(changes)
    return payload


def _rocksalt(a: Any) -> UnitcellStructure:
    half = Fraction(1, 2)
    coords = [[0, 0, 0], [0, half, half], [half, 0, half], [half, half, 0]]
    coords += [[(x + half) % 1, y, z] for x, y, z in coords]
    return UnitcellStructure(
        [[a, 0, 0], [0, a, 0], [0, 0, a]],
        coords,
        [Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))],
        ["Na"] * 4 + ["Cl"] * 4,
    )


def test_bohr_and_angstrom_units_are_exact() -> None:
    bohr = Fraction(BOHR)
    in_bohr = _structure_from_geometry(_payload(cell_unit="bohr", bohr_angstrom=BOHR))
    in_angstrom = _structure_from_geometry(_payload())
    assert in_bohr.cell.basis == SurdVector([[4 * bohr if i == j else 0 for j in range(3)] for i in range(3)])
    assert in_angstrom.cell.basis == SurdVector([[4 if i == j else 0 for j in range(3)] for i in range(3)])
    assert in_bohr.cell.precision == Fraction(1, 10) * bohr
    with pytest.raises(ValueError, match="bohr_angstrom"):
        _structure_from_geometry(_payload(cell_unit="bohr"))


def test_cartesian_positions_reduce_exactly_across_units() -> None:
    # Cell in Bohr, Cartesian positions in Å: the Bohr constant must cancel exactly.
    bohr = Fraction(BOHR)
    structure = _structure_from_geometry(
        _payload(
            cell=[["8.0", "0", "0"], ["0", "8.0", "0"], ["0", "0", "8.0"]],
            cell_unit="bohr",
            bohr_angstrom=BOHR,
            positions=[["0", "0", "0"], [4 * bohr, "1.0", "2.5d-1"]],
            positions_kind="cartesian",
            positions_unit="angstrom",
        )
    )
    assert structure.sites.reduced_coords.to_fractions()[1] == [Fraction(1, 2), 1 / (8 * bohr), 1 / (32 * bohr)]


def test_exact_scalar_cell_entries() -> None:
    root = exactmath.sqrt(3, exact=True) / 2
    hexagonal = [[1, 0, 0], [Fraction(-1, 2), root, 0], ["0", "0", "1.5"]]
    structure = _structure_from_geometry(_payload(cell=hexagonal, positions=[["0", "0", "0"], ["0.5", "0", "0"]]))
    assert structure.cell.basis == SurdVector.from_radicand_map(
        {
            1: [[1, 0, 0], [Fraction(-1, 2), 0, 0], [0, 0, Fraction(3, 2)]],
            3: [[0, 0, 0], [0, Fraction(1, 2), 0], [0, 0, 0]],
        }
    )
    # Exact entries make no claim; only the written "1.5" does.
    assert structure.cell.precision == Fraction(1, 10)
    cartesian = _structure_from_geometry(
        _payload(
            cell=hexagonal,
            positions=[["0", "0", "0"], ["0.25", "0", "0"]],
            positions_kind="cartesian",
            positions_unit="angstrom",
        )
    )
    assert cartesian.sites.reduced_coords.to_fractions()[1] == [Fraction(1, 4), 0, 0]
    assert cartesian.coordinate_precision is not None and cartesian.coordinate_precision > Fraction(1, 100)


@pytest.mark.parametrize("kind", ["Direct", "Cartesian"])
def test_precision_matches_poscar(kind: str) -> None:
    cell = [["4.0", "0.0", "0.0"], ["1.5", "3.25", "0.0"], ["0.5", "0.75", "5.125"]]
    positions = [["0.0", "0.0", "0.0"], ["0.125", "0.2500", "0.375"]]
    text = "x\n1.0\n" + "".join(" ".join(row) + "\n" for row in cell) + "Na Cl\n1 1\n" + kind + "\n"
    text += "".join(" ".join(row) + "\n" for row in positions)
    poscar = _structure_from_poscar(read_poscar(text.splitlines(keepends=True)))
    geometry = _structure_from_geometry(
        _payload(
            cell=cell,
            positions=positions,
            positions_kind="cartesian" if kind == "Cartesian" else "fractional",
            positions_unit="angstrom",
        )
    )
    assert geometry.cell == poscar.cell
    assert geometry.sites == poscar.sites
    assert geometry.cell.precision == poscar.cell.precision is not None
    assert geometry.coordinate_precision == poscar.coordinate_precision is not None


def test_precision_override() -> None:
    structure = _structure_from_geometry(_payload(precision_override=1e-3))
    assert structure.cell.precision == Fraction(1, 1000)
    assert structure.cartesian_precision() == Fraction(1, 1000)
    with pytest.raises(ValueError, match="precision"):
        _structure_from_geometry(_payload(precision_override=-1.0))


def test_periodicity() -> None:
    assert _structure_from_geometry(_payload()).cell.periodicity == (True, True, True)
    slab = _structure_from_geometry(_payload(periodicity=[True, True, False]))
    assert slab.cell.periodicity == (True, True, False)
    with pytest.raises(ValueError, match="periodicity"):
        _structure_from_geometry(_payload(periodicity=[1, 1, 0]))


def test_species_mass() -> None:
    species = [{"name": "Na", "element": "Na", "mass": "22.99"}, {"name": "Cl", "element": "Cl", "mass": "3.5453D1"}]
    structure = _structure_from_geometry(_payload(species=species))
    assert [item.mass for item in structure.species] == [(22.99,), (35.453,)]


def test_malformed_payloads_are_refused() -> None:
    with pytest.raises(ValueError, match="undeclared species \\['K'\\]"):
        _structure_from_geometry(_payload(species_at_sites=["Na", "K"]))
    with pytest.raises(ValueError, match="one species name per site"):
        _structure_from_geometry(_payload(species_at_sites=["Na"]))
    with pytest.raises(ValueError, match="cell must be 3 rows"):
        _structure_from_geometry(_payload(cell=[["1", "0", "0"]]))
    with pytest.raises(ValueError, match="exact numbers"):
        _structure_from_geometry(_payload(positions=[[0.0, 0, 0], [0, 0, 0]]))
    with pytest.raises(ValueError, match="Unknown element"):
        _structure_from_geometry(_payload(species=[{"name": "Na", "element": "Xx"}, SPECIES[1]]))
    with pytest.raises(ValueError, match="positions_unit"):
        _structure_from_geometry(_payload(positions_kind="cartesian"))
    with pytest.raises(ValueError, match="format"):
        _structure_from_geometry({key: value for key, value in _payload().items() if key != "format"})


def test_serializer_round_trip() -> None:
    exact = _rocksalt(5)
    payload = _geometry_from_structure(exact, "t")
    assert payload["cell_unit"] == "angstrom" and payload["positions_kind"] == "fractional"
    assert payload["positions"][1] == ["0", "0.5", "0.5"]
    assert "raw" not in payload
    assert _structure_from_geometry(payload) == exact
    # Decimal cells round-trip their geometry exactly; the written digits then state a precision.
    decimal = _rocksalt(Fraction("5.64"))
    rebuilt = _structure_from_geometry(_geometry_from_structure(decimal, "t"))
    assert rebuilt.cell == decimal.cell and rebuilt.sites == decimal.sites
    assert rebuilt.cell.precision == Fraction(1, 100)


def test_serializer_refuses_disorder_and_empty() -> None:
    alloy = UnitcellStructure(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 0]],
        [Species("FeNi", ("Fe", "Ni"), (Fraction(1, 2), Fraction(1, 2)))],
        ["FeNi"],
    )
    with pytest.raises(ValueError, match="disorder/partial occupancy"):
        _geometry_from_structure(alloy, "t")
    empty = UnitcellStructure([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [], [], [])
    with pytest.raises(ValueError, match="empty structure"):
        _geometry_from_structure(empty, "t")


def test_frames_build_native_trajectory_with_per_frame_precision() -> None:
    first = {key: value for key, value in _payload().items() if key != "format"}
    second = dict(first, cell=[["4.001", "0", "0"], ["0", "4.002", "0"], ["0", "0", "4.003"]])
    trajectory = _trajectory_from_frames({"format": "test", "frames": [first, second]})
    assert isinstance(trajectory, Trajectory) and trajectory.nframes == 2
    assert trajectory.frame(0).cell.precision == Fraction(1, 10)
    assert trajectory.frame(1).cell.precision == Fraction(1, 1000)
    assert _trajectory_from_frames({"format": "test", "frames": [first]}).nframes == 1
    with pytest.raises(ValueError, match="geometry frame 1"):
        _trajectory_from_frames({"format": "test", "frames": [first, dict(first, cell_unit="pm")]})


def test_number_and_render() -> None:
    assert _number("1.0d-3") == Fraction(1, 1000)
    assert _number(" -2.5D0 ") == Fraction(-5, 2)
    assert _number(".5E+1") == 5
    for garbage in ("", "1/3", "nan", "1.0q3", "1,0"):
        with pytest.raises(ValueError):
            _number(garbage)
    assert _render(Fraction(1, 2)) == "0.5"
    assert _render(3) == "3"
    assert _render(Fraction(1, 3)) == "0.3333333333333333"


@pytest.mark.parametrize(
    ("label", "element"),
    [
        ("Fe1", "Fe"),
        ("FE", "Fe"),
        ("O_2", "O"),
        ("Si-a", "Si"),
        ("Xx", None),
        ("C", "C"),
        ("Co", "Co"),
        ("Ca2", "Ca"),
        ("O1_up", "O"),
        ("", None),
        ("1Fe", None),
    ],
)
def test_element_from_label(label: str, element: str | None) -> None:
    assert _element_from_label(label) == element


def _render_payload(payload: Any, *, suffix: str = "") -> str:
    return f"rendered {payload['format']}{suffix}\n"


def test_write_text(tmp_path: Any) -> None:
    raw = "original\r\ntext\n"
    path = tmp_path / "raw.txt"
    _write_text(path, {"format": "t", "raw": raw}, {}, _render_payload)
    assert path.read_bytes() == raw.encode()
    with pytest.raises(ValueError, match="raw"):
        _write_text(path, {"format": "t", "raw": raw}, {"suffix": "!"}, _render_payload)
    _write_text(path, {"format": "t"}, {"suffix": "!"}, _render_payload)
    assert path.read_text(encoding="utf-8") == "rendered t!\n"
    stream = io.StringIO()
    _write_text(stream, {"format": "t"}, {}, _render_payload)
    assert stream.getvalue() == "rendered t\n"
