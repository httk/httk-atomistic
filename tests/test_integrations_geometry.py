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


def _parameters(angles: tuple[str, str, str], **changes: Any) -> dict[str, Any]:
    payload = _payload(**changes)
    del payload["cell"]
    payload.setdefault("cell_parameters", ["3.00", "4.00", "5.00", *angles])
    return payload


@pytest.mark.parametrize("angles", [("90", "104.5", "90"), ("90", "90", "120"), ("80", "95", "110")])
def test_cell_parameters_use_the_cellparams_frame(angles: tuple[str, str, str]) -> None:
    basis = _structure_from_geometry(_parameters(angles)).cell.basis
    entry = basis._element
    # a along +x, b in the xy plane with positive y, c with positive z (right-handed).
    assert [entry((0, column)).sign() for column in range(3)] == [1, 0, 0]
    assert entry((1, 1)).sign() == 1 and entry((1, 2)).sign() == 0
    assert entry((2, 2)).sign() == 1 and basis.det().sign() == 1


def test_cell_parameters_monoclinic_fractional() -> None:
    structure = _structure_from_geometry(_parameters(("90", "104.5", "90")))
    # The structure keeps the parameters themselves, not the approximated basis's.
    assert list(structure.cell.lengths) == [3, 4, 5]
    assert structure.cell.angles == (90, Fraction("104.5"), 90)
    assert structure.cell.precision == Fraction(1, 100)
    assert structure.sites.reduced_coords.to_fractions() == [[0, 0, 0], [Fraction(1, 2)] * 3]
    integers = _structure_from_geometry(
        _parameters(("90", "104.5", "90"), cell_parameters=[3, "4", "5", 90, "104.5", 90])
    )
    assert integers.cell.precision is None
    bohr = _structure_from_geometry(_parameters(("90", "104.5", "90"), cell_unit="bohr", bohr_angstrom=BOHR))
    assert list(bohr.cell.lengths) == [length * Fraction(BOHR) for length in (3, 4, 5)]
    assert bohr.cell.angles == (90, Fraction("104.5"), 90)
    assert bohr.cell.precision == Fraction(1, 100) * Fraction(BOHR)


def test_cell_parameters_monoclinic_cartesian() -> None:
    positions = [["0", "0", "0"], ["1.2500000000000000", "0.5000000000000000", "2.0000000000000000"]]
    cartesian = {"positions": positions, "positions_kind": "cartesian", "positions_unit": "angstrom"}
    structure = _structure_from_geometry(_parameters(("90", "104.5", "90"), **cartesian))
    basis = structure.cell.basis.coefficient(1).to_fractions()  # the approximated basis is rational
    reduced = structure.sites.reduced_coords.to_fractions()[1]
    assert [sum(reduced[i] * basis[i][j] for i in range(3)) for j in range(3)] == [Fraction("1.25"), Fraction("0.5"), 2]
    # The 16 written digits claim ~1e-16; the approximated basis caps the claim near 1e-11.
    precision = structure.sites.precision
    assert precision is not None and Fraction(1, 10**12) < precision < Fraction(1, 10**9)
    exact = _structure_from_geometry(_parameters(("90", "90", "90"), **cartesian))
    assert exact.sites.precision is not None and exact.sites.precision < Fraction(1, 10**15)
    hexagonal = _structure_from_geometry(_parameters(("90", "90", "120"), **cartesian))
    # Irrational fractional coordinates in the surd cell are rounded and claim no better.
    assert hexagonal.sites.precision == exactmath.default_accuracy
    override = _structure_from_geometry(_parameters(("90", "104.5", "90"), precision_override=0.001, **cartesian))
    assert override.cell.precision == Fraction(1, 1000)
    assert override.sites.precision == Fraction(1, 1000) / 5


def test_cell_parameters_are_validated() -> None:
    with pytest.raises(ValueError, match="either cell or cell_parameters"):
        _structure_from_geometry(_payload(cell_parameters=["3", "4", "5", "90", "90", "90"]))
    with pytest.raises(ValueError, match="six entries"):
        _structure_from_geometry(_parameters(("90", "90", "90"), cell_parameters=["3", "4", "5"]))
    with pytest.raises(ValueError, match="rational"):
        _structure_from_geometry(
            _parameters(("90", "90", "90"), cell_parameters=[exactmath.sqrt(2, exact=True), 4, 5, 90, 90, 90])
        )
    with pytest.raises(ValueError, match="degenerate"):
        _structure_from_geometry(_parameters(("60", "60", "120")))


def test_cell_parameters_with_partial_periodicity_keep_the_basis() -> None:
    full = _structure_from_geometry(_parameters(("90", "104.5", "90")))
    slab = _structure_from_geometry(_parameters(("90", "104.5", "90"), periodicity=[True, True, False]))
    assert slab.cell.periodicity == (True, True, False)
    assert slab.cell.basis == full.cell.basis and slab.cell.precision == Fraction(1, 100)


@pytest.mark.parametrize("parameters", [False, True])
def test_cell_precision_key(parameters: bool) -> None:
    def build(**changes: Any) -> UnitcellStructure:
        return _structure_from_geometry(
            _parameters(("90", "90", "90"), **changes) if parameters else _payload(**changes)
        )

    assert build(cell_precision="0.005").cell.precision == Fraction(1, 200)
    assert build(cell_precision=Fraction(1, 200), cell_unit="bohr", bohr_angstrom=BOHR).cell.precision == Fraction(
        1, 200
    ) * Fraction(BOHR)
    assert build(cell_precision="0.005", precision_override=0.001).cell.precision == Fraction(1, 1000)
    with pytest.raises(ValueError, match="cell_precision must be positive"):
        build(cell_precision="-1")
    with pytest.raises(ValueError, match="rational"):
        build(cell_precision=exactmath.sqrt(2, exact=True))


def test_positions_precision_key() -> None:
    fractional = _structure_from_geometry(_payload(positions_precision="0.001"))
    assert fractional.sites.precision == Fraction(1, 1000)
    cartesian = {"positions_kind": "cartesian", "positions_unit": "angstrom"}
    digits = _structure_from_geometry(_payload(positions=[["0", "0", "0"], ["2.00", "2.00", "2.00"]], **cartesian))
    given = _structure_from_geometry(
        _payload(positions=[[0, 0, 0], [2, 2, 2]], positions_precision=Fraction(1, 100), **cartesian)
    )
    # A Cartesian precision converts to fractional exactly as the digits of "2.00" do.
    assert given.sites.precision == digits.sites.precision == Fraction(1, 400)
    bohr = _structure_from_geometry(
        _payload(
            positions=[[0, 0, 0], [2, 2, 2]],
            positions_precision="0.01",
            positions_unit="bohr",
            bohr_angstrom=BOHR,
            positions_kind="cartesian",
        )
    )
    assert bohr.sites.precision == Fraction(1, 400) * Fraction(BOHR)
    override = _structure_from_geometry(_payload(positions_precision="0.001", precision_override=0.004))
    assert override.sites.precision == Fraction(1, 1000)  # 0.004 Å / 4 Å
    with pytest.raises(ValueError, match="positions_precision must be positive"):
        _structure_from_geometry(_payload(positions_precision=0))
