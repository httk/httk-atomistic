"""CartesianSites: Cartesian coordinates held exactly as given, and the Cartesian POSCAR round trip."""

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy
import pytest
from httk.core import FracVector, SurdVector, load, save, unwrap
from httk.core.storage import content_id

from httk.atomistic import (
    CartesianSites,
    Cell,
    CellParams,
    NumericUnitcellStructureView,
    Sites,
    SitesView,
    UnitcellStructure,
    UnitcellStructureRecord,
    UnitcellStructureView,
    same_crystal,
    structure_tolerance,
)
from httk.atomistic.models.sites.cartesian import _cartesian_precision_factor

CUBIC = [["5.64", 0, 0], [0, "5.64", 0], [0, 0, "5.64"]]

SKEWED_CELL = "4.0 0.0 0.0\n1.5 3.5 0.0\n0.25 0.75 5.125"

SKEWED_CARTESIAN = """skewed cell
{scale}
4.0 0.0 0.0
1.5 3.5 0.0
0.25 0.75 5.125
Na Cl
1 2
Cartesian
0.0 0.0 0.0
1.234 2.5 0.125
-0.5 0.333 4.0625
"""

DIRECT = """rocksalt cell
1.0
5.64 0.0 0.0
0.0 5.64 0.0
0.0 0.0 5.64
Na Cl
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""


def _cartesian(precision: object = None) -> CartesianSites:
    return CartesianSites([[0, 0, 0], ["2.82", "2.82", "2.82"]], CUBIC, precision)


def _structure(sites: object) -> UnitcellStructure:
    return UnitcellStructure(Cell(CUBIC), sites, None, ["Na", "Cl"])  # type: ignore[arg-type]


def _load(text: str, tmp_path: Path, name: str = "POSCAR", **kwargs: Any) -> UnitcellStructure:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    structure = load(str(path), **kwargs)
    assert isinstance(structure, UnitcellStructure)
    return structure


def _coordinate_lines(text: str) -> list[list[Fraction]]:
    lines = text.splitlines()
    mode = next(index for index, line in enumerate(lines) if line.strip().lower() in ("cartesian", "direct"))
    return [[Fraction(token) for token in line.split()[:3]] for line in lines[mode + 1 :] if line.strip()]


def test_holds_values_and_derives_reduced_coords_exactly() -> None:
    sites = _cartesian("0.001")
    assert sites.cartesian_coords == SurdVector([[0, 0, 0], ["2.82", "2.82", "2.82"]])
    assert sites.basis == SurdVector(CUBIC)
    assert sites.cartesian_precision == Fraction(1, 1000)
    assert sites.reduced_coords == FracVector([[0, 0, 0], ["1/2", "1/2", "1/2"]])
    assert sites.precision == Fraction(1, 1000) / Fraction("5.64")
    assert _cartesian().precision is None


@pytest.mark.parametrize(
    ("coords", "basis"),
    [
        ([[0, 0]], CUBIC),
        ([], CUBIC),
        ([[0, 0, 0]], [[1, 0], [0, 1]]),
        ([[0, 0, 0]], [[1, 0, 0], [0, 1, 0], [1, 1, 0]]),
    ],
)
def test_rejects_bad_shapes_and_degenerate_basis(coords: object, basis: object) -> None:
    with pytest.raises(ValueError, match="CartesianSites"):
        CartesianSites(coords, basis)  # type: ignore[arg-type]


def test_irrational_reduced_coords_raise() -> None:
    hexagonal = CellParams((3, 3, 5, 90, 90, 120)).basis
    sites = CartesianSites([[0, 1, 0]], hexagonal)
    with pytest.raises(ValueError, match="irrational"):
        sites.reduced_coords  # noqa: B018 - the property access is the behavior under test


def test_raw_arrays_are_not_adopted_as_cartesian() -> None:
    assert type(unwrap(SitesView([[0, 0, 0]]))) is not CartesianSites
    sites = _cartesian()
    assert SitesView(sites, kind="cartesian").unwrap() is sites
    with pytest.raises(TypeError):
        SitesView(sites, kind="plain")


@pytest.mark.parametrize("precision", [None, "0.001", Fraction(1, 3)])
def test_repr_round_trips_through_the_constructor(precision: object) -> None:
    sites = _cartesian(precision)
    text = repr(sites)
    assert text.startswith('CartesianSites(cartesian_coords=[[0, 0, 0], ["2.82", "2.82", "2.82"]], basis=[["5.64", 0')
    rebuilt = eval(text, {"CartesianSites": CartesianSites})
    assert (rebuilt.cartesian_coords, rebuilt.basis, rebuilt.cartesian_precision) == (
        sites.cartesian_coords,
        sites.basis,
        sites.cartesian_precision,
    )


def test_structure_requires_the_cell_basis() -> None:
    with pytest.raises(ValueError, match="basis"):
        UnitcellStructure(Cell(CUBIC, 2), _cartesian(), None, ["Na", "Cl"])


def test_structure_keeps_the_backend_through_views() -> None:
    sites = _cartesian("0.001")
    structure = _structure(sites)
    assert unwrap(structure.sites) is sites
    assert structure.cartesian_sites() is sites.cartesian_coords
    view = UnitcellStructureView(structure)
    assert view.unview() is structure
    assert unwrap(view.sites) is sites
    assert numpy.asarray(NumericUnitcellStructureView(structure).sites.reduced_coords).tolist() == [
        [0, 0, 0],
        [0.5, 0.5, 0.5],
    ]


def test_content_id_matches_the_reduced_equivalent() -> None:
    # Precision is part of content identity, so both carry the same precision metadata.
    for precision in (None, "0.001"):
        cartesian = _structure(_cartesian(precision))
        reduced = _structure(Sites([[0, 0, 0], ["0.5", "0.5", "0.5"]], cartesian.sites.precision))
        assert cartesian == reduced
        assert content_id(cartesian) == content_id(reduced)
        record = UnitcellStructureRecord.from_structure(cartesian)
        assert content_id(record) == content_id(UnitcellStructureRecord.from_structure(reduced))


@pytest.mark.parametrize("scale", ["1.0", "2.5"])
def test_cartesian_poscar_read_keeps_values_and_old_precision(scale: str, tmp_path: Path) -> None:
    text = SKEWED_CARTESIAN.format(scale=scale)
    structure = _load(text, tmp_path)
    sites = unwrap(structure.sites)
    assert isinstance(sites, CartesianSites)
    written = _coordinate_lines(text)
    assert structure.cartesian_sites() == Fraction(scale) * SurdVector(written)
    raw_basis = SurdVector([["4.0", 0, 0], ["1.5", "3.5", 0], ["0.25", "0.75", "5.125"]])
    token_precision = Fraction(load(str(tmp_path / "POSCAR"), raw=True)["coordinate_precision"])
    # The rule the builder applied before CartesianSites: token precision times the raw-basis factor.
    assert structure.sites.precision == token_precision * _cartesian_precision_factor(raw_basis)
    assert sites.cartesian_precision == token_precision * Fraction(scale)
    # The stated length is reported directly (the cell precision here is the same length).
    assert structure.cartesian_precision() == token_precision * Fraction(scale)
    assert structure.sites.reduced_coords == FracVector(SurdVector(written) * raw_basis.inv())


@pytest.mark.parametrize("scale", ["1.0", "2.5"])
@pytest.mark.parametrize("cell", [SKEWED_CELL, "3.0 0.0 0.0\n0.0 4.5 0.0\n0.0 0.0 7.25"])
def test_precision_override_is_the_cartesian_precision(scale: str, cell: str, tmp_path: Path) -> None:
    text = SKEWED_CARTESIAN.format(scale=scale).replace(SKEWED_CELL, cell)
    structure = _load(text, tmp_path, precision=5e-4)
    assert isinstance(unwrap(structure.sites), CartesianSites)
    assert structure.cartesian_precision() == Fraction(1, 2000)
    assert structure_tolerance(structure) == 2 * 5e-4


@pytest.mark.parametrize("scale", ["1.0", "2.5"])
def test_cartesian_poscar_writes_back_its_numbers(scale: str, tmp_path: Path) -> None:
    text = SKEWED_CARTESIAN.format(scale=scale)
    original = _load(text, tmp_path, precision=5e-4)
    destination = tmp_path / "out.vasp"
    save(original, destination)
    written = destination.read_text(encoding="utf-8")
    assert "Cartesian" in written
    assert _coordinate_lines(written) == _coordinate_lines(text)
    # The writer pads to 16 digits, so the digit-derived precision differs; state it for both reads.
    reread = _load(written, tmp_path, "REREAD.vasp", precision=5e-4)
    assert isinstance(unwrap(reread.sites), CartesianSites)
    assert reread == original
    assert same_crystal(reread, original)


def test_direct_poscar_is_unaffected(tmp_path: Path) -> None:
    structure = _load(DIRECT, tmp_path)
    assert not isinstance(unwrap(structure.sites), CartesianSites)
    destination = tmp_path / "out.vasp"
    save(structure, destination)
    written = destination.read_text(encoding="utf-8")
    assert "Direct" in written
    assert _coordinate_lines(written) == _coordinate_lines(DIRECT)
