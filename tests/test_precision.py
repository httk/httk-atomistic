"""Precision carried by the structure classes.

The failure mode this feature has is *silent loss*, not breakage: an optional precision
that a view forgets to carry becomes ``None`` with no error, and a tolerance derived from
it quietly falls back to a constant. So every reconstruction site gets its own test rather
than one representative test standing in for the rest.
"""

import fractions

import pytest
from httk.core import FracVector

from httk.atomistic import (
    Cell,
    CellClassView,
    CellNumericView,
    CellParamsView,
    Sites,
    SitesClassView,
    SitesNumericView,
    Species,
    Structure,
    StructureSimpleView,
    same_crystal,
)

pytest.importorskip("httk.io", reason="the readers live in httk-io")

import io  # noqa: E402

from httk.io.vasp import read_poscar  # noqa: E402

from httk.atomistic import (  # noqa: E402
    ASUSite,
    ASUStructure,
    Spacegroup,
    recognize_asu,
    structure_from_poscar,
)

F = fractions.Fraction

CUBIC = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
COORD_PRECISION = F(1, 10000)
BASIS_PRECISION = F(1, 1000)


def _species() -> list[Species]:
    return [Species(name="Na", chemical_symbols=("Na",), concentration=(1.0,))]


def _structure() -> Structure:
    return Structure(
        Cell(CUBIC, 1, BASIS_PRECISION),
        Sites([[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]], COORD_PRECISION),
        _species(),
        ["Na", "Na"],
    )


# --- storing it ---


@pytest.mark.parametrize(
    ("given", "expected"),
    [(1e-4, F(1, 10000)), ("1/10000", F(1, 10000)), (F(1, 10000), F(1, 10000)), (0.0003, F(3, 10000)), (None, None)],
    ids=["float", "rational-string", "fraction", "esd-like-float", "unknown"],
)
def test_a_precision_is_stored_exactly(given: object, expected: F | None) -> None:
    """A float goes through its decimal spelling, so 1e-4 is 1/10000.

    Embedding it the way a float literally holds it would record a number nobody stated —
    the same trap that makes a CIF's ``0.3333`` become 6004199023210345/18014398509481984.
    """
    assert Cell(CUBIC, 1, given).precision == expected
    assert Sites([[0, 0, 0]], given).precision == expected


def test_precision_defaults_to_unknown() -> None:
    """Unknown is a real answer, and not the same as claiming exactness."""
    assert Cell(CUBIC).precision is None
    assert Sites([[0, 0, 0]]).precision is None
    assert Structure(CUBIC, [[0, 0, 0]], _species(), ["Na"]).coordinate_precision is None


@pytest.mark.parametrize("bad", [0, -1, "-1/2"])
def test_a_non_positive_precision_is_rejected(bad: object) -> None:
    """Zero would claim an exactness no measurement has; use None to say unknown."""
    with pytest.raises(ValueError, match="strictly positive"):
        Cell(CUBIC, 1, bad)
    with pytest.raises(ValueError, match="strictly positive"):
        Sites([[0, 0, 0]], bad)


# --- carrying it through every reconstruction site ---


def test_cell_class_view_carries_precision() -> None:
    assert CellClassView(Cell(CUBIC, 1, BASIS_PRECISION)).precision == BASIS_PRECISION


def test_cell_params_view_carries_precision() -> None:
    """This view rebuilds a reference Cell internally; it must rebuild it with the precision."""
    view = CellParamsView(Cell(CUBIC, 1, BASIS_PRECISION))
    assert view.a == pytest.approx(5.0)
    assert view._backend.precision == BASIS_PRECISION


def test_cell_numeric_view_carries_precision_as_a_float() -> None:
    pytest.importorskip("numpy")
    assert CellNumericView(Cell(CUBIC, 1, BASIS_PRECISION)).precision == pytest.approx(1e-3)


def test_sites_class_view_carries_precision() -> None:
    assert SitesClassView(Sites([[0, 0, 0]], COORD_PRECISION)).precision == COORD_PRECISION


def test_sites_numeric_view_carries_precision_as_a_float() -> None:
    pytest.importorskip("numpy")
    assert SitesNumericView(Sites([[0, 0, 0]], COORD_PRECISION)).precision == pytest.approx(1e-4)


def test_structure_view_carries_both_precisions() -> None:
    view = StructureSimpleView(_structure())
    assert view.coordinate_precision == COORD_PRECISION
    assert view.basis_precision == BASIS_PRECISION


def test_precision_survives_repeated_rewrapping() -> None:
    """Views are meant to be applied freely, so a round trip must not erode anything."""
    cell = Cell(CUBIC, 1, BASIS_PRECISION)
    assert CellClassView(CellClassView(CellClassView(cell))).precision == BASIS_PRECISION

    sites = Sites([[0, 0, 0]], COORD_PRECISION)
    assert SitesClassView(SitesClassView(sites)).precision == COORD_PRECISION


def test_numeric_layer_round_trips_back_to_the_exact_value() -> None:
    pytest.importorskip("numpy")
    structure = _structure()
    assert structure.numeric().cell.precision == pytest.approx(1e-3)
    assert structure.numeric().exact.basis_precision == BASIS_PRECISION


# --- a backend with no source of precision ---


def test_a_backend_that_knows_no_precision_reports_unknown() -> None:
    """`CellParams` and the primitive backends have nothing to derive one from.

    They inherit the concrete `None` from the API rather than being forced to invent a
    value, which is also what keeps out-of-tree backends working unchanged.
    """
    from httk.atomistic import CellBackend, CellParams, SitesBackend

    assert CellParams((5, 5, 5, 90, 90, 90)).precision is None
    assert CellBackend.create([[5, 0, 0], [0, 5, 0], [0, 0, 5]]).precision is None
    assert SitesBackend.create([[0, 0, 0]]).precision is None


# --- the derived Cartesian value ---


def test_cartesian_precision_scales_with_the_cell() -> None:
    """A tolerance is a distance, and a fractional precision is not."""
    small = Structure(Cell(CUBIC), Sites([[0, 0, 0]], COORD_PRECISION), _species(), ["Na"])
    large = Structure(
        Cell([[30, 0, 0], [0, 30, 0], [0, 0, 30]]), Sites([[0, 0, 0]], COORD_PRECISION), _species(), ["Na"]
    )
    assert float(small.cartesian_precision()) == pytest.approx(5e-4)
    assert float(large.cartesian_precision()) == pytest.approx(3e-3)


def test_cartesian_precision_uses_the_longest_edge() -> None:
    """The conservative choice: the largest displacement the uncertainty can produce."""
    oblong = Structure(
        Cell([[2, 0, 0], [0, 5, 0], [0, 0, 20]]), Sites([[0, 0, 0]], COORD_PRECISION), _species(), ["Na"]
    )
    assert float(oblong.cartesian_precision()) == pytest.approx(2e-3)


def test_a_coarse_cell_precision_dominates() -> None:
    """A cell known only to 1e-3 cannot place an atom better than that."""
    structure = _structure()
    assert float(structure.cartesian_precision()) == pytest.approx(1e-3)

    sharper_cell = Structure(Cell(CUBIC, 1, F(1, 1000000)), Sites([[0, 0, 0]], COORD_PRECISION), _species(), ["Na"])
    assert float(sharper_cell.cartesian_precision()) == pytest.approx(5e-4)


def test_cartesian_precision_is_unknown_when_the_coordinates_are() -> None:
    structure = Structure(Cell(CUBIC, 1, BASIS_PRECISION), Sites([[0, 0, 0]]), _species(), ["Na"])
    assert structure.cartesian_precision() is None


# --- precision is not identity ---


def test_precision_does_not_affect_equality() -> None:
    """The same atoms read from a more carefully written file are the same structure."""
    assert Cell(CUBIC, 1, F(1, 10)) == Cell(CUBIC, 1, F(1, 1000000)) == Cell(CUBIC)
    assert Sites([[0, 0, 0]], F(1, 10)) == Sites([[0, 0, 0]])

    precise = _structure()
    vague = Structure(Cell(CUBIC), Sites([[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]]), _species(), ["Na", "Na"])
    assert precise == vague
    assert same_crystal(precise, vague)


# --- precision arriving from a file ---


def _poscar(*, scale: str = "1.0", mode: str = "Direct", coords: str = "0.0000 0.0000 0.0000") -> str:
    return (
        "NaCl\n"
        f"{scale}\n"
        "5.6400 0.0000 0.0000\n0.0000 5.6400 0.0000\n0.0000 0.0000 5.6400\n"
        "Na\n1\n"
        f"{mode}\n{coords}\n"
    )


def test_poscar_direct_coordinates_pass_their_precision_through() -> None:
    """Direct coordinates are already fractional, and the scale cancels for them."""
    structure = structure_from_poscar(read_poscar(io.StringIO(_poscar(coords="0.5000 0.5000 0.5000"))))
    assert structure.coordinate_precision == F(1, 10000)
    assert structure.basis_precision == F(1, 10000)


def test_poscar_cartesian_coordinates_are_converted_to_fractional() -> None:
    """A Cartesian precision is a length; dividing by the shortest edge makes it fractional."""
    structure = structure_from_poscar(
        read_poscar(io.StringIO(_poscar(mode="Cartesian", coords="2.8200 2.8200 2.8200")))
    )
    assert float(structure.coordinate_precision) == pytest.approx(1e-4 / 5.64)


def test_the_poscar_scale_multiplies_the_basis_precision() -> None:
    """The cell entries are scaled, so their absolute precision scales with them."""
    assert structure_from_poscar(read_poscar(io.StringIO(_poscar(scale="1.0")))).basis_precision == F(1, 10000)
    assert structure_from_poscar(read_poscar(io.StringIO(_poscar(scale="2.0")))).basis_precision == F(1, 5000)


def test_the_scales_own_digits_are_not_charged_as_uncertainty() -> None:
    """The scaling factor defines units; it is not measured apart from the rows it scales.

    Charging its digits would double-count the same measurement, and would declare the cell
    of a 5.64 A structure good to only half an angstrom because the file wrote ``1.0``.
    """
    structure = structure_from_poscar(read_poscar(io.StringIO(_poscar(scale="1.0"))))
    assert float(structure.basis_precision) == pytest.approx(1e-4)


def test_an_asu_structure_carries_and_propagates_its_precision() -> None:
    """Recorded in the structure's own setting, so it needs no transforming on the way out."""
    asu = ASUStructure(
        Cell(CUBIC, 1, BASIS_PRECISION),
        225,
        [ASUSite("a", FracVector.create(()), "Na")],
        _species(),
        coordinate_precision=COORD_PRECISION,
    )
    assert asu.coordinate_precision == COORD_PRECISION
    assert asu.expand_sites().precision == COORD_PRECISION

    expanded = StructureSimpleView(asu)
    assert expanded.coordinate_precision == COORD_PRECISION
    assert expanded.basis_precision == BASIS_PRECISION


def test_recognition_carries_the_precision_onto_the_asu() -> None:
    """Nothing about recognizing symmetry sharpens the data, so the claim is inherited."""
    structure = Structure(
        Cell(CUBIC, 1, BASIS_PRECISION),
        Sites([[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]], COORD_PRECISION),
        _species() + [Species(name="Cl", chemical_symbols=("Cl",), concentration=(1.0,))],
        ["Na", "Cl"],
    )
    recovered = recognize_asu(structure, setting=Spacegroup.standard(221))
    assert recovered.coordinate_precision == COORD_PRECISION
    assert recovered.cell.precision == BASIS_PRECISION
