"""repr policy: value-type ``__repr__`` renders a public-constructor call that round-trips."""

from fractions import Fraction

from httk.atomistic import (
    Cell,
    ChemicalFormula,
    Composition,
    Chromaformula,
    Species,
)
from httk.core.vectors import FracScalar, FracVector, SurdScalar, SurdVector

# eval(repr(x)) needs every class the repr can name in scope.
_NS = {
    "Species": Species,
    "Cell": Cell,
    "Composition": Composition,
    "ChemicalFormula": ChemicalFormula,
    "Chromaformula": Chromaformula,
    "FracVector": FracVector,
    "FracScalar": FracScalar,
    "SurdVector": SurdVector,
    "SurdScalar": SurdScalar,
    "Fraction": Fraction,
}


def _roundtrips(value: object) -> None:
    restored = eval(repr(value), dict(_NS))  # noqa: S307 - trusted repr, verifies eval(repr(x)) fidelity
    assert restored == value, f"roundtrip changed value: {value!r} -> {restored!r}"


def test_species_repr_roundtrips() -> None:
    _roundtrips(Species("Na", ["Na"], [1]))
    _roundtrips(Species("Fe", ["Fe"], [1], charges=[Fraction(3)], spins=[Fraction(5, 2)]))
    _roundtrips(Species("mix", ["Fe", "Ni"], [Fraction(1, 2), Fraction(1, 2)]))


def test_cell_repr_roundtrips_including_scale() -> None:
    # Regression: repr previously emitted the scaled basis AND scale, so eval re-applied scale (scale**2).
    scaled = Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]], scale=2)
    restored = eval(repr(scaled), dict(_NS))  # noqa: S307
    assert restored == scaled
    assert restored.basis == scaled.basis  # not scale**2 * unscaled
    _roundtrips(Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]], periodicity=[True, True, False]))


def test_composition_repr_roundtrips() -> None:
    _roundtrips(Composition({"Fe": 2, "O": 3}))


def test_formula_reprs_name_their_class() -> None:
    f = ChemicalFormula("Fe2O3")
    assert repr(f) == "ChemicalFormula('Fe2O3')"
    _roundtrips(f)
    a = Chromaformula("A3B2")
    assert repr(a) == "Chromaformula('A3B2')"
    _roundtrips(a)


def test_spacegroup_repr_is_builder_form_and_roundtrips() -> None:
    from httk.atomistic.symmetry.spacegroup import Spacegroup

    sg = Spacegroup.standard(225)
    assert repr(sg) == "Spacegroup.standard(225)"
    assert eval(repr(sg), {"Spacegroup": Spacegroup}) == sg  # noqa: S307


def test_affine_operation_repr_roundtrips_via_xyz_constructor() -> None:
    from httk.atomistic.symmetry.affine_operation import AffineOperation

    op = AffineOperation([[0, -1, 0], [1, 0, 0], [0, 0, 1]], ["1/2", "1/2", "0"])
    assert eval(repr(op), {"AffineOperation": AffineOperation}) == op  # noqa: S307
    assert AffineOperation("x,y,z") == AffineOperation.identity()


def test_structure_and_trajectory_repr_and_str_are_clean() -> None:
    from httk.atomistic import Trajectory, UnitcellStructure

    cell = Cell([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    structure = UnitcellStructure(
        cell,
        [[0, 0, 0], ["1/2", "1/2", "1/2"]],
        [Species("Na", ["Na"], [1]), Species("Cl", ["Cl"], [1])],
        ["Na", "Cl"],
    )
    # A small structure round-trips through its repr (needs the model names in scope).
    ns = dict(_NS)
    ns["UnitcellStructure"] = UnitcellStructure
    ns["Sites"] = __import__("httk.atomistic", fromlist=["Sites"]).Sites
    assert eval(repr(structure), ns) == structure  # noqa: S307
    # __str__ is a human summary, distinct from the constructor-form repr.
    assert str(structure).startswith("UnitcellStructure(") and "sites" in str(structure)
    assert "0x" not in str(structure)

    traj = Trajectory([structure, structure])
    assert repr(traj).startswith("Trajectory(") and "..." in repr(traj)
    assert "0x" not in repr(traj)
