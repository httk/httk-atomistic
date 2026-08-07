"""Tests for standard-setting prototypes."""

from fractions import Fraction

import pytest
from httk.core import FracVector

from httk.atomistic import (
    AnonymousFormulaView,
    AnonymousFormula,
    AnonymousStructure,
    AnonymousStructureView,
    ASUStructure,
    ChemicalFormulaView,
    CompositionView,
    Prototype,
    PrototypeView,
    Spacegroup,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
)

CELL = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
EMPTY = FracVector.create(())


def _rocksalt_asu() -> ASUStructure:
    return ASUStructure(
        CELL,
        225,
        (WyckoffSite("a", EMPTY, "Na"), WyckoffSite("b", EMPTY, "Cl")),
        (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
    )


def test_prototype_standard_setting_and_canonical_site_order() -> None:
    species = (Species("A", ("X",), (1,), labels=("A",)), Species("B", ("X",), (1,), labels=("B",)))
    first = Prototype(CELL, 225, (WyckoffSite("b", EMPTY, "B"), WyckoffSite("a", EMPTY, "A")), species)
    second = Prototype(CELL, 225, tuple(reversed(first.wyckoff_sites)), species)
    assert first == second
    with pytest.raises(ValueError, match="standard setting"):
        Prototype(
            CELL, Spacegroup.for_setting("15:c1"), (WyckoffSite("e", FracVector.create([1, 3]), "A"),), (species[0],)
        )


def test_prototype_rejects_representatives_moments_and_bad_free_count() -> None:
    with pytest.raises(ValueError, match="representative"):
        Prototype(
            CELL, 225, (WyckoffSite("a", EMPTY, "A", representative=FracVector.create([0, 0, 0])),), (dummy("A"),)
        )
    with pytest.raises(ValueError, match="free parameter"):
        Prototype(CELL, 225, (WyckoffSite("a", FracVector.create([1]), "A"),), (dummy("A"),))


def dummy(label: str) -> Species:
    return Species(label, ("X",), (1,), labels=(label,))


def test_exact_asu_path_and_expansion() -> None:
    asu = _rocksalt_asu()
    prototype = PrototypeView(asu)
    assert PrototypeView(asu, kind="structure") == prototype
    with pytest.raises(TypeError):
        PrototypeView(asu, kind="bogus")
    assert prototype.spacegroup.is_standard_setting
    assert prototype.multiplicities() == (4, 4)
    assert len(AnonymousStructureView(prototype).sites) == prototype.nsites_conventional
    assert prototype.anonymous_formula == "AB"
    assert isinstance(AnonymousFormulaView(prototype), str)
    with pytest.raises(ValueError):
        PrototypeView(asu, tolerance=1e-5)
    with pytest.raises(ValueError):
        CompositionView(prototype)
    with pytest.raises(ValueError):
        ChemicalFormulaView(prototype)
    with pytest.raises(TypeError):
        ChemicalFormulaView(prototype, kind="bogus")


def test_prototype_view_rewrap_rejects_arguments() -> None:
    view = PrototypeView(_rocksalt_asu())
    with pytest.raises(ValueError):
        PrototypeView(view, tolerance=0.123)
    with pytest.raises(ValueError):
        PrototypeView(view, setting="bogus")
    with pytest.raises(ValueError):
        PrototypeView(view, kind="bogus")


def test_anonymous_prototype_source_uses_original_labels() -> None:
    anonymous = AnonymousStructure(
        CELL, [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]], species_at_sites=("A", "B")
    )
    pytest.importorskip("spglib")
    prototype = PrototypeView(anonymous)
    assert prototype.anonymous_formula == AnonymousStructureView(anonymous).anonymous_formula


def test_formula_views_reduce_non_coprime_amounts_and_remain_parseable() -> None:
    asu = _rocksalt_asu()
    prototype = PrototypeView(asu)
    assert AnonymousFormula(str(prototype.anonymous_formula)) == "AB"

    real_structure = UnitcellStructureView(asu)
    assert AnonymousFormula(str(AnonymousFormulaView(real_structure))) == "AB"
    assert ChemicalFormulaView(real_structure) == "ClNa"


def test_anonymous_formula_cross_consistency_for_non_coprime_counts() -> None:
    asu = _rocksalt_asu()
    assert AnonymousStructureView(asu).anonymous_formula == UnitcellStructureView(asu).chemical_formula_anonymous

    two_to_two = UnitcellStructureView(
        UnitcellStructure(
            CELL,
            [
                [0, 0, 0],
                [0, 0, 0],
                [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
                [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)],
            ],
            (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
            ("Na", "Na", "Cl", "Cl"),
        )
    )
    assert AnonymousStructureView(two_to_two).anonymous_formula == two_to_two.chemical_formula_anonymous


def test_recognition_path_is_spglib_gated() -> None:
    pytest.importorskip("spglib")
    asu = _rocksalt_asu()
    unitcell = UnitcellStructureView(asu)
    assert PrototypeView(unitcell) == PrototypeView(asu)
