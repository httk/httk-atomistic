"""Formulatype counts sites per species class; AnonymousFormula and OPTIMADE count composition."""

from fractions import Fraction
from functools import reduce
from math import gcd, lcm
from pathlib import Path
from typing import Any

import pytest
from httk.core import load

from httk.atomistic import (
    AnonymousFormula,
    AnonymousFormulaView,
    BareProtostructure,
    BareProtostructureView,
    BarePrototypeView,
    ChemicalFormula,
    Composition,
    CompositionView,
    Formulatype,
    FormulatypeView,
    Protonetstructure,
    Species,
    StructuretypeView,
    UnitcellStructure,
    UnitcellStructureView,
)
from httk.atomistic.composition import project_composition

FIXTURES = Path(__file__).parent / "fixtures"
# cod/1001844 and the first six disorder fixtures carry distinct species names for one chemistry.
NORMAL = (
    "disorder/16.cif",
    "disorder/66.cif",
    "disorder/168.cif",
    "disorder/184.cif",
    "disorder/209.cif",
    "disorder/217.cif",
    "disorder/52.cif",
    "cod/1008801.cif",
    "cod/4002853.cif",
    "cod/4002910.cif",
    "cod/4002923.cif",
    "cod/1001844.cif",
    "cod/1000008.cif",
    "cod/1509425.cif",
    "structreading/1.cif",
    "structreading/62.cif",
    "structreading/167.cif",
    "structreading/186.cif",
    "structreading/194.cif",
    "structreading/225.cif",
)
EXTENDED = tuple(
    name
    for name in sorted(
        str(path.relative_to(FIXTURES))
        for folder in ("disorder", "cod", "structreading")
        for path in (FIXTURES / folder).glob("*.cif")
    )
    if name not in NORMAL
)


def _profile(normal: tuple[Any, ...], extended: tuple[Any, ...]) -> list[Any]:
    return [*normal, *(pytest.param(value, marks=pytest.mark.extended) for value in extended)]


def _prefix(label: object) -> str:
    return str(label).split("_", 1)[0]


def _composition_anonymous(structure: Any) -> str:
    """Return the OPTIMADE anonymous formula computed directly from the projected composition."""
    amounts = sorted((amount for _, amount in CompositionView(structure).amounts), reverse=True)
    scale = reduce(lcm, (amount.denominator for amount in amounts))
    counts = [int(amount * scale) for amount in amounts]
    common = reduce(gcd, counts)
    return "".join(chr(ord("A") + i) + (str(c // common) if c != common else "") for i, c in enumerate(counts))


def _site_names(structure: Any) -> list[str]:
    if hasattr(structure, "wyckoff_sites"):
        return [site.species for site in structure.wyckoff_sites]
    return list(structure.species_at_sites)


@pytest.mark.parametrize("name", _profile(NORMAL, EXTENDED))
def test_formulatype_equals_every_level_anonymous_formula_and_label_prefix(name: str) -> None:
    structure = load(FIXTURES / name, repair=True)
    formulatype = str(FormulatypeView(structure))
    bare = BareProtostructureView(structure)
    erased = BarePrototypeView(bare)
    for level in (bare, erased):
        assert str(level.anonymous_formula) == _prefix(level.label) == formulatype
    assert str(FormulatypeView(bare)) == str(FormulatypeView(bare.unview())) == formulatype
    # The OPTIMADE field and AnonymousFormula stay composition-based.
    assert structure.chemical_formula_anonymous == _composition_anonymous(structure)
    assert str(AnonymousFormulaView(structure)) == structure.chemical_formula_anonymous

    try:
        merged = str(StructuretypeView(structure).anonymous_formula)
    except ValueError as exc:  # mixed, vacancy or partially occupied species are not anonymizable
        assert "anonymize" in str(exc)
        return
    recognized = BarePrototypeView(structure)
    assert merged == str(recognized.anonymous_formula) == _prefix(recognized.label)
    by_name = {species.name: species.chemical_symbols for species in structure.species}
    used = set(_site_names(structure))
    if len({by_name[n] for n in used}) == len(used):
        assert merged == formulatype
    else:  # the documented limit: recognition merges same-element names, the structure does not
        assert name == "cod/1001844.cif"


def test_recognition_merges_same_element_names_unlike_the_structure() -> None:
    structure = load(FIXTURES / "cod/1001844.cif", repair=True)
    assert str(FormulatypeView(structure)) == "A24B6CD"
    assert str(BarePrototypeView(BareProtostructureView(structure)).label) == "A24B6CD_aP32_1_24a_6a_a_a"
    assert str(StructuretypeView(structure).anonymous_formula) == "A12B3C"
    assert str(BarePrototypeView(structure).label) == "A12B3C_aP32_1_24a_6a_2a"


MIXED = Species("FeNi", ("Fe", "Ni"), (Fraction(1, 2), Fraction(1, 2)))
HALF = Species("Fe0.5", ("Fe", "vacancy"), (Fraction(1, 2), Fraction(1, 2)))
FE = Species("Fe", ("Fe",), (1,))
O = Species("O", ("O",), (1,))
FACES = [[0, Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 2), 0, Fraction(1, 2)], [Fraction(1, 2), Fraction(1, 2), 0]]


def _perovskite_like(corner: Species, face: Species) -> UnitcellStructure:
    return UnitcellStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 4]], [[0, 0, 0], *FACES], [corner, face], [corner.name, *[face.name] * 3]
    )


@pytest.mark.parametrize(
    ("corner", "face", "formulatype", "anonymous"),
    [(MIXED, O, "A3B", "A6BC"), (HALF, O, "A3B", "A6B"), (FE, HALF, "A3B", "A")],
    ids=["mixed-site", "half-occupied", "half-occupied-shared-element"],
)
def test_mixed_and_partial_sites_count_fully_as_one_class(
    corner: Species, face: Species, formulatype: str, anonymous: str
) -> None:
    structure = _perovskite_like(corner, face)
    bare = BareProtostructure(221, [("a", corner), ("c", face)])
    net = Protonetstructure(221, [("c", face), ("a", corner)], [])
    assert str(FormulatypeView(structure)) == formulatype
    for level in (bare, net, net.protonettype, BarePrototypeView(bare)):
        assert str(FormulatypeView(level)) == str(level.anonymous_formula) == _prefix(level.label) == formulatype
    assert str(AnonymousFormulaView(structure)) == str(AnonymousFormulaView(bare)) == anonymous
    assert structure.chemical_formula_anonymous == anonymous


def test_distinct_species_names_are_distinct_classes() -> None:
    other = Species("Fe2", ("Fe",), (1,))
    structure = UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]], [[0, 0, 0], [Fraction(1, 2)] * 3], [FE, other], ["Fe", "Fe2"]
    )
    assert str(FormulatypeView(structure)) == _prefix(BareProtostructureView(structure).label) == "AB"
    assert str(AnonymousFormulaView(structure)) == str(StructuretypeView(structure).anonymous_formula) == "A"


def test_formulatype_from_a_bare_formula_needs_integer_amounts() -> None:
    for source in ({"Al": 2, "O": 3}, "Al2O3", ChemicalFormula("Al2O3"), Composition({"Al": 2, "O": 3})):
        assert FormulatypeView(source) == "A3B2"
    assert type(FormulatypeView(AnonymousFormula("A3B2")).unview()) is Formulatype
    for fractional in ({"Fe": Fraction(1, 2), "O": 1}, Composition({"Fe": Fraction(1, 2), "O": 1})):
        with pytest.raises(ValueError, match="integer amounts"):
            FormulatypeView(fractional)
        assert AnonymousFormulaView(fractional) == "A2B"


@pytest.mark.parametrize("name", ["cod/1008801.cif", "disorder/184.cif"])
def test_lazy_datastream_structure_counts_like_the_loaded_one(name: str) -> None:
    path = FIXTURES / name
    loaded = load(path, repair=True)
    lazy = UnitcellStructureView(str(path))
    expected = _prefix(BareProtostructureView(loaded).label)
    assert str(FormulatypeView(lazy)) == str(FormulatypeView(loaded)) == expected
    backend = lazy._backend
    assert backend.multiplicities() == loaded.multiplicities()
    assert backend.nsites == loaded.nsites
    assert project_composition(backend) == project_composition(loaded)
    assert str(FormulatypeView(AnonymousFormulaView(lazy))) == expected


def test_kind_hints_cannot_turn_a_structural_formulatype_into_composition() -> None:
    structure = _perovskite_like(HALF, O)
    assert str(FormulatypeView(structure, kind="sites")) == "A3B"
    with pytest.raises(ValueError, match="AnonymousFormulaView"):
        FormulatypeView(structure, kind="structure")
    with pytest.raises(ValueError, match="AnonymousFormulaView"):
        FormulatypeView(BareProtostructure(221, [("a", HALF), ("c", O)]), kind="wyckoff")


def test_sibling_views_keep_their_own_meaning() -> None:
    half = _perovskite_like(HALF, O)
    fe_fe2 = UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        [[0, 0, 0], [Fraction(1, 2)] * 3],
        [FE, Species("Fe2", ("Fe",), (1,))],
        ["Fe", "Fe2"],
    )
    bare = BareProtostructure(221, [("a", HALF), ("c", O)])
    assert str(AnonymousFormulaView(FormulatypeView(half))) == "A6B"
    assert str(AnonymousFormulaView(FormulatypeView(bare))) == "A6B"
    assert str(FormulatypeView(AnonymousFormulaView(fe_fe2))) == "AB"
    assert str(FormulatypeView(AnonymousFormulaView(half))) == "A3B"
    assert str(FormulatypeView(AnonymousFormulaView(bare))) == "A3B"
    view = FormulatypeView(half)
    assert view.amounts == (("A", Fraction(3)), ("B", Fraction(1)))
    assert FormulatypeView(bare).amounts == (("A", Fraction(3)), ("B", Fraction(1)))
