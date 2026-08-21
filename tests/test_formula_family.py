import pickle
from fractions import Fraction

import pytest
from httk.core import coerce, coerce_view

from httk.atomistic import (
    AnonymousFormula,
    AnonymousFormulaView,
    ASUStructure,
    Cell,
    ChemicalFormula,
    ChemicalFormulaBackend,
    ChemicalFormulaView,
    Composition,
    CompositionView,
    FundamentalDomainStructure,
    Chromaformula,
    ChromaformulaView,
    Spacegroup,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
)
from httk.atomistic.composition import project_composition
from httk.atomistic.models.formula.chromaformula_string import ChromaformulaString
from httk.atomistic.models.formula.diagnostics import CompositionDiagnostic
from httk.atomistic.models.formula.formula_string import FormulaString
from httk.atomistic.models.formula.notation import parse_anonymous_formula, parse_reduced_formula
from httk.atomistic.models.formula.plain import PlainComposition
from httk.atomistic.models.formula.record import RecordComposition
from httk.atomistic.models.formula.structure import StructureComposition
from httk.atomistic.storage.records import _normalized_composition_record_from_result


class _MalformedReducedBackend(ChemicalFormulaBackend):
    def __init__(self) -> None:
        pass

    @property
    def amounts(self):
        return (("O", Fraction(1)), ("Al", Fraction(1)))


class _MalformedAnonymousBackend(ChemicalFormulaBackend):
    def __init__(self) -> None:
        pass

    @property
    def amounts(self):
        return (("B", Fraction(2)), ("A", Fraction(1)))

    @property
    def is_anonymous(self):
        return True


def _unitcell() -> UnitcellStructure:
    return UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]],
        [Species("Al", ("Al",), (1,)), Species("O", ("O",), (1,))],
        ["Al", "O"],
    )


def _asu(cls: type[FundamentalDomainStructure] = ASUStructure) -> FundamentalDomainStructure:
    return cls(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        Spacegroup.standard(225),
        (WyckoffSite("a", (), "Al"), WyckoffSite("b", (), "O")),
        (Species("Al", ("Al",), (1,)), Species("O", ("O",), (1,))),
    )


def test_value_and_view_construction_and_string_behavior() -> None:
    formula = ChemicalFormulaView("Al2O3")
    assert formula == "Al2O3"
    assert isinstance(formula, str)
    assert hash(formula) == hash("Al2O3")
    assert f"{formula}" == "Al2O3"
    assert CompositionView({"Al": 2, "O": 3}).amounts == (("Al", Fraction(2)), ("O", Fraction(3)))
    assert ChromaformulaView("A3B2") == "A3B2"
    with pytest.raises(TypeError):
        ChemicalFormulaView("not a formula")


def test_backend_kinds_and_round_trips() -> None:
    assert isinstance(ChemicalFormulaBackend._select_backend({"Al": 2}, kind="plain"), PlainComposition)
    assert isinstance(ChemicalFormulaBackend._select_backend("Al2O3", kind="formula"), FormulaString)
    assert isinstance(ChemicalFormulaBackend._select_backend("A3B2", kind="anonymous"), ChromaformulaString)
    record = _normalized_composition_record_from_result(Composition({"Al": 2, "O": 3}))
    assert isinstance(ChemicalFormulaBackend._select_backend(record), RecordComposition)
    structure = _unitcell()
    view = CompositionView(structure)
    assert isinstance(view._backend, StructureComposition)
    assert view.unwrap() is structure
    assert ChemicalFormulaView(view)._backend is view._backend
    assert type(ChemicalFormulaView(view).unview()) is ChemicalFormula
    assert type(view.unview()) is Composition
    value = ChemicalFormula("Al2O3")
    assert ChemicalFormulaView(value).unview() is value
    assert CompositionView(Composition({"Al": 2, "O": 3})).unview() is not None


@pytest.mark.parametrize("structure", [_unitcell(), _asu(FundamentalDomainStructure), _asu(ASUStructure)])
def test_structure_projection_includes_wyckoff_multiplicities(structure: FundamentalDomainStructure) -> None:
    view = CompositionView(structure)
    assert view == project_composition(structure)
    assert view.amounts == project_composition(structure).amounts


def test_record_and_structure_laziness() -> None:
    structure = _unitcell()
    view = CompositionView(structure)
    assert "amounts" not in view.__dict__
    assert view.amounts == project_composition(structure).amounts
    assert "amounts" in view.__dict__
    record = _normalized_composition_record_from_result(Composition({"Al": 2, "O": 3}))
    assert CompositionView(record) == Composition({"Al": 2, "O": 3})


def test_asu_with_cached_composition_pickles() -> None:
    source = _asu()
    structure = ASUStructure(
        Cell(source.cell.basis),
        source.spacegroup,
        source.wyckoff_sites,
        source.species,
    )
    _ = structure.composition.amounts

    restored = pickle.loads(pickle.dumps(structure))

    assert restored == structure
    assert restored.composition == structure.composition


def test_formula_directionality_and_validation() -> None:
    with pytest.raises(ValueError, match="anonymous"):
        ChemicalFormulaView(Chromaformula("A2B"))
    with pytest.raises(ValueError, match="anonymous"):
        CompositionView(Chromaformula("A2B"))
    assert ChromaformulaView(ChemicalFormula("Al2O3")) == "A3B2"
    assert ChromaformulaView(Composition({"O": 2, "Al": 2})) == "AB"
    incomplete = UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        [[0, 0, 0]],
        [Species("unknown", ("X",), (1,))],
        ["unknown"],
    )
    incomplete_view = CompositionView(incomplete)
    with pytest.raises(ValueError, match="incomplete"):
        ChemicalFormulaView(incomplete_view)
    with pytest.raises(ValueError, match="incomplete"):
        ChromaformulaView(incomplete_view)
    assert incomplete_view.complete is False
    with pytest.raises(ValueError, match="empty"):
        ChemicalFormulaView(Composition({}))
    with pytest.raises(ValueError, match="empty"):
        ChromaformulaView(Composition({}))


def test_eager_formula_views_preserve_backend_hubs() -> None:
    diagnostic = CompositionDiagnostic("test", "preserved")
    backend = Composition(
        {"Al": 4, "O": 6},
        uncertainties={"Al": Fraction(1, 10), "O": Fraction(1, 5)},
        exact=False,
        normalized=False,
        diagnostics=(diagnostic,),
    )
    reduced = ChemicalFormulaView(backend)
    assert str(reduced) == "Al2O3"
    assert reduced.amounts == backend.amounts
    assert reduced.uncertainties == backend.uncertainties
    assert reduced.exact is False
    assert reduced.normalized is False
    assert reduced.normalization_status == backend.normalization_status
    assert reduced.diagnostics == (diagnostic,)

    anonymous = ChromaformulaView(backend)
    assert str(anonymous) == "A3B2"
    assert anonymous.amounts == (("A", Fraction(6)), ("B", Fraction(4)))
    assert anonymous.uncertainties == (("A", Fraction(1, 5)), ("B", Fraction(1, 10)))
    assert anonymous.exact is False
    assert anonymous.normalized is False
    assert anonymous.normalization_status == backend.normalization_status
    assert anonymous.diagnostics == (diagnostic,)


def test_synthesized_formula_text_is_parser_validated() -> None:
    with pytest.raises(ValueError):
        ChemicalFormulaView(_MalformedReducedBackend())
    with pytest.raises(ValueError):
        ChromaformulaView(_MalformedAnonymousBackend())


def test_legacy_formula_aliases_are_canonical_identities() -> None:
    assert AnonymousFormula is Chromaformula
    assert AnonymousFormulaView is ChromaformulaView


@pytest.mark.parametrize("text", ["Al2O2", "H2", "OAl", "Al1O", "", "Aluminum"])
def test_reduced_parser_is_strict(text: str) -> None:
    with pytest.raises(ValueError):
        parse_reduced_formula(text)


@pytest.mark.parametrize("text", ["B2C", "A2B3", "A1B"])
def test_anonymous_parser_is_strict(text: str) -> None:
    with pytest.raises(ValueError):
        parse_anonymous_formula(text)


def test_anonymous_parser_accepts_canonical_forms() -> None:
    assert parse_anonymous_formula("A") == (("A", 1),)
    assert parse_anonymous_formula("AB") == (("A", 1), ("B", 1))
    assert parse_anonymous_formula("A2B2C") == (("A", 2), ("B", 2), ("C", 1))


def test_coercion_and_datastream_guard() -> None:
    structure = _unitcell()
    assert isinstance(coerce_view(structure, ChemicalFormulaView), ChemicalFormulaView)
    result = coerce(structure, ChemicalFormula)
    assert type(result) is ChemicalFormula
    incomplete = UnitcellStructure(
        [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
        [[0, 0, 0]],
        [Species("unknown", ("X",), (1,))],
        ["unknown"],
    )
    with pytest.raises(TypeError):
        coerce(incomplete, ChemicalFormula)
    with pytest.raises(TypeError):
        UnitcellStructureView(ChemicalFormulaView("Al2O3"))
