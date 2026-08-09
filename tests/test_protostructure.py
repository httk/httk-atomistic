"""Tests for standard-setting, geometry-free protostructures."""

from fractions import Fraction
from types import SimpleNamespace

import pytest
from httk.core import FracVector

from httk.atomistic import (
    AnonymousFormulaView,
    AnonymousStructure,
    ASUStructure,
    Assembly,
    CartesianSiteMoments,
    ChemicalComposition,
    ChemicalFormulaView,
    CompositionView,
    FundamentalDomainStructure,
    PrototypeView,
    Protostructure,
    ProtostructureView,
    Spacegroup,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffOccupation,
    WyckoffSite,
)
from httk.atomistic.models.cell.numeric import NumericCell
from httk.atomistic.models.sites.numeric import NumericSites


CELL = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
EMPTY = FracVector(())


def _species() -> tuple[Species, Species]:
    return Species("Na", ("Na",), (1,), charges=(1,), labels=("sodium",)), Species(
        "Cl", ("Cl",), (1,), spins=(1,), labels=("chloride",)
    )


def _rocksalt_asu(*, domain: type[FundamentalDomainStructure] = ASUStructure) -> FundamentalDomainStructure:
    sodium, chlorine = _species()
    return domain(
        CELL,
        225,
        (WyckoffSite("a", EMPTY, "Na"), WyckoffSite("b", EMPTY, "Cl")),
        (sodium, chlorine),
    )


def test_construction_promotes_inputs_and_canonicalizes() -> None:
    first = Protostructure(225, [("a", "Na"), ("b", "Cl")])
    second = Protostructure(Spacegroup.standard(225), [("b", "Cl"), ("a", "Na")])
    assert first == second
    assert first.occupations[0].species.name == "Cl"
    assert first.multiplicities() == (4, 4)
    assert first.nsites_conventional == 8
    assert {first: "value"}[second] == "value"

    with pytest.raises(ValueError, match="standard setting"):
        Protostructure(Spacegroup.for_setting("15:c1"), [("e", "Na")])
    with pytest.raises(ValueError, match="non-empty"):
        Protostructure(225, [])
    with pytest.raises(ValueError, match="no Wyckoff letter"):
        Protostructure(225, [("zz", "Na")])


def test_same_name_requires_equal_species_and_unknown_symbols_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal Species"):
        Protostructure(
            225,
            [
                ("a", Species("Na", ("Na",), (1,))),
                ("b", Species("Na", ("Na",), (1,), charges=(1,))),
            ],
        )
    with pytest.raises(ValueError, match="unknown.*X"):
        WyckoffOccupation("a", Species("unknown", ("X",), (1,)))
    with pytest.raises(ValueError, match="unknown.*X"):
        WyckoffOccupation("a", Species("attached", ("Na",), (1,), attached=("X",), nattached=(1,)))


def test_disorder_and_vacancy_project_with_tabulated_multiplicity() -> None:
    mixed = Species("mixed", ("Fe", "Ni"), (Fraction(1, 2), Fraction(1, 2)))
    vacancy = Species("vacancy", ("vacancy",), (1,))
    value = Protostructure(221, [("a", mixed), ("b", vacancy)])
    assert CompositionView(value).amounts == (("Fe", Fraction(1, 2)), ("Ni", Fraction(1, 2)))
    assert ChemicalFormulaView(value) == "FeNi"
    assert AnonymousFormulaView(value) == "AB"


def test_exact_paths_preserve_species_and_match_structure_formula() -> None:
    asu = _rocksalt_asu()
    value = ProtostructureView(asu)
    assert value.unwrap() is asu
    assert value.occupations[0].species in asu.species
    assert value.formula == UnitcellStructureView(asu).formula
    assert value.anonymous_formula == UnitcellStructureView(asu).chemical_formula_anonymous
    assert (
        ProtostructureView(
            FundamentalDomainStructure(
                CELL,
                225,
                asu.wyckoff_sites,
                asu.species,
            )
        )
        == value
    )
    with pytest.raises(ValueError):
        ProtostructureView(asu, tolerance=1e-5)
    assert ProtostructureView(value) is value
    with pytest.raises(ValueError):
        ProtostructureView(value, tolerance=1e-5)


def test_transform_scaled_source_uses_standard_conventional_scale() -> None:
    transform = Spacegroup.for_setting("166:R").transform_from_standard
    bi = Species("Bi", ("Bi",), (1,))
    oxygen = Species("O", ("O",), (1,))
    asu = ASUStructure(
        CELL,
        166,
        (WyckoffSite("a", EMPTY, "Bi"), WyckoffSite("b", EMPTY, "O")),
        (bi, oxygen),
        transform=transform,
    )
    source = UnitcellStructureView(asu)
    value = ProtostructureView(asu)
    direct = Protostructure(166, [("a", bi), ("b", oxygen)])

    assert value == direct
    assert CompositionView(value).elements_ratios == source.elements_ratios
    assert CompositionView(value).amounts == (("Bi", Fraction(3)), ("O", Fraction(3)))
    assert CompositionView(source).amounts == (("Bi", Fraction(1)), ("O", Fraction(1)))
    assert value.formula == source.chemical_formula_reduced


def test_recognition_probe_re_raises_errors_from_a_matched_structure_backend() -> None:
    pytest.importorskip("numpy")
    raw = SimpleNamespace(
        cell=NumericCell(CELL),
        sites=NumericSites([[0, 0, 0]]),
        species=(object(),),
        species_at_sites=("invalid",),
    )
    with pytest.raises(TypeError, match="SpeciesBackend"):
        ProtostructureView(raw)


def test_recognition_path_and_rejections() -> None:
    pytest.importorskip("spglib")
    asu = _rocksalt_asu()
    assert ProtostructureView(UnitcellStructureView(asu)) == ProtostructureView(asu)
    with pytest.raises(TypeError):
        ProtostructureView(asu, kind="bogus")

    anonymous = AnonymousStructure(CELL, [[0, 0, 0]], species_at_sites=("A",))
    with pytest.raises(TypeError, match="dummy species"):
        ProtostructureView(anonymous)
    with pytest.raises(TypeError):
        UnitcellStructureView(Protostructure(225, [("a", "Na")]))
    with pytest.raises(TypeError):
        PrototypeView(Protostructure(225, [("a", "Na")]))

    structure = UnitcellStructure(
        CELL,
        [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]],
        _species(),
        ("Na", "Cl"),
        assemblies=(Assembly(((0,),), (1,)),),
    )
    with pytest.raises(ValueError, match="assemblies"):
        ProtostructureView(structure)

    chemical = UnitcellStructure(
        CELL,
        [[0, 0, 0]],
        (_species()[0],),
        ("Na",),
        chemical_composition=ChemicalComposition({"Na": 1}),
    )
    with pytest.raises(ValueError, match="chemical_composition"):
        ProtostructureView(chemical)

    moments = UnitcellStructure(
        CELL,
        [[0, 0, 0], [Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)]],
        _species(),
        ("Na", "Cl"),
        site_moments=CartesianSiteMoments([[1, 0, 0], [0, 1, 0]]),
    )
    with pytest.raises(ValueError, match="site_moments"):
        ProtostructureView(moments)
