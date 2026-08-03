from fractions import Fraction as F

import pytest
from httk.core import FracVector

from httk.atomistic import (
    Assembly,
    WyckoffSite,
    ASUStructure,
    Cell,
    Spacegroup,
    Species,
    PlainSpeciesView,
    SpeciesView,
)
from httk.atomistic.composition import (
    ChemicalComposition,
    anonymous_symbol,
    derive_structure_features,
    project_composition,
    validate_assemblies,
)


class _Structure:
    def __init__(self, species, names, *, assemblies=None, chemical_composition=None):
        self.species = tuple(species)
        self.species_at_sites = tuple(names)
        self.assemblies = None if assemblies is None else tuple(assemblies)
        self.chemical_composition = chemical_composition


def test_species_exact_inputs_and_source_precision_roundtrip() -> None:
    species = Species("mixed", ("Ge", "Si"), (F(1, 3), "2/3"))
    assert species.concentration == (F(1, 3), F(2, 3))
    assert species.concentration_precision == (None, None)
    measured = Species("measured", ("Ge",), ("0.3333(7)",))
    assert measured.concentration == (F(3333, 10000),)
    assert measured.concentration_precision == (F(7, 10000),)
    explicit = Species("explicit", ("Ge",), (0.5,), concentration_precision=(F(1, 1000),))
    assert explicit.concentration == (F(1, 2),)
    assert explicit.concentration_precision == (F(1, 1000),)
    assert explicit == Species("explicit", ("Ge",), (F(1, 2),), concentration_precision=(F(1, 1000),))
    assert hash(explicit) == hash(Species("explicit", ("Ge",), (F(1, 2),), concentration_precision=(F(1, 1000),)))
    assert explicit != Species("explicit", ("Ge",), (F(1, 2),), concentration_precision=(F(1, 100),))


def test_species_class_and_primitive_views_keep_exactness_off_standard_json() -> None:
    raw = {
        "name": "half",
        "chemical_symbols": ["Ge"],
        "concentration": ["0.5000"],
        "_httk_concentration_precision": [F(7, 10000)],
    }
    class_view = SpeciesView(raw)
    assert class_view.concentration == (F(1, 2),)
    assert class_view.concentration_precision == (F(7, 10000),)
    private_primitive = PlainSpeciesView(raw)
    assert private_primitive["concentration"] == [0.5]
    assert private_primitive["_httk_concentration_precision"] == [0.0007]

    standard = PlainSpeciesView(Species("Ge", ("Ge",), (F(1, 2),)))
    assert standard["concentration"] == [0.5]
    assert "_httk_concentration_precision" not in standard


def test_species_validation_and_normalization_diagnostic() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Species("bad", (), ())
    with pytest.raises(ValueError, match="vacancy mass"):
        Species("bad", ("vacancy",), (1,), mass=(1,))
    assert Species("zero", ("C",), (1,), attached=("H",), nattached=(0,)).nattached == (0,)
    with pytest.raises(ValueError, match="non-negative integers"):
        Species("bad", ("C",), (1,), attached=("H",), nattached=(-1,))
    with pytest.raises(TypeError, match="name"):
        Species(1, ("C",), (1,))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="original_name"):
        Species("bad", ("C",), (1,), original_name=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mass values cannot be bool"):
        Species("bad", ("C",), (1,), mass=(True,))
    for value in (-1, 2, float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r"\[0, 1\]|finite"):
            Species("bad", ("Ge",), (value,))
    for mass in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="mass"):
            Species("bad", ("Ge",), (1,), mass=(mass,))
    with pytest.raises(ValueError, match="vacancy mass"):
        Species("bad", ("vacancy",), (1,), mass=(F(1),))
    with pytest.raises(ValueError, match="elements or 'X'"):
        Species("bad", ("C",), (1,), attached=("vacancy",), nattached=(1,))
    with pytest.raises(ValueError, match="same length"):
        Species("bad", ("C",), (1,), attached=("H", "X"), nattached=(1,))
    assert Species("valid", ("C",), (1,), attached=("X",), nattached=(1,)).attached == ("X",)
    inside = Species("inside", ("Ge", "Si"), ("0.4", "0.5"))
    assert inside.normalized
    assert inside.normalization_status == "within_precision"
    species = Species("bad", ("Ge", "Si"), ("0.4", "0.3"))
    assert not species.normalized
    assert species.normalization_diagnostic is not None
    assert species.concentration == (F(2, 5), F(3, 10))


def test_projection_formula_attachments_and_full_composition() -> None:
    mixed = Species("mixed", ("Ge", "Si", "vacancy"), (F(5, 8), F(3, 8), 0))
    result = project_composition(_Structure((mixed,), ("mixed",)))
    assert result.amounts == (("Ge", F(5, 8)), ("Si", F(3, 8)))
    assert result.elements_ratios == (F(5, 8), F(3, 8))
    assert result.chemical_formula_reduced == "Ge5Si3"
    assert result.chemical_formula_anonymous == "A5B3"

    attached = Species("CH3", ("C",), (1,), attached=("H",), nattached=(3,))
    full = ChemicalComposition({"C": 2, "H": 6}, mode="full")
    result = project_composition(_Structure((attached,), ("CH3",), chemical_composition=full))
    assert result.amounts == (("C", F(2)), ("H", F(6)))
    assert result.complete
    assert any(item.code == "full_composition_mismatch" for item in result.diagnostics)

    implicit = ChemicalComposition({"H": F(1, 2)}, mode="implicit")
    result = project_composition(_Structure((mixed,), ("mixed",), chemical_composition=implicit))
    assert result.amounts == (("Ge", F(5, 8)), ("H", F(1, 2)), ("Si", F(3, 8)))
    assert derive_structure_features(_Structure((mixed,), ("mixed",), chemical_composition=implicit)) == (
        "disorder",
        "implicit_atoms",
    )


def test_projection_ignores_unused_species_and_handles_empty_elemental_results() -> None:
    used = Species("used", ("Ge",), (1,))
    unused = Species("unused", ("Si",), (1,))
    result = project_composition(_Structure((used, unused), ("used",)))
    assert result.amounts == (("Ge", F(1)),)

    vacancy = project_composition(_Structure((Species("vac", ("vacancy",), (1,)),), ("vac",)))
    assert vacancy.elements == vacancy.elements_ratios == ()
    assert vacancy.chemical_formula_reduced is vacancy.chemical_formula_anonymous is None
    unknown = project_composition(_Structure((Species("unknown", ("X",), (1,)),), ("unknown",)))
    assert not unknown.complete
    assert unknown.elements == ()


def test_assembly_and_measured_formula_reconstruction_uses_central_values() -> None:
    ge = Species("Ge", ("Ge",), (1,))
    si = Species("Si", ("Si",), (1,))
    assembly = Assembly(((0,), (1,)), (F(5, 8), F(3, 8)))
    result = project_composition(_Structure((ge, si), ("Ge", "Si"), assemblies=(assembly,)))
    assert result.chemical_formula_reduced == "Ge5Si3"

    mixed = Species("mixed", ("Ge", "Si"), ("0.3333", "0.6666"))
    assert project_composition(_Structure((mixed,), ("mixed",))).chemical_formula_reduced == "GeSi2"

    impossible = Species(
        "impossible",
        ("Ge", "Si"),
        (F(1, 1001), F(1000, 1001)),
        concentration_precision=(F(1, 10**12), F(1, 10**12)),
    )
    result = project_composition(_Structure((impossible,), ("impossible",)))
    assert result.chemical_formula_reduced == "GeSi1000"
    assert result.chemical_formula_anonymous == "A1000B"


def test_assembly_validation_and_asu_multiplicity_projection() -> None:
    exact = Assembly(((0,), (1,)), (F(5, 8), F(3, 8)), (None, None))
    assert exact.normalized
    outside = Assembly(((0,), (1,)), ("0.4", "0.3"))
    assert not outside.normalized
    assert outside.normalization_diagnostic is not None
    with pytest.raises(ValueError, match="more than one Assembly"):
        validate_assemblies((exact, Assembly(((1,),), (1,))))
    with pytest.raises(ValueError, match="outside the structure"):
        validate_assemblies((Assembly(((2,),), (1,)),), nsites=2)

    asu = ASUStructure(
        Cell([[5, 0, 0], [0, 5, 0], [0, 0, 5]]),
        Spacegroup.standard(225),
        (WyckoffSite("a", FracVector.create(()), "Ge"),),
        (Species("Ge", ("Ge",), (1,)),),
    )
    assert asu.multiplicities() == (4,)
    assert project_composition(asu).amounts == (("Ge", F(4)),)


def test_partial_single_symbol_is_disorder_and_attached_x_is_incomplete() -> None:
    partial = Species("half_Ge", ("Ge",), (F(1, 2),))
    assert derive_structure_features(_Structure((partial,), ("half_Ge",))) == ("disorder",)
    attached_unknown = Species("C_X", ("C",), (1,), attached=("X",), nattached=(1,))
    result = project_composition(_Structure((attached_unknown,), ("C_X",)))
    assert result.amounts == (("C", F(1)),)
    assert not result.complete


def test_unbounded_anonymous_symbols() -> None:
    assert [anonymous_symbol(index) for index in (0, 25, 26, 51, 52, 701, 702)] == [
        "A",
        "Z",
        "Aa",
        "Za",
        "Ab",
        "Zz",
        "Aaa",
    ]
    assert "".join(anonymous_symbol(index) for index in range(702, 705)) == "AaaBaaCaa"
