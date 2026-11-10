"""Species classes in labels and canonical values follow Formulatype order."""

from collections.abc import Callable
from fractions import Fraction as F
from typing import Any

import pytest
from httk.core import FracVector
from test_anonymous_canonicalization import _low_symmetry, _nacl

from httk.atomistic import (
    ASUStructure,
    BareProtostructure,
    BareProtostructureView,
    BarePrototype,
    BarePrototypeView,
    Cell,
    Protostructure,
    Prototype,
    PrototypeView,
    Species,
    WyckoffSite,
    canonical_bare_protostructure,
    canonical_bare_prototype,
    canonical_protostructure,
    canonical_prototype,
)
from httk.atomistic.models.prototype.notation import (
    parse_bare_protostructure_label,
    parse_bare_prototype_label,
)


def _species(name: str) -> Species:
    return Species(name, (name,), (1,))


# (space group, (letter, species) occupations, canonical label, plain label when it differs).
TABLE = {
    "K2AgSbS4": (
        34,
        [("c", "S")] * 4 + [("a", "K"), ("b", "K"), ("c", "K"), ("a", "Ag"), ("b", "Ag"), ("c", "Sb")],
        "A4B2CD_oP32_34_4c_abc_ab_c:S-K-Ag-Sb",
        None,
    ),
    "calcite": (167, [("a", "Ca"), ("b", "C"), ("e", "O")], "A3BC_hR10_167_e_a_b:O-Ca-C", None),
    # The normalizer swaps a and b; the classes then tie on count and letters, so the
    # species name decides and the canonical value puts Cl on a.
    "NaCl": (225, [("a", "Na"), ("b", "Cl")], "AB_cF8_225_a_b:Cl-Na", "AB_cF8_225_a_b:Na-Cl"),
    "ZrO2": (29, [("a", "Zr"), ("a", "O"), ("a", "O")], "A2B_oP12_29_2a_a:O-Zr", None),
    "FeS2": (29, [("a", "Fe"), ("a", "S"), ("a", "S")], "A2B_oP12_29_2a_a:S-Fe", None),
    "wurtzite": (186, [("b", "Zn"), ("b", "S")], "AB_hP4_186_b_b:S-Zn", None),
    "rutile": (136, [("a", "Ti"), ("f", "O")], "A2B_tP6_136_f_a:O-Ti", None),
    "perovskite": (221, [("a", "Ca"), ("b", "Ti"), ("c", "O")], "A3BC_cP5_221_c_a_b:O-Ca-Ti", None),
    # The origin shift (1/2,1/2,1/2) of 227 maps a<->b and c<->d. Al (16 sites) is the
    # second class, and c < d at that position selects Al on c, hence Mg on b.
    "spinel": (
        227,
        [("a", "Mg"), ("d", "Al"), ("e", "O")],
        "A4B2C_cF56_227_e_c_b:O-Al-Mg",
        "A4B2C_cF56_227_e_d_a:O-Al-Mg",
    ),
}


def _bare(name: str) -> BareProtostructure:
    it_number, occupations, _, _ = TABLE[name]
    return BareProtostructure(it_number, [(letter, _species(species)) for letter, species in occupations])


def _rutile_asu() -> ASUStructure:
    return ASUStructure(
        Cell(((F(23, 5), 0, 0), (0, F(23, 5), 0), (0, 0, 3))),
        136,
        (WyckoffSite("a", FracVector(()), "Ti"), WyckoffSite("f", FracVector((F(3, 10),)), "O")),
        (_species("Ti"), _species("O")),
    )


def _assigned_values() -> list[Any]:
    bare = [_bare(name) for name in TABLE]
    canonical = [canonical_bare_protostructure(value) for value in bare]
    refined = [Protostructure(representative=_rutile_asu()), Protostructure(representative=_nacl())]
    return bare + canonical + refined + [canonical_protostructure(value) for value in refined]


def _anonymous_values() -> list[Any]:
    values: list[Any] = [BarePrototypeView(value).unview() for value in _assigned_values()]
    refined = [PrototypeView(Protostructure(representative=build())).unview() for build in (_rutile_asu, _low_symmetry)]
    return values + refined + [canonical_prototype(value) for value in refined]


@pytest.mark.parametrize("name", list(TABLE))
def test_table_labels(name: str) -> None:
    _, _, canonical, plain = TABLE[name]
    value = _bare(name)
    assert str(value.label) == (plain or canonical)
    assert str(canonical_bare_protostructure(value).label) == canonical


def test_calcite_bare_prototype_label_and_aflow_label() -> None:
    assert str(BarePrototype(167, [("a", "A"), ("b", "B"), ("e", "C")]).label) == "A3BC_hR10_167_e_a_b"
    assert _bare("calcite").aflow_label == "ABC3_hR10_167_b_a_e:C-Ca-O"


def test_label_prefix_is_the_anonymous_formula() -> None:
    for value in _assigned_values():
        assert str(value.label).split("_")[0] == str(value.anonymous_formula)
    for value in _anonymous_values():
        assert str(value.label).split("_")[0] == str(value.anonymous_formula)


def test_assigned_label_is_the_erased_label_plus_names() -> None:
    for value in _assigned_values():
        text = str(value.label)
        prefix, _, suffix = text.partition(":")
        assert prefix == str(BarePrototypeView(value).label)
        names = suffix.split("-")
        assert sorted(names) == sorted({o.species.name for o in BareProtostructureView(value).occupations})


def test_parse_render_round_trips_and_old_order_is_rejected() -> None:
    for value in _assigned_values():
        bare = BareProtostructureView(value).unview()
        text = str(bare.label)
        assert parse_bare_protostructure_label(text) == bare
        assert str(parse_bare_protostructure_label(text).label) == text
    for value in _anonymous_values():
        bare = BarePrototypeView(value).unview()
        text = str(bare.label)
        assert parse_bare_prototype_label(text) == bare
        assert str(parse_bare_prototype_label(text).label) == text
    for old in ("ABC3_hR10_167_a_b_e", "AB2_hP12_194_ab_2f"):
        with pytest.raises(ValueError):
            parse_bare_prototype_label(old)
    with pytest.raises(ValueError):
        parse_bare_protostructure_label("ABC3_hR10_167_a_b_e:Ca-C-O")


def _refined(value: Any) -> Any:
    """Return a refined value, wrapping a bare one as a discriminated base-only classification."""
    if isinstance(value, BareProtostructure):
        return Protostructure(value.spacegroup, value.occupations, discriminator="001")
    if isinstance(value, BarePrototype):
        return Prototype(value.spacegroup, value.occupations, discriminator="001")
    return value


def test_canonicalisers_are_idempotent() -> None:
    for value in _assigned_values():
        once = canonical_bare_protostructure(value)
        assert canonical_bare_protostructure(once) == once
        refined = canonical_protostructure(_refined(value))
        assert canonical_protostructure(refined) == refined
    for value in _anonymous_values():
        anonymous = canonical_bare_prototype(value)
        assert canonical_bare_prototype(anonymous) == anonymous
        refined_anonymous = canonical_prototype(_refined(value))
        assert canonical_prototype(refined_anonymous) == refined_anonymous


@pytest.mark.parametrize("build", (_rutile_asu, _low_symmetry), ids=("rutile", "low-symmetry"))
def test_refined_representative_names_classes_in_label_order(build: Callable[[], ASUStructure]) -> None:
    value = canonical_prototype(PrototypeView(Protostructure(representative=build())))
    assert value.representative is not None
    # The occupations carry the label's class order; the representative must name classes alike.
    for label in {o.label for o in value.occupations}:
        letters = sorted(site.wyckoff for site in value.representative.wyckoff_sites if site.species == label)
        assert letters == sorted(o.wyckoff for o in value.occupations if o.label == label)
