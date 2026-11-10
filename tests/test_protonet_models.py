"""Protonettype and Protonetstructure values, their label notation, views and bare adapters."""

import pickle
from fractions import Fraction

import pytest

from httk.atomistic import (
    BareProtostructureView,
    BarePrototypeView,
    Formulatype,
    Protonetstructure,
    ProtonetstructureLabelString,
    ProtonetstructureView,
    Protonettype,
    ProtonettypeLabelString,
    ProtonettypeView,
    Species,
)
from httk.atomistic.models.bareprotostructure.label_string import BareProtostructureLabelString
from httk.atomistic.models.bareprototype.label_string import BarePrototypeLabelString
from httk.atomistic.models.prototype.notation import (
    parse_protonetstructure_label,
    parse_protonettype_label,
    try_parse_bare_protostructure,
    try_parse_bare_prototype,
)

BARE = "A4B2CD_oP32_34_4c_abc_ab_c"
NET = "1~8.10_2~8.10_3~9.10_4~9.10@3"
KS_NET = "1~5.6.7.7@4.8.10_2~5.6@4.7@4.7@2@556.8@1@556.10@1@556_3~5.6.7@1@556.9.10@4@455_4~5.7.7@3@445.9@4@454.10@4@455"
ROWS = [(i, j, 1, 0, 0, 0) for i, j in ((1, 8), (1, 10), (2, 8), (2, 10), (3, 9), (3, 10), (4, 9))] + [
    (4, 10, 3, 0, 0, 0)
]
LETTERS = ["c"] * 4 + ["a", "b", "c", "a", "b", "c"]
CLASSES = ["A"] * 4 + ["B"] * 3 + ["C"] * 2 + ["D"]
NAMES = ["S"] * 4 + ["K"] * 3 + ["Ag"] * 2 + ["Sb"]


def _sp(name: str) -> Species:
    return Species(name, (name,), (1,))


def _anon(net=ROWS) -> Protonettype:
    return Protonettype(34, list(zip(LETTERS, CLASSES)), net)


def _assigned() -> Protonetstructure:
    return Protonetstructure(34, [(letter, _sp(name)) for letter, name in zip(LETTERS, NAMES)], ROWS)


def test_k2agsbs4_labels_and_prefix_nesting() -> None:
    v, ps = _anon(), _assigned()
    assert str(v.label) == f"{BARE}__{NET}"
    assert str(ps.label) == f"{BARE}__{NET}:S-K-Ag-Sb"
    formula = str(Formulatype("A4B2CD"))
    bare = str(BarePrototypeView(v).label)
    assert bare == BARE and bare.startswith(formula) and str(v.label).startswith(bare + "__")
    assert str(ps.label).replace("__" + NET, "") == BARE + ":S-K-Ag-Sb"
    assert v.pearson_symbol == "oP32" and v.nsites_conventional == 32


@pytest.mark.parametrize(
    "text",
    [
        f"{BARE}__{NET}",
        f"{BARE}__{NET}:S-K-Ag-Sb",
        f"{BARE}__{KS_NET}",
        "AB_cF8_225_a_b__1~2",
        "AB_cF8_225_a_b__1~2:Cl-Na",
        "A_cF8_227_a__1~1@14@654",
        "AB_hP4_186_b_b__1~2.2@2:Zn-S",
        "A_cP1_221_a__",
        "AB_cF8_225_a_b__:Na-Cl",
        "A_oP8_47_A__",
    ],
)
def test_parse_render_round_trip(text: str) -> None:
    parse = parse_protonetstructure_label if ":" in text else parse_protonettype_label
    value = parse(text)
    assert str(value.label) == text
    assert parse(str(type(value)(value.spacegroup, value.occupations, value.net).label)) == value


def test_wurtzite_keeps_tied_class_order() -> None:
    value = parse_protonetstructure_label("AB_hP4_186_b_b__1~2.2@2:Zn-S")
    assert [o.species.name for o in value.occupations] == ["Zn", "S"]
    assert value.net == ((1, 2, 1, 0, 0, 0), (1, 2, 2, 0, 0, 0))


@pytest.mark.parametrize(
    "net",
    [
        "1~8@1",
        "1~8@3@555",
        "1-8",
        "1~8,10",
        "1~8.",
        "1~8@3@55",
        "01~8",
        "8~1",
        "1~8.8",
        "2~3_1~2",
        "1~2_1~3",
        "1~9@999",
        "0~1",
        "1~11",
        "1~1",
        "1~2__3~4",
        "1~2=3",
    ],
)
def test_parser_rejects_non_canonical_nets(net: str) -> None:
    with pytest.raises(ValueError):
        parse_protonettype_label(f"{BARE}__{net}")


def test_parser_rejects_other_forms() -> None:
    for text in (
        "AB2CD4_oP32_34_ab_abc_c_4c__1~2",  # pre-Formulatype class order
        f"{BARE}__{NET}:S-K-Ag-Sb",  # suffix on the anonymous level
        f"{BARE}:S-K-Ag-Sb__{NET}",  # suffix before the net
        BARE,
    ):
        with pytest.raises(ValueError):
            parse_protonettype_label(text)
    with pytest.raises(ValueError):
        parse_protonetstructure_label(f"{BARE}__{NET}")
    with pytest.raises(ValueError):
        parse_protonetstructure_label(f"{BARE}__{NET}:S-K-Ag-Xx")


@pytest.mark.parametrize(
    "row",
    [
        (1, 8, 1, 5, 0, 0),  # translation the notation cannot write
        (1, 8, 1, 0, 0, -6),
        (8, 1, 1, 0, 0, 0),  # i > j
        (0, 1, 1, 0, 0, 0),
        (1, 11, 1, 0, 0, 0),
        (1, 8, 5, 0, 0, 0),  # group 34 has four operations
        (1, 8, 0, 0, 0, 0),
        (1, 1, 1, 0, 0, 0),  # atom bonded to itself
        (1, 8, True, 0, 0, 0),
        (1, 8, 1, 0, 0),
    ],
)
def test_construction_rejects_bad_rows(row: tuple) -> None:
    with pytest.raises(ValueError):
        _anon([*ROWS, row])


def test_construction_rejects_duplicate_rows_and_bad_occupations() -> None:
    with pytest.raises(ValueError):
        _anon([*ROWS, ROWS[0]])
    bad_occupations = [
        ([("a", "A"), ("b", "A"), ("c", "A"), *[("c", "B")] * 4, ("a", "C"), ("b", "C"), ("c", "D")], "largest site"),
        (list(zip(LETTERS, ["A"] * 4 + ["C"] * 3 + ["B"] * 2 + ["D"])), "anonymous_symbol"),
        (list(zip(LETTERS, CLASSES))[:7] + [("c", "D"), ("a", "C"), ("b", "C")], "largest site"),
        (
            list(zip(LETTERS, CLASSES))[:4] + [("b", "B"), ("a", "B"), ("c", "B")] + list(zip(LETTERS, CLASSES))[7:],
            "sorted",
        ),
        ([("c", "A"), ("a", "B"), ("c", "A")], "contiguous"),
        ([], "non-empty"),
    ]
    for occupations, reason in bad_occupations:
        with pytest.raises(ValueError, match=reason):
            Protonettype(34, occupations)
    split = [("a", _sp("Zn")), ("b", _sp("S")), ("c", _sp("Zn"))]  # Zn split into two runs
    with pytest.raises(ValueError, match="contiguous"):
        Protonetstructure(34, split)
    with pytest.raises(ValueError, match="equal Species"):
        Protonetstructure(34, [("c", _sp("S")), ("c", Species("S", ("S",), (1,), mass=(32,)))])
    with pytest.raises(ValueError):
        Protonettype(34, [("q", "A")])  # letter absent from the group


def test_erasure_is_positional() -> None:
    ps = _assigned()
    assert ProtonettypeView(ps).label == str(ps.label).rsplit(":", 1)[0]
    assert ps.protonettype == _anon()
    tied = Protonetstructure(4, [("a", _sp("Zn")), ("a", _sp("S"))], [(1, 1, 2, 0, 0, 0), (1, 2, 1, 0, 0, 0)])
    assert str(tied.label) == "AB_mP4_4_a_a__1~1@2.2:Zn-S"
    assert ProtonettypeView(tied).label == "AB_mP4_4_a_a__1~1@2.2"
    assert ProtonettypeView(tied).label == str(tied.label).rsplit(":", 1)[0]


def test_views_and_bare_adapters() -> None:
    v, ps = _anon(), _assigned()
    view = ProtonettypeView(str(v.label))
    assert view == v and type(view.unview()) is Protonettype and view.unview() == v
    assert view.label == v.label
    sview = ProtonetstructureView(str(ps.label))
    assert type(sview.unview()) is Protonetstructure and sview.unview() == ps
    for item in (view, sview, ProtonettypeView(ps), ProtonettypeView(v), v.label, ps.label):
        restored = pickle.loads(pickle.dumps(item))
        assert type(restored) is type(item) and restored == item and repr(restored.unview()) == repr(item.unview())
    assert BarePrototypeView(v).label == BARE
    assert BarePrototypeView(view).label == BARE
    assert BarePrototypeView(ps).label == BARE
    assert str(v.anonymous_formula) == "A4B2CD"
    assert str(ps.anonymous_formula) == "A4B2CD"
    assert str(ps.formula) == str(ps.bare_protostructure.formula)
    assert BareProtostructureView(ps).label == BARE + ":S-K-Ag-Sb"
    assert BareProtostructureView(sview).label == BARE + ":S-K-Ag-Sb"
    assert v.bare_prototype == BarePrototypeView(BARE).unview()
    assert v.protonettype is v and ps.protonetstructure is ps


def test_label_string_backends_decline_each_other() -> None:
    anon, assigned = f"{BARE}__{NET}", f"{BARE}__{NET}:S-K-Ag-Sb"
    for text in (anon, assigned, "A_cP1_221_a__"):
        assert try_parse_bare_prototype(text) is None and try_parse_bare_protostructure(text) is None
        assert BarePrototypeLabelString._backend_adopt(text) is None
        assert BareProtostructureLabelString._backend_adopt(text) is None
    for text in (BARE, BARE + ":S-K-Ag-Sb"):
        assert ProtonettypeLabelString._backend_adopt(text) is None
        assert ProtonetstructureLabelString._backend_adopt(text) is None
    assert ProtonettypeLabelString._backend_adopt(assigned) is None
    assert ProtonetstructureLabelString._backend_adopt(anon) is None
    assert ProtonettypeLabelString._backend_adopt(anon) is not None
    assert ProtonetstructureLabelString._backend_adopt(assigned) is not None


def test_repr_equality_and_hash() -> None:
    namespace = {
        "Fraction": Fraction,
        "Protonettype": Protonettype,
        "Protonetstructure": Protonetstructure,
        "Species": Species,
    }
    for value in (_anon(), _assigned(), parse_protonettype_label("A_cP1_221_a__")):
        assert eval(repr(value), namespace) == value
        assert hash(eval(repr(value), namespace)) == hash(value)
    assert _anon() != _anon(ROWS[:-1])
    assert len({_anon(), _anon(), _anon(ROWS[:-1])}) == 2
    assert _anon(list(reversed(ROWS))) == _anon()
