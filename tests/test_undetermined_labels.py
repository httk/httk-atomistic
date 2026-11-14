"""Undetermined labels: ``X`` for a part the canonicalisers could not determine within their budget."""

import dataclasses
import time
from pathlib import Path
from typing import Any

import httk.core
import pytest
from _protonet_support import generic_cutoffs, load_structure, raw_rows
from httk.core.storage import content_id
from test_formulatype_sites import NORMAL as LABEL_FIXTURES
from test_protonet_canonical_api import LABELS, _asu, _bonds, _value
from test_protonet_models import _respelled
from test_protonet_molecular import MOLECULAR
from test_protonet_records import CLASSES, LETTERS, ROWS

import httk.atomistic.symmetry.canonical_classification as classification
from httk.atomistic import (
    BareProtostructure,
    BareProtostructureView,
    BarePrototype,
    BarePrototypeView,
    Protonetstructure,
    ProtonetstructureView,
    Protonettype,
    ProtonettypeView,
    Protostructure,
    ProtostructureView,
    Prototype,
    PrototypeView,
    Species,
    UnitcellStructureView,
    canonicalize,
)
from httk.atomistic.models.bareprotostructure.label import BareProtostructureLabel
from httk.atomistic.models.bareprototype.label import BarePrototypeLabel
from httk.atomistic.models.protonetstructure.label import ProtonetstructureLabel
from httk.atomistic.models.protonettype.label import ProtonettypeLabel
from httk.atomistic.models.protostructure.label import ProtostructureLabel
from httk.atomistic.models.prototype.label import PrototypeLabel
from httk.atomistic.models.prototype.notation import (
    _marks_undetermined,
    parse_bare_protostructure_label,
    parse_bare_prototype_label,
    parse_protonetstructure_label,
    parse_protonettype_label,
    try_parse_bare_protostructure,
    try_parse_bare_prototype,
    try_parse_protonetstructure,
    try_parse_protonettype,
)
from httk.atomistic.storage.records import (
    _bare_protostructure_from_record,
    _bare_protostructure_record_from_value,
    _bare_prototype_from_record,
    _bare_prototype_record_from_value,
    _protonetstructure_from_record,
    _protonetstructure_record_from_value,
    _protonettype_from_record,
    _protonettype_record_from_value,
    _protostructure_from_record,
    _protostructure_record_from_value,
    _prototype_from_record,
    _prototype_record_from_value,
)
from httk.atomistic.symmetry import protonet
from httk.atomistic.symmetry.limits import CanonicalizationLimitError, _budget_scope

FIXTURES = Path(__file__).parent / "fixtures"
K2AGSBS4 = FIXTURES / "cod" / "1509425.cif"
BARE = "A4B2CD_oP32_34_4c_abc_ab_c"
NAMES = ":S-K-Ag-Sb"
NET = "1~8.10_2~8.10_3~9.10_4~9.10@3"

# the stage forced over the limit, the entry patched for it, and the labels it gives
STAGES = {
    "net": ("_protonetstructure_net", f"{BARE}__X{NAMES}", f"{BARE}__X"),
    "wyckoff": ("_canonical_bare", f"A4B2CD_oP32_34_X__X{NAMES}", "A4B2CD_oP32_34_X__X"),
}
BARE_STAGES = {
    "wyckoff": ("_canonical_bare", f"A4B2CD_oP32_34_X{NAMES}", "A4B2CD_oP32_34_X"),
    "symmetry": ("_converted", f"A4B2CD_X{NAMES}", "A4B2CD_X"),
}
PARSERS = {
    BarePrototype: parse_bare_prototype_label,
    BareProtostructure: parse_bare_protostructure_label,
    Protonettype: parse_protonettype_label,
    Protonetstructure: parse_protonetstructure_label,
}


def _over_limit(*args: Any, **kwargs: Any) -> Any:
    raise CanonicalizationLimitError("forced over the limit")


def _structure() -> Any:
    return UnitcellStructureView(httk.core.load(str(K2AGSBS4))).unview()


def _refined() -> Protostructure:
    """K2AgSbS4 with an exact canonical representative (the CIF's ASU retains imprecise points)."""
    return classification.canonical_protostructure(Protostructure(representative=httk.core.load(str(K2AGSBS4))))


def _assert_round_trips(value: Any) -> None:
    text = str(value.label)
    assert not value.determined
    assert eval(repr(value)) == value and hash(eval(repr(value))) == hash(value)
    assert type(value).undetermined(text) == value
    if type(value) in PARSERS:
        assert PARSERS[type(value)](text) == value


# ---------------------------------------------------------------------------------------------
# each stage forced over the limit, as in the table of the note
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("stage", list(STAGES))
def test_protonet_stage_over_the_limit_gives_the_table_labels(stage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    entry, assigned, anonymous = STAGES[stage]
    value = _value("k2agsbs4-frame")
    monkeypatch.setattr(classification, entry, _over_limit)
    if stage == "net":
        monkeypatch.setattr(classification, "_protonettype_net", _over_limit)
    results = [classification.canonical_protonetstructure(value), classification.canonical_protonettype(value)]
    assert [str(result.label) for result in results] == [assigned, anonymous]
    for result in results:
        _assert_round_trips(result)
    with pytest.raises(CanonicalizationLimitError):
        classification.canonical_protonetstructure(value, on_limit="raise")
    with pytest.raises(CanonicalizationLimitError):
        classification.canonical_protonettype(value, on_limit="raise")


@pytest.mark.parametrize("stage", list(BARE_STAGES))
def test_bare_stage_over_the_limit_gives_the_table_labels(stage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    entry, assigned, anonymous = BARE_STAGES[stage]
    structure = _structure()
    monkeypatch.setattr(classification, entry, _over_limit)
    results = [
        classification.canonical_bare_protostructure(structure),
        classification.canonical_bare_prototype(structure),
    ]
    assert [str(result.label) for result in results] == [assigned, anonymous]
    for result in results:
        _assert_round_trips(result)
    with pytest.raises(CanonicalizationLimitError):
        classification.canonical_bare_protostructure(structure, on_limit="raise")
    with pytest.raises(CanonicalizationLimitError):
        classification.canonical_bare_prototype(structure, on_limit="raise")
    # the protonet functions take no structure: the protonet form of this row is the undetermined value (U2)
    net_text = assigned.replace(":", "__X:")
    _assert_round_trips(Protonetstructure.undetermined(net_text))
    assert str(Protonetstructure.undetermined(net_text).bare_protostructure.label) == assigned
    assert str(Protonettype.undetermined(net_text.partition(":")[0]).bare_prototype.label) == anonymous


def test_refined_stages_over_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    refined = _refined()
    anonymous = PrototypeView(refined).unview()
    with monkeypatch.context() as patch:
        # the geometry is not determined: the determined bare label is kept, without representative (U6)
        patch.setattr(classification, "_canonical_protostructure_asu", _over_limit)
        value, erased = classification.canonical_protostructure(refined), classification.canonical_prototype(anonymous)
        assert (str(value.label), str(erased.label)) == (BARE + NAMES, BARE)
        assert value.representative is None and value.discriminator is None
        assert value != Protostructure(value.spacegroup, value.occupations, discriminator="d")
        assert [o.wyckoff for o in value.occupations] == [o.wyckoff for o in refined.occupations]
        _assert_round_trips(value)
        _assert_round_trips(erased)
        with pytest.raises(CanonicalizationLimitError):
            classification.canonical_protostructure(refined, on_limit="raise")
    monkeypatch.setattr(classification, "_canonical_bare", _over_limit)
    value = classification.canonical_protostructure(refined)
    assert str(value.label) == f"A4B2CD_oP32_34_X{NAMES}"
    assert str(classification.canonical_prototype(anonymous).label) == "A4B2CD_oP32_34_X"
    with pytest.raises(ValueError, match="determined"):
        _ = value.occupations


def test_wyckoff_timeout_keeps_pearson_symbol_and_number(monkeypatch: pytest.MonkeyPatch) -> None:
    determined = str(classification.canonical_bare_protostructure(_structure()).label)
    monkeypatch.setattr(classification, "_canonical_bare", _over_limit)
    value = classification.canonical_bare_protostructure(_structure())
    assert str(value.label).split("_")[:3] == determined.split("_")[:3]
    assert (value.spacegroup.it_number, value.nsites_conventional) == (34, 32)
    assert classification.canonical_bare_prototype(_structure()).pearson_symbol == "oP32"
    assert str(value.anonymous_formula) == "A4B2CD"


def test_wyckoff_timeout_without_chirality_names_the_lower_partner(monkeypatch: pytest.MonkeyPatch) -> None:
    carbon = Species("C", ("C",), (1,))
    monkeypatch.setattr(classification, "_canonical_bare", _over_limit)
    value = BareProtostructure(78, [("a", carbon)])  # P4_3, the higher partner of P4_1 (76)
    assert str(classification.canonical_bare_protostructure(value).label) == "A_tP4_78_X:C"
    assert str(classification.canonical_bare_protostructure(value, preserve_chirality=False).label) == "A_tP4_76_X:C"


def test_levels_nest_as_prefixes() -> None:
    for text in (f"A4B2CD_oP32_34_X__X{NAMES}", f"A4B2CD_X__X{NAMES}", f"{BARE}__X{NAMES}"):
        value = Protonetstructure.undetermined(text)
        bare = text.replace("__X", "")
        assert str(value.bare_protostructure.label) == bare and str(value.anonymous_formula) == "A4B2CD"
        assert str(value.protonettype.label) == text.partition(":")[0]
        assert value.bare_protostructure.determined == (bare == BARE + NAMES)


def test_canonicalizers_return_an_undetermined_input_unchanged() -> None:
    for function, value in (
        (classification.canonical_bare_protostructure, BareProtostructure.undetermined(f"A4B2CD_X{NAMES}")),
        (classification.canonical_bare_prototype, BarePrototype.undetermined("A4B2CD_X")),
        (classification.canonical_protostructure, Protostructure.undetermined(BARE + NAMES)),
        (classification.canonical_prototype, Prototype.undetermined("A4B2CD_X")),
        (classification.canonical_protonetstructure, Protonetstructure.undetermined(f"{BARE}__X{NAMES}")),
        (classification.canonical_protonettype, Protonettype.undetermined("A4B2CD_X__X")),
    ):
        assert function(value) is value
        if type(value) in PARSERS:  # the label-string families
            assert function(str(value.label)) == value


@pytest.mark.parametrize(
    ("function", "source", "expected"),
    [
        (classification.canonical_protonettype, f"A4B2CD_oP32_34_X__X{NAMES}", "A4B2CD_oP32_34_X__X"),
        (classification.canonical_bare_protostructure, f"A4B2CD_oP32_34_X__X{NAMES}", f"A4B2CD_oP32_34_X{NAMES}"),
        (classification.canonical_bare_prototype, f"A4B2CD_X__X{NAMES}", "A4B2CD_X"),
        (classification.canonical_bare_prototype, f"{BARE}__X{NAMES}", BARE),  # the bare part is determined
        (classification.canonical_bare_protostructure, f"{BARE}__X{NAMES}", BARE + NAMES),
    ],
)
def test_broader_levels_of_an_undetermined_protonetstructure_nest(function: Any, source: str, expected: str) -> None:
    assert str(function(Protonetstructure.undetermined(source)).label) == expected


def test_broader_levels_of_undetermined_bare_and_refined_values() -> None:
    assert (
        str(classification.canonical_bare_prototype(BareProtostructure.undetermined(f"A4B2CD_X{NAMES}")).label)
        == "A4B2CD_X"
    )
    assert classification.canonical_prototype(Protostructure.undetermined(BARE + NAMES)) == Prototype.undetermined(BARE)
    assert classification.canonical_bare_prototype(
        Protonettype.undetermined("A4B2CD_X__X")
    ) == BarePrototype.undetermined("A4B2CD_X")
    assert classification.canonical_bare_prototype(Prototype.undetermined(BARE)).determined


@pytest.mark.parametrize(
    ("function", "family", "text", "kept", "flipped"),
    [
        # a Wyckoff-level label maps its number as the determined label does; the Pearson symbol stays
        (classification.canonical_bare_protostructure, BareProtostructure, "A_tP4_78_X:C", None, "A_tP4_76_X:C"),
        (classification.canonical_protonettype, Protonettype, "A_tP4_78_X__X", None, "A_tP4_76_X__X"),
        # a determined bare part is canonicalized and keeps its __X
        (classification.canonical_protonetstructure, Protonetstructure, "A_tP4_78_a__X:C", None, "A_tP4_76_a__X:C"),
        (
            classification.canonical_protonetstructure,
            Protonetstructure,
            "AB_cF8_225_a_b__X:Na-Cl",
            "AB_cF8_225_a_b__X:Cl-Na",
            "AB_cF8_225_a_b__X:Cl-Na",
        ),
        (classification.canonical_protostructure, Protostructure, "A_tP4_78_a:C", None, "A_tP4_76_a:C"),  # U6
        (classification.canonical_prototype, Prototype, "A_tP4_78_X", None, "A_tP4_76_X"),
        # without symmetry nothing is canonicalized
        (classification.canonical_protonetstructure, Protonetstructure, "A_X__X:C", None, "A_X__X:C"),
    ],
)
def test_the_determined_part_of_an_undetermined_input_is_canonicalized(
    function: Any, family: Any, text: str, kept: str | None, flipped: str
) -> None:
    value = family.undetermined(text)
    assert str(function(value).label) == (kept or text)
    assert str(function(value, preserve_chirality=False).label) == flipped
    assert (function(value) is value) == (kept is None)


def test_similar_refuses_an_undetermined_refined_value() -> None:
    refined = _refined()
    undetermined = Protostructure.undetermined(BARE + NAMES)
    for first, second in ((undetermined, refined), (refined, undetermined), (undetermined, undetermined)):
        with pytest.raises(ValueError, match="determined is False"):
            first.similar(second, 0.1)
    with pytest.raises(ValueError, match="determined is False"):
        Prototype.undetermined(BARE).similar(PrototypeView(refined).unview(), 0.1)


def test_a_label_string_is_read_without_a_search_and_still_checked() -> None:
    # parsing would run the constructor's net search before the bare part; a non-canonical spelling
    # still raises, whether the bare key is canonical (the result decides) or not (the label's own search)
    value = _value("k2agsbs4-frame")
    for function, text in (
        (classification.canonical_protonetstructure, str(value.label)),
        (classification.canonical_protonettype, str(value.protonettype.label)),
    ):
        for chirality in (True, False):
            assert str(function(text, preserve_chirality=chirality).label) == str(
                function(value, preserve_chirality=chirality).label
            )
    occupations, shapes = list(zip(LETTERS, CLASSES, strict=True)), value.protonettype.shapes  # S sites alike
    respelled = str(Protonettype._of(34, occupations, _respelled(ROWS), shapes).label)
    assert respelled != str(value.protonettype.label)
    for chirality in (True, False):
        with pytest.raises(ValueError, match="not a canonical protonettype label"):
            classification.canonical_protonettype(respelled, preserve_chirality=chirality)
    with pytest.raises(ValueError, match="not a canonical protonetstructure label"):
        classification.canonical_protonetstructure("AB_cF8_225_a_b__1~2:Na-Cl")  # D3 puts Cl on a
    assert str(classification.canonical_protonetstructure("AB_cF8_225_a_b__1~2:Cl-Na").label) == (
        "AB_cF8_225_a_b__1~2:Cl-Na"
    )


VIEW_LEVELS = [
    (ProtonetstructureView, ProtonetstructureLabel, "protonetstructure"),
    (ProtonettypeView, ProtonettypeLabel, "protonettype"),
    (ProtostructureView, ProtostructureLabel, "protostructure"),
    (PrototypeView, PrototypeLabel, "prototype"),
    (BareProtostructureView, BareProtostructureLabel, "bare_protostructure"),
    (BarePrototypeView, BarePrototypeLabel, "bare_prototype"),
]
VIEW_SOURCES = [
    (Protonetstructure, f"A4B2CD_oP32_34_X__X{NAMES}"),
    (Protonetstructure, f"A4B2CD_X__X{NAMES}"),
    (Protonetstructure, f"{BARE}__X{NAMES}"),
    (Protonettype, f"{BARE}__X"),
    (Protonettype, "A4B2CD_oP32_34_X__X"),
    (Protostructure, BARE + NAMES),
    (Protostructure, f"A4B2CD_oP32_34_X{NAMES}"),
    (Prototype, BARE),
    (Prototype, "A4B2CD_X"),
    (BareProtostructure, f"A4B2CD_oP32_34_X{NAMES}"),
    (BarePrototype, "A4B2CD_X"),
]


@pytest.mark.parametrize(("family", "text"), VIEW_SOURCES)
def test_views_of_an_undetermined_value_at_every_broader_level(family: Any, text: str) -> None:
    source, presented = family.undetermined(text), 0
    for view, label_view, kind in VIEW_LEVELS:
        try:
            shown = view(source)
        except TypeError:  # not a broader level of this family
            continue
        cut = text.partition(":")[0] if kind in ("protonettype", "prototype", "bare_prototype") else text
        cut = cut if kind.startswith("protonet") else cut.replace("__X", "", 1)
        determined = kind not in ("prototype", "protostructure") and not cut.partition(":")[0].endswith("_X")
        assert shown.determined == determined and shown._backend.determined == determined
        assert str(shown.label) == str(label_view(source)) == cut
        assert str(shown.unview().label) == cut and label_view(source).unview() == shown.unview()
        assert shown.unview().determined == determined
        assert str(shown.anonymous_formula) == "A4B2CD"
        presented += 1
    assert presented >= (1 if family is BarePrototype else 2)


def test_on_limit_is_validated_and_canonicalize_keeps_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    value = _value("nacl")
    for function in (
        classification.canonical_bare_protostructure,
        classification.canonical_bare_prototype,
        classification.canonical_protonetstructure,
        classification.canonical_protonettype,
    ):
        with pytest.raises(ValueError, match="on_limit"):
            function(value, on_limit="ignore")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="on_limit"):
        Protonetstructure.from_asu(httk.core.load(str(K2AGSBS4)), [], on_limit="ignore")  # type: ignore[arg-type]
    monkeypatch.setattr(classification, "_canonical_bare", _over_limit)
    with pytest.raises(CanonicalizationLimitError):
        canonicalize(BareProtostructureView(value).unview())


def test_from_asu_net_timeout_bins_by_the_canonical_bare_part(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_protonet_canonical_api import _case

    asu, bonds, _, _ = _case("nacl-exchanged")  # Cl on a, Na on b: the canonical bare part puts Cl on a too
    monkeypatch.setattr(Protonetstructure, "__init__", _over_limit)
    value = Protonetstructure.from_asu(asu, bonds)
    assert str(value.label) == "AB_cF8_225_a_b__X:Cl-Na"
    _assert_round_trips(value)
    with pytest.raises(CanonicalizationLimitError):
        Protonetstructure.from_asu(asu, bonds, on_limit="raise")


def _clear_net_caches() -> None:
    protonet._search.cache_clear()
    protonet._component_codes.cache_clear()


def test_a_large_molecule_over_a_short_budget_gets_its_bare_label_with_an_undetermined_net() -> None:
    # COD 7218683 (Cu(C9H9BrNO)2 in P2_1/c, 22 sites, 9 bond orbits by the generic cutoffs): its net
    # search takes about 0.3 s with cold caches, the warm bare part about 10 ms; it finishes within the
    # default budget. Real deadlines below the measured search time: every result is one of the valid
    # forms, and a budget the bare part meets but the net search does not gives the bare label with __X.
    path = str(FIXTURES / "cod" / "7218683.cif")
    loaded = load_structure(path)
    assert loaded is not None
    it, sites, refs, lattice = loaded
    cif = httk.core.load(path)
    spec = [(name, ws.wyckoff, tuple(ws.free_params.to_fractions())) for (name, _), ws in zip(sites, cif.wyckoff_sites)]
    asu = _asu(it, spec, cif.cell)
    tables = protonet._tables(it)
    rows, _ = raw_rows(tables, sites, refs, lattice, generic_cutoffs(tables, sites, refs, lattice))
    determined = Protonetstructure.from_asu(asu, _bonds(rows))  # also warms the bare caches
    bare = str(classification.canonical_bare_protostructure(asu).label)
    assert bare == "A18B18C2D2E2F_mP86_14_9e_9e_e_e_e_a:C-H-Br-N-O-Cu"
    expected = bare.replace(":", "__X:")
    _clear_net_caches()
    start = time.perf_counter()
    classification.canonical_protonetstructure(determined, timeout=None)
    search = time.perf_counter() - start
    budgets = [search / 2**k for k in range(1, 8)]
    for source in (determined, str(determined.label)):  # a value, and its label string read without a search
        seen = set()
        for budget in budgets:
            _clear_net_caches()
            result = str(classification.canonical_protonetstructure(source, timeout=budget).label)
            assert result in {str(determined.label), expected, "A18B18C2D2E2F_mP86_14_X__X:C-H-Br-N-O-Cu"}
            seen.add(result)
        assert expected in seen
    seen = set()
    for budget in budgets:
        _clear_net_caches()
        with _budget_scope(budget):
            value = Protonetstructure.from_asu(asu, _bonds(rows))
        assert str(value.label) in {str(determined.label), expected, "A18B18C2D2E2F_mP86_14_X__X:C-H-Br-N-O-Cu"}
        seen.add(str(value.label))
        if not value.determined:
            _assert_round_trips(value)
    assert expected in seen


# ---------------------------------------------------------------------------------------------
# grammar
# ---------------------------------------------------------------------------------------------
def _fields(label: str) -> list[str]:
    main, _, names = label.partition(":")
    return [*main.replace("__", "_").split("_"), *(names.split("-") if names else [])]


def _fixture_labels() -> list[str]:
    labels = [*LABELS.values(), *(label for _, label in MOLECULAR.values())]
    for name in LABEL_FIXTURES:
        structure = httk.core.load(FIXTURES / name, repair=True)
        labels.append(str(BareProtostructureView(structure).label))
        labels.append(str(BarePrototypeView(structure).label))
    return labels


def test_a_determined_label_never_has_a_field_x() -> None:
    labels = _fixture_labels()
    assert len(labels) > 40
    for label in labels:
        assert "X" not in _fields(label), label
        assert not _marks_undetermined(label)


@pytest.mark.parametrize(
    ("family", "text"),
    [
        (BarePrototype, "A4B2CD_X:S-K-Ag-Sb"),  # no names at the anonymous level
        (BareProtostructure, "A4B2CD_X"),  # names needed
        (BareProtostructure, "A4B2CD_X:S-K-Ag"),  # one name per class
        (BareProtostructure, "A2B2_X:S-K"),  # not a canonical Formulatype
        (BareProtostructure, "AB_X:Na-Cl"),  # equal counts are ordered by name
        (BareProtostructure, "AB_X:Na-Na"),
        (BareProtostructure, "AB_X:Na-X"),  # the unknown symbol is no class name
        (BareProtostructure, "AB_X:Na-S-2"),
        (BarePrototype, "A4B2CD_oP31_34_X"),  # Pearson count not a multiple of the Formulatype
        (BarePrototype, "A4B2CD_tP32_34_X"),  # Pearson symbol of another crystal system
        (BarePrototype, "A_cF4_231_X"),
        (BarePrototype, "A_X_X"),
        (BarePrototype, BARE),  # a determined label is no undetermined bare value
        (Protonettype, "A4B2CD_oP32_34_X"),  # the net marker is required
        (Protonettype, f"{BARE}__1~2"),
        (Protonettype, f"{BARE}_a__X"),  # the bare part must be a label
        (Protonetstructure, "AB_hP4_186_b_b__X:Zn-S"),  # names in the bare label's order
        (Prototype, "A4B2CD_oP32_34_X__X"),
        (BarePrototype, "A_cP1_0221_X"),  # the number is written canonically
        (BarePrototype, "A_cP1_\u0662\u0662\u0661_X"),  # in ASCII digits
        (BarePrototype, "A_cP0_221_X"),  # a Pearson count is positive
        (BarePrototype, "A_cF1_225_X"),  # a count the group's lattice points cannot give
        (BarePrototype, "AB_cF4_225_X"),  # one primitive site cannot hold two classes
        (BarePrototype, "AB_oC2_65_X"),
        (BarePrototype, "A2B_hR1_166_X"),  # hR counts primitive sites
    ],
)
def test_malformed_undetermined_labels_are_rejected(family: Any, text: str) -> None:
    with pytest.raises(ValueError):
        family.undetermined(text)


def test_parsers_and_views_accept_undetermined_labels_without_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(protonet, "_canonical_net", _over_limit)  # no search may run
    cases = [
        (try_parse_bare_prototype, BarePrototypeView, "A4B2CD_oP32_34_X"),
        (try_parse_bare_protostructure, BareProtostructureView, f"A4B2CD_X{NAMES}"),
        (try_parse_protonettype, ProtonettypeView, f"{BARE}__X"),
        (try_parse_protonetstructure, ProtonetstructureView, f"A4B2CD_X__X{NAMES}"),
    ]
    for parse, view, text in cases:
        value = parse(text)
        assert value is not None and str(value.label) == text and not value.determined
        assert view(text).unview() == value and not view(text).determined
    assert try_parse_bare_prototype(f"{BARE}__X") is None and try_parse_protonettype(f"A4B2CD_X{NAMES}") is None


def test_unknown_parts_raise_value_errors_naming_determined() -> None:
    value = Protonetstructure.undetermined(f"A4B2CD_X__X{NAMES}")
    for part in ("spacegroup", "occupations", "net", "nsites_conventional", "pearson_symbol"):
        with pytest.raises(ValueError, match="determined is False"):
            getattr(value, part)
    net_only = Protonetstructure.undetermined(f"{BARE}__X{NAMES}")
    assert [o.species.name for o in net_only.occupations] == ["S"] * 4 + ["K"] * 3 + ["Ag"] * 2 + ["Sb"]
    assert net_only.multiplicities() == _value("k2agsbs4-frame").multiplicities()
    with pytest.raises(ValueError, match="determined is False"):
        _ = net_only.net


def test_equal_labels_are_equal_values_whatever_the_structures() -> None:
    first, second = (Protonettype.undetermined("A4B2CD_oP32_34_X__X") for _ in range(2))
    assert first == second and hash(first) == hash(second) and first is not second
    assert len({first, second, Protonettype.undetermined("A4B2CD_X__X")}) == 2
    assert BarePrototype.undetermined("A_X") != BareProtostructure.undetermined("A_X:C")
    assert Protonettype.undetermined(f"{BARE}__X") != _value("k2agsbs4-frame").protonettype


# ---------------------------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------------------------
RECORD_CASES = [
    (BarePrototype.undetermined("A4B2CD_X"), _bare_prototype_record_from_value, _bare_prototype_from_record),
    (
        BareProtostructure.undetermined(f"A4B2CD_oP32_34_X{NAMES}"),
        _bare_protostructure_record_from_value,
        _bare_protostructure_from_record,
    ),
    (Prototype.undetermined(BARE), _prototype_record_from_value, _prototype_from_record),
    (Protostructure.undetermined(f"A4B2CD_X{NAMES}"), _protostructure_record_from_value, _protostructure_from_record),
    (Protonettype.undetermined(f"{BARE}__X"), _protonettype_record_from_value, _protonettype_from_record),
    (
        Protonetstructure.undetermined(f"A4B2CD_oP32_34_X__X{NAMES}"),
        _protonetstructure_record_from_value,
        _protonetstructure_from_record,
    ),
]


@pytest.mark.parametrize(("value", "to_record", "from_record"), RECORD_CASES)
def test_records_round_trip_undetermined_values(value: Any, to_record: Any, from_record: Any) -> None:
    record = to_record(value)
    assert record.undetermined == str(value.label) == record.label
    assert record.occupations == () and getattr(record, "net", ()) == ()
    spacegroup = None if record.spacegroup_it_number is None else value.spacegroup
    assert record.spacegroup_it_number == (None if spacegroup is None else spacegroup.it_number)
    assert from_record(record) == value
    assert content_id(record) == content_id(value)
    type(record).__httk_validate__(record)
    with pytest.raises(ValueError, match="contradicts"):
        from_record(dataclasses.replace(record, spacegroup_it_number=2, spacegroup_hall_entry="-P 1"))
    with pytest.raises(ValueError):
        from_record(dataclasses.replace(record, undetermined=str(value.label) + "_X"))


def test_undetermined_records_store_no_occupations_and_differ_from_determined_ones() -> None:
    value = BareProtostructure.undetermined(f"A4B2CD_oP32_34_X{NAMES}")
    record = _bare_protostructure_record_from_value(value)
    determined = _bare_protostructure_record_from_value(classification.canonical_bare_protostructure(_structure()))
    assert determined.undetermined is None and content_id(record) != content_id(determined)
    with pytest.raises(ValueError, match="only its label"):
        dataclasses.replace(record, occupations=determined.occupations)
    with pytest.raises(TypeError, match="spacegroup_it_number"):
        dataclasses.replace(determined, spacegroup_it_number=None, spacegroup_hall_entry=None)
