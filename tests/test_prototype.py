"""Tests for the merged anonymous Prototype family."""

import pickle

import pytest
from httk.core import FracVector

from httk.atomistic import (
    ASUStructure,
    DerivedPrototype,
    FundamentalDomainTemplate,
    Protostructure,
    Prototype,
    PrototypeBackend,
    PrototypeLabel,
    PrototypeView,
    RecognizedPrototype,
    Species,
    UnitcellStructure,
    WyckoffSite,
)

CELL = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
EMPTY = FracVector(())


def _dummy(label: str) -> Species:
    return Species(label, ("X",), (1,), labels=(label,))


def _template() -> FundamentalDomainTemplate:
    return FundamentalDomainTemplate(
        CELL,
        225,
        (WyckoffSite("a", EMPTY, "A"), WyckoffSite("b", EMPTY, "B")),
        (_dummy("A"), _dummy("B")),
    )


def _asu() -> ASUStructure:
    return ASUStructure(
        CELL,
        225,
        (WyckoffSite("a", EMPTY, "Na"), WyckoffSite("b", EMPTY, "Cl")),
        (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
    )


def _base() -> Prototype:
    return Prototype(225, [("a", "A"), ("b", "B")])


class IdentityCarryingPrototypeBackend(PrototypeBackend):
    """Expose optional class identity through a non-value prototype backend."""

    def __init__(self, value: Prototype) -> None:
        self.value = value

    @property
    def spacegroup(self):
        return self.value.spacegroup

    @property
    def occupations(self):
        return self.value.occupations

    @property
    def representative(self):
        return self.value.representative

    @property
    def discriminator(self):
        return self.value.discriminator

    def unwrap(self):
        return self.value


def test_base_only_and_representative_only_are_valid() -> None:
    base = _base()
    representative = _template()
    with_rep = Prototype(representative=representative)
    assert with_rep.spacegroup == base.spacegroup
    assert with_rep.occupations == base.occupations
    assert with_rep.representative == representative
    assert with_rep.discriminator is None


def test_discriminator_only_requires_base() -> None:
    with pytest.raises(ValueError, match="spacegroup and occupations"):
        Prototype(discriminator="001")
    assert Prototype(225, [("a", "A")], discriminator="001").discriminator == "001"
    with pytest.raises(ValueError, match="non-empty string"):
        Prototype(225, [("a", "A")], discriminator="")


def test_representative_base_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="disagrees with its representative"):
        Prototype(221, [("a", "A")], representative=_template())
    with pytest.raises(ValueError, match="supplied together"):
        Prototype(spacegroup=225, representative=_template())
    with pytest.raises(ValueError, match="supplied together"):
        Prototype(occupations=[("b", "A")], representative=_template())


def test_canonical_anonymous_occupations_and_label() -> None:
    first = Prototype(225, [("a", "A"), ("b", "B")])
    permuted = Prototype(225, [("b", "A"), ("a", "B")])
    assert first == permuted
    assert str(first.label) == "AB_cF8_225_a_b"
    assert first.pearson_symbol == "cF8"
    assert first.anonymous_formula == "AB"


def test_exact_equality_includes_optional_information_and_is_unhashable() -> None:
    representative = _template()
    assert Prototype(representative=representative) == Prototype(representative=representative)
    assert Prototype(225, [("a", "A"), ("b", "B")], discriminator="001") != Prototype(
        225, [("a", "A"), ("b", "B")], discriminator="002"
    )
    assert Prototype(225, [("a", "A"), ("b", "B")]) != Prototype(225, [("a", "A"), ("b", "B")], discriminator="001")
    with pytest.raises(TypeError):
        hash(Prototype(representative=representative))


def test_fundamental_template_and_structure_views_derive_prototypes() -> None:
    representative = _template()
    assert representative.prototype == _base()
    recognized = PrototypeView(_asu()).unview()
    assert isinstance(recognized, Prototype)
    assert recognized.representative is not None
    assert recognized.label == "AB_cF8_225_a_b"


def test_raw_structure_recognition_options_reach_recognized_backend() -> None:
    structure = UnitcellStructure(CELL, [(0, 0, 0)], (Species("Na", ("Na",), (1,)),), ("Na",))
    view = PrototypeView(structure, tolerance=0.123, limit_denominator=97)
    assert isinstance(view._backend, RecognizedPrototype)
    assert view._backend._tolerance == 0.123
    assert view._backend._limit_denominator == 97


def test_protostructure_view_erases_to_anonymous_base_and_preserves_class_info() -> None:
    assigned = Protostructure(225, [("a", "Na"), ("b", "Cl")], discriminator="001")
    view = PrototypeView(assigned)
    assert view._resolved_prototype is None
    assert view.unwrap() is assigned
    value = view.unview()
    assert value.discriminator == "001"
    assert value.representative is None
    assert value.label == "AB_cF8_225_a_b"


def test_prototype_views_and_labels_preserve_optional_identity() -> None:
    assigned = Protostructure(representative=_asu(), discriminator="001")
    derived = DerivedPrototype(assigned)
    expected = derived.resolve()

    view = PrototypeView(derived)
    assert view._resolved_prototype is None
    assert view.unview() == expected
    assert PrototypeLabel(derived).unview() == expected

    recognized = RecognizedPrototype(_asu())
    recognized_view = PrototypeView(recognized)
    assert recognized_view._resolved_prototype is None
    assert recognized_view.unview().representative is not None
    assert PrototypeLabel(recognized).unview() == recognized_view.unview()

    value = Prototype(representative=_template(), discriminator="003")
    generic_backend = IdentityCarryingPrototypeBackend(value)
    assert PrototypeView(generic_backend).unview() == value
    label = PrototypeLabel(generic_backend)
    assert str(label) == "AB_cF8_225_a_b"
    assert label.unview() == value


def test_view_is_lazy_and_pickle_preserves_state() -> None:
    view = PrototypeView(_asu())
    assert view._resolved_prototype is None
    restored = pickle.loads(pickle.dumps(view))
    assert restored._resolved_prototype is None
    assert restored.unview() == view.unview()
    resolved = PrototypeView(_asu())
    _ = resolved.spacegroup
    restored = pickle.loads(pickle.dumps(resolved))
    assert restored.unview() == resolved.unview()


def test_similar_optional_fields_and_delta_validation() -> None:
    one = Prototype(225, [("a", "A"), ("b", "B")])
    two = Prototype(225, [("a", "A"), ("b", "B")], discriminator="001")
    assert one.similar(two, 0.0)
    assert two.similar(one, 0.0)
    assert not two.similar(Prototype(225, [("a", "A"), ("b", "B")], discriminator="002"), 0.0)
    assert not one.similar(Protostructure(225, [("a", "Na"), ("b", "Cl")]), 0.0)
    with pytest.raises(ValueError):
        one.similar(one, -1)
    with pytest.raises(ValueError):
        one.similar(one, float("nan"))
    with pytest.raises(TypeError):
        one.similar(one, "0")
