"""Tests for the assigned geometrical-class crystallotype family."""

import pickle
from fractions import Fraction

import pytest
from httk.core import FracVector

from httk.atomistic import (
    Assembly,
    ASUStructure,
    FundamentalDomainStructure,
    Protostructure,
    ProtostructureView,
    Prototype,
    PrototypeView,
    SettingTransform,
    Spacegroup,
    Species,
    Crystallotype,
    CrystallotypeView,
    UnitcellStructureView,
    WyckoffSite,
)

CELL = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]
EMPTY = FracVector(())


def _rocksalt_asu(transform: SettingTransform | None = None) -> ASUStructure:
    return ASUStructure(
        CELL,
        225,
        (WyckoffSite("a", EMPTY, "Na"), WyckoffSite("b", EMPTY, "Cl")),
        (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
        transform=transform,
    )


def _rocksalt_protostructure() -> Protostructure:
    # Use the same exact Species the ASU carries (a bare "Na" string would add a default
    # concentration precision and make an unequal protostructure).
    return Protostructure(225, [("a", Species("Na", ("Na",), (1,))), ("b", Species("Cl", ("Cl",), (1,)))])


def test_requires_at_least_one_of_representative_or_discriminator() -> None:
    with pytest.raises(ValueError, match="at least one of representative or discriminator"):
        Crystallotype(_rocksalt_protostructure())


def test_discriminator_only_requires_a_protostructure() -> None:
    with pytest.raises(ValueError, match="needs a protostructure"):
        Crystallotype(discriminator="001")


def test_discriminator_must_be_non_empty_string() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        Crystallotype(_rocksalt_protostructure(), discriminator="")


def test_non_standard_setting_representative_is_rejected() -> None:
    representative = ASUStructure(
        CELL,
        Spacegroup.from_setting("15:c1"),
        (WyckoffSite("a", EMPTY, "Na"),),
        (Species("Na", ("Na",), (1,)),),
    )
    with pytest.raises(ValueError, match="IT standard setting"):
        Crystallotype(representative=representative)


def test_non_identity_transform_representative_is_rejected() -> None:
    transform = SettingTransform(FracVector.eye((3, 3)), (Fraction(1, 2), 0, 0))
    with pytest.raises(ValueError, match="identity setting transform"):
        Crystallotype(representative=_rocksalt_asu(transform=transform))


def test_representative_with_site_moments_is_rejected() -> None:
    from httk.atomistic.models.moments.collinear import CollinearSiteMoments

    representative = ASUStructure(
        CELL,
        225,
        (
            WyckoffSite("a", EMPTY, "Na", moment=CollinearSiteMoments((1,))),
            WyckoffSite("b", EMPTY, "Cl", moment=CollinearSiteMoments((0,))),
        ),
        (Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))),
    )
    with pytest.raises(ValueError, match="site moments"):
        Crystallotype(representative=representative)


def test_occupation_mismatch_between_protostructure_and_representative_is_rejected() -> None:
    with pytest.raises(ValueError, match="disagrees with its representative"):
        Crystallotype(Protostructure(225, [("a", "Na")]), representative=_rocksalt_asu())


def test_representative_only_derives_the_protostructure() -> None:
    crystallotype = Crystallotype(representative=_rocksalt_asu())
    assert crystallotype.protostructure == _rocksalt_protostructure()
    assert crystallotype.discriminator is None


def test_equality_semantics_across_the_triple() -> None:
    representative = _rocksalt_asu()
    protostructure = _rocksalt_protostructure()
    assert Crystallotype(representative=representative) == Crystallotype(representative=representative)
    assert Crystallotype(protostructure, discriminator="001") == Crystallotype(protostructure, discriminator="001")
    assert Crystallotype(protostructure, discriminator="001") != Crystallotype(protostructure, discriminator="002")
    assert Crystallotype(representative=representative) != Crystallotype(protostructure, discriminator="001")
    with pytest.raises(TypeError):
        hash(Crystallotype(representative=representative))


def test_label_and_aflow_label_delegate_to_the_protostructure() -> None:
    crystallotype = Crystallotype(representative=_rocksalt_asu())
    assert crystallotype.label == "AB_cF8_225_a_b:Na-Cl"
    assert crystallotype.aflow_label == crystallotype.protostructure.aflow_label
    assert crystallotype.pearson_symbol == "cF8"
    assert crystallotype.protochroma.label == "AB_cF8_225_a_b"


def test_protostructure_view_erases_a_crystallotype() -> None:
    crystallotype = Crystallotype(representative=_rocksalt_asu())
    assert ProtostructureView(crystallotype).unview() == _rocksalt_protostructure()


def test_prototype_view_erases_a_crystallotype_and_carries_the_discriminator() -> None:
    with_representative = Crystallotype(representative=_rocksalt_asu(), discriminator="001")
    prototype = PrototypeView(with_representative).unview()
    assert isinstance(prototype, Prototype)
    assert prototype.discriminator == "001"
    assert prototype.representative is not None
    assert prototype.label == "AB_cF8_225_a_b"

    discriminator_only = Crystallotype(_rocksalt_protostructure(), discriminator="XYZ")
    erased = PrototypeView(discriminator_only).unview()
    assert erased.discriminator == "XYZ"
    assert erased.representative is None
    assert erased.protochroma.label == "AB_cF8_225_a_b"


def test_recognition_from_exact_fundamental_domain_needs_no_spglib() -> None:
    crystallotype = CrystallotypeView(_rocksalt_asu()).unview()
    assert isinstance(crystallotype.representative, FundamentalDomainStructure)
    assert crystallotype.protostructure == _rocksalt_protostructure()
    assert crystallotype.discriminator is None


def test_recognition_from_raw_structure_is_spglib_gated() -> None:
    pytest.importorskip("spglib")
    unitcell = UnitcellStructureView(_rocksalt_asu())
    assert CrystallotypeView(unitcell).unview() == CrystallotypeView(_rocksalt_asu()).unview()


def test_crystallotype_value_pickle_round_trip() -> None:
    representative_only = Crystallotype(representative=_rocksalt_asu())
    discriminator_only = Crystallotype(_rocksalt_protostructure(), discriminator="001")
    for value in (representative_only, discriminator_only):
        restored = pickle.loads(pickle.dumps(value))
        assert restored == value


def test_crystallotype_view_pickle_preserves_resolved_value() -> None:
    view = CrystallotypeView(_rocksalt_asu())
    _ = view.protostructure  # resolve
    restored = pickle.loads(pickle.dumps(view))
    assert restored.unview() == view.unview()


def test_representative_with_assemblies_is_rejected() -> None:
    representative = ASUStructure(
        CELL,
        221,
        (WyckoffSite("a", EMPTY, "Na"),),
        (Species("Na", ("Na",), (1,)),),
        assemblies=(Assembly(((0,),), (1,)),),
    )
    with pytest.raises(ValueError, match="assemblies"):
        Crystallotype(representative=representative)


def test_crystallotype_view_unresolved_pickle_stays_lazy() -> None:
    view = CrystallotypeView(_rocksalt_asu())  # no field access -> unresolved
    restored = pickle.loads(pickle.dumps(view))
    assert restored._resolved_crystallotype is None
    assert restored.unview() == view.unview()


def test_lazy_erasure_arrows_are_not_resolved_at_construction_and_recover_source() -> None:
    crystallotype = Crystallotype(representative=_rocksalt_asu(), discriminator="001")

    prototype_view = PrototypeView(crystallotype)
    assert prototype_view._resolved_prototype is None  # lazy: no erasure at construction
    assert prototype_view.unwrap() is crystallotype  # unwrap recovers the source
    restored = pickle.loads(pickle.dumps(prototype_view))
    assert restored._resolved_prototype is None
    assert restored.unview().discriminator == "001"

    protostructure_view = ProtostructureView(crystallotype)
    assert protostructure_view._resolved_protostructure is None
    assert protostructure_view.unwrap() is crystallotype
    assert protostructure_view.unview() == crystallotype.protostructure
