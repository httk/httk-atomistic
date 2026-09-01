import fractions

import pytest

from httk.atomistic import (
    CartesianSiteMoments,
    Cell,
    CollinearSiteMoments,
    CrystalAxisSiteMoments,
    FundamentalDomainStructure,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    build_supercell,
    recognize_asu,
    same_crystal,
)
from httk.atomistic.composition import derive_structure_features
from httk.atomistic.symmetry.standardization import conventional_cell

F = fractions.Fraction


def _cell() -> Cell:
    return Cell([[2, 0, 0], [0, 3, 0], [0, 0, 4]])


def _species() -> tuple[Species, Species]:
    return Species("Na", ("Na",), (1,)), Species("Cl", ("Cl",), (1,))


def _unitcell(moments=None, *, cell=None) -> UnitcellStructure:
    return UnitcellStructure(
        cell or _cell(),
        [[0, 0, 0], [F(1, 2), F(1, 2), F(1, 2)]],
        _species(),
        ("Na", "Cl"),
        site_moments=moments,
    )


def test_unitcell_moments_validate_roundtrip_and_equality() -> None:
    moments = CartesianSiteMoments([[1, 2, 3], [-1, 0, 1]])
    structure = _unitcell(moments)
    assert structure.site_moments is moments
    assert structure == _unitcell(CartesianSiteMoments([[1, 2, 3], [-1, 0, 1]]))
    assert structure != _unitcell(CartesianSiteMoments([[1, 2, 3], [0, 0, 1]]))
    assert structure != _unitcell()

    with pytest.raises(ValueError, match="same length as sites"):
        _unitcell(CartesianSiteMoments([[1, 2, 3]]))
    with pytest.raises(TypeError, match="frame-ambiguous"):
        _unitcell([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="incoherent"):
        _unitcell(CrystalAxisSiteMoments([[1, 2, 3], [4, 5, 6]], Cell([[3, 0, 0], [0, 3, 0], [0, 0, 3]])))


def test_unitcell_view_carries_moments() -> None:
    moments = CartesianSiteMoments([[1, 2, 3], [4, 5, 6]])
    assert UnitcellStructureView(_unitcell(moments)).site_moments is moments


def test_wyckoff_moments_and_fundamental_domain_validation() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        WyckoffSite("a", (), "Na", moment=CartesianSiteMoments([[1, 2, 3], [4, 5, 6]]))

    good = WyckoffSite("a", (), "Na", moment=CartesianSiteMoments([[1, 2, 3]]))
    other = WyckoffSite("b", (), "Cl")
    with pytest.raises(ValueError, match="state moments for all sites or none"):
        FundamentalDomainStructure(_cell(), 225, (good, other), _species())
    with pytest.raises(ValueError, match="same kind"):
        FundamentalDomainStructure(
            _cell(),
            225,
            (good, WyckoffSite("b", (), "Cl", moment=CollinearSiteMoments([1]))),
            _species(),
        )
    with pytest.raises(ValueError, match="incoherent"):
        FundamentalDomainStructure(
            _cell(),
            225,
            (
                WyckoffSite(
                    "a", (), "Na", moment=CrystalAxisSiteMoments([[1, 2, 3]], Cell([[3, 0, 0], [0, 3, 0], [0, 0, 3]]))
                ),
                WyckoffSite("b", (), "Cl", moment=CrystalAxisSiteMoments([[4, 5, 6]], _cell())),
            ),
            _species(),
        )


def test_fundamental_domain_expands_moments_in_species_block_order() -> None:
    asu = FundamentalDomainStructure(
        _cell(),
        225,
        (
            WyckoffSite("a", (), "Na", moment=CartesianSiteMoments([[1, 2, 3]])),
            WyckoffSite("b", (), "Cl", moment=CartesianSiteMoments([[-1, 0, 2]])),
        ),
        _species(),
    )
    moments = asu.site_moments
    assert moments is not None
    assert len(moments) == len(asu.species_at_sites) == 8
    assert (
        tuple(
            tuple(moments.cartesian_moments._element((row, column)) for column in range(3))
            for row in range(len(moments))
        )
        == (tuple(F(value) for value in (1, 2, 3)),) * 4 + (tuple(F(value) for value in (-1, 0, 2)),) * 4
    )
    assert UnitcellStructureView(asu).site_moments == moments

    molecular = FundamentalDomainStructure(
        _cell(),
        2,
        (
            WyckoffSite("a", (), "Na", representative=(0, 0, 0), moment=CartesianSiteMoments([[1, 2, 3]])),
            WyckoffSite("b", (), "Cl", representative=(0, 0, F(1, 2)), moment=CartesianSiteMoments([[-1, 0, 2]])),
        ),
        _species(),
        molecular=True,
    )
    assert len(molecular.site_moments) == 2


def test_same_crystal_compares_exact_moments_across_vector_frames() -> None:
    cartesian = CartesianSiteMoments([[1, 2, 3], [4, 5, 6]])
    crystalaxis = CrystalAxisSiteMoments([[1, 2, 3], [4, 5, 6]], _cell())
    assert same_crystal(_unitcell(cartesian), _unitcell(crystalaxis))
    assert not same_crystal(_unitcell(cartesian), _unitcell(CartesianSiteMoments([[1, 2, 3], [4, 5, 7]])))
    assert not same_crystal(_unitcell(cartesian), _unitcell())
    assert not same_crystal(_unitcell(CollinearSiteMoments([1, 2])), _unitcell(cartesian))


def test_structure_features_mark_magnetism() -> None:
    assert "_httk_magnetism" not in derive_structure_features(_unitcell())
    features = derive_structure_features(_unitcell(CartesianSiteMoments([[1, 2, 3], [4, 5, 6]])))
    assert "_httk_magnetism" in features
    assert features == tuple(sorted(features))


def test_numeric_view_renders_cartesian_moments() -> None:
    numpy = pytest.importorskip("numpy")
    numeric = _unitcell(CartesianSiteMoments([[1, 2, 3], [4, 5, 6]])).numeric()
    assert type(numeric.site_moments) is numpy.ndarray
    assert numeric.site_moments.dtype == numpy.float64
    assert numeric.site_moments.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_recognition_collapses_uniform_orbit_moments_and_round_trips() -> None:
    asu = FundamentalDomainStructure(
        Cell([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        225,
        (
            WyckoffSite("a", (), "Na", moment=CartesianSiteMoments([[1, 2, 3]])),
            WyckoffSite("b", (), "Cl", moment=CartesianSiteMoments([[-1, 0, 2]])),
        ),
        _species(),
    )
    expanded = UnitcellStructureView(asu)
    recovered = recognize_asu(expanded, setting=asu.setting())

    assert [site.moment for site in recovered.wyckoff_sites] == [
        CartesianSiteMoments([[1, 2, 3]]),
        CartesianSiteMoments([[-1, 0, 2]]),
    ]
    assert same_crystal(expanded, UnitcellStructureView(recovered))


def test_recognition_rejects_antiferromagnetic_orbit_moments() -> None:
    asu = FundamentalDomainStructure(
        Cell([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        225,
        (WyckoffSite("a", (), "Na"), WyckoffSite("b", (), "Cl")),
        _species(),
    )
    expanded = UnitcellStructureView(asu)
    moments = CartesianSiteMoments([[1, 0, 0], [-1, 0, 0], [1, 0, 0], [1, 0, 0]] + [[0, 0, 0]] * 4)
    broken = UnitcellStructure(
        expanded.cell, expanded.sites, expanded.species, expanded.species_at_sites, site_moments=moments
    )

    with pytest.raises(ValueError, match="non-uniform site moments"):
        recognize_asu(broken, setting=asu.setting())


def test_conventional_cell_carries_uniform_asu_moments() -> None:
    asu = FundamentalDomainStructure(
        Cell([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        225,
        (WyckoffSite("a", (), "Na", moment=CartesianSiteMoments([[1, 2, 3]])),),
        (_species()[0],),
    )
    result = conventional_cell(asu)
    moments = result.structure.site_moments
    assert isinstance(moments, CartesianSiteMoments)
    # SG 225 is F-centred, so the single ASU site expands to four sites all sharing the moment.
    assert len(moments) == len(result.structure.sites) == 4
    assert moments == CartesianSiteMoments([[1, 2, 3]] * 4)


def test_supercell_replicates_moments_in_site_order_and_converts_crystal_axes() -> None:
    cartesian = CartesianSiteMoments([[1, 2, 3], [4, 5, 6]])
    collinear = CollinearSiteMoments([1, -2])
    transformation = [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
    cartesian_result = build_supercell(_unitcell(cartesian), transformation).structure.site_moments
    collinear_result = build_supercell(_unitcell(collinear), transformation).structure.site_moments
    crystalaxis = CrystalAxisSiteMoments([[1, 2, 3], [4, 5, 6]], _cell())
    crystalaxis_result = build_supercell(_unitcell(crystalaxis), transformation).structure.site_moments

    assert cartesian_result == CartesianSiteMoments([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]])
    assert collinear_result == CollinearSiteMoments([1, -2, 1, -2])
    assert crystalaxis_result == CartesianSiteMoments([[1, 2, 3], [4, 5, 6], [1, 2, 3], [4, 5, 6]])
