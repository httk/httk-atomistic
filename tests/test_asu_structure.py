"""Tests for the asymmetric-unit structure representation and its expansion.

Expansion is exact: every assertion here compares whole coordinates as ``Fraction``s, and
none uses an approximate comparison, because there is no tolerance anywhere in the path
from an ASUStructure to a full cell.
"""

import fractions

import pytest
from httk.core import FracVector, unwrap

from httk.atomistic import (
    Assembly,
    ASUStructure,
    ASUStructureView,
    Cell,
    FundamentalDomainStructure,
    SettingTransform,
    Spacegroup,
    Species,
    StructureBackend,
    StructureEntryProvider,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    same_crystal,
)

F = fractions.Fraction

CUBIC = [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]
HEXAGONAL = [[4, 0, 0], [0, 4, 0], [0, 0, 12]]
NO_PARAMETERS = FracVector(())


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _rocksalt() -> ASUStructure:
    """NaCl in Fm-3m: sodium on Wyckoff a, chlorine on Wyckoff b."""
    return ASUStructure(
        cell=CUBIC,
        spacegroup=225,
        wyckoff_sites=[WyckoffSite("a", NO_PARAMETERS, "Na"), WyckoffSite("b", NO_PARAMETERS, "Cl")],
        species=_species("Na", "Cl"),
    )


def _fractions_of(structure: object) -> list[tuple[F, ...]]:
    view = UnitcellStructureView(structure)
    return [tuple(row) for row in view.sites.reduced_coords.to_fractions()]


def test_asu_view_keeps_representative_domain_presentation() -> None:
    view = ASUStructureView(_rocksalt())

    assert view.asu is view
    assert view.nsites == 2
    assert len(view.sites) == 2
    assert view.species_at_sites == ("Na", "Cl")
    assert repr(view).startswith("ASUStructureView(")


# --- expansion ---


def test_rocksalt_expands_to_the_exact_face_centred_cell() -> None:
    structure = UnitcellStructureView(_rocksalt())
    assert len(structure.sites) == 8
    assert structure.species_at_sites == ("Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl")
    assert structure.sites.reduced_coords == FracVector(
        [
            [F(0), F(0), F(0)],
            [F(0), F(1, 2), F(1, 2)],
            [F(1, 2), F(0), F(1, 2)],
            [F(1, 2), F(1, 2), F(0)],
            [F(0), F(0), F(1, 2)],
            [F(0), F(1, 2), F(0)],
            [F(1, 2), F(0), F(0)],
            [F(1, 2), F(1, 2), F(1, 2)],
        ]
    )


def test_expansion_multiplicities_match_the_tabulated_ones() -> None:
    asu = _rocksalt()
    standard = Spacegroup.standard(225)
    assert asu.multiplicities() == (
        standard.wyckoff_position("a").multiplicity,
        standard.wyckoff_position("b").multiplicity,
    )


def test_expansion_is_deterministic_and_cached() -> None:
    asu = _rocksalt()
    assert _fractions_of(asu) == _fractions_of(asu)
    assert asu.expand_sites().reduced_coords == asu.expand_sites().reduced_coords


def test_wyckoff_float_coefficients_are_cached() -> None:
    branch = Spacegroup.standard(15).wyckoff_position("e").branches[0]
    first = branch._float_coefficients
    assert branch._float_coefficients is first


def test_a_free_parameter_places_a_whole_orbit() -> None:
    """One value of ``y`` on Wyckoff e of SG 15 places four atoms, exactly."""
    asu = ASUStructure(
        cell=[[5, 0, 0], [0, 6, 0], [0, 0, 7]],
        spacegroup=15,
        wyckoff_sites=[WyckoffSite("e", FracVector(["1/3"]), "Si")],
        species=_species("Si"),
    )
    assert asu.multiplicities() == (4,)
    assert sorted(_fractions_of(asu)) == sorted(
        [
            (F(0), F(1, 3), F(1, 4)),
            (F(0), F(2, 3), F(3, 4)),
            (F(1, 2), F(1, 6), F(3, 4)),
            (F(1, 2), F(5, 6), F(1, 4)),
        ]
    )


def test_partial_occupancy_survives_expansion() -> None:
    """Occupancy lives in the Species, so every generated site carries it."""
    half_sodium = Species(name="Na_half", chemical_symbols=("Na",), concentration=(0.5,))
    asu = ASUStructure(CUBIC, 225, [WyckoffSite("a", NO_PARAMETERS, "Na_half")], [half_sodium])
    structure = UnitcellStructureView(asu)
    assert len(structure.sites) == 4
    assert structure.species[0].concentration == (0.5,)


@pytest.mark.parametrize("structure_type", [FundamentalDomainStructure, ASUStructure])
def test_redundant_wyckoff_site_raises(structure_type: type[FundamentalDomainStructure]) -> None:
    asu = structure_type(
        CUBIC,
        225,
        [WyckoffSite("a", NO_PARAMETERS, "Na"), WyckoffSite("a", NO_PARAMETERS, "Cl")],
        _species("Na", "Cl"),
    )
    assert isinstance(asu, structure_type)

    with pytest.raises(ValueError, match=r"WyckoffSite\('Cl' at a\).*different species"):
        _ = UnitcellStructureView(asu).sites


def test_distinct_sites_same_letter_do_not_raise() -> None:
    asu = ASUStructure(
        [[5, 0, 0], [0, 6, 0], [0, 0, 7]],
        15,
        [WyckoffSite("e", FracVector(["1/3"]), "Si"), WyckoffSite("e", FracVector(["1/4"]), "Ge")],
        _species("Si", "Ge"),
    )

    assert all(count > 0 for count in asu.multiplicities())


# --- settings ---


def test_a_volume_changing_transform_collapses_the_orbit_exactly() -> None:
    """Rhombohedral axes hold a third of the standard hexagonal cell.

    The standard setting's orbit is three times too big for the smaller cell, and the
    surplus points coincide *exactly* after wrapping. Nothing here is within a tolerance.
    """
    rhombohedral = Spacegroup.from_setting("166:R")
    assert rhombohedral.transform_from_standard.determinant() == F(3)

    in_standard = ASUStructure(HEXAGONAL, 166, [WyckoffSite("a", NO_PARAMETERS, "Bi")], _species("Bi"))
    in_rhombohedral = ASUStructure(
        HEXAGONAL,
        166,
        [WyckoffSite("a", NO_PARAMETERS, "Bi")],
        _species("Bi"),
        transform=rhombohedral.transform_from_standard,
    )
    setting_local = ASUStructure(
        HEXAGONAL,
        rhombohedral,
        [WyckoffSite("a", NO_PARAMETERS, "Bi")],
        _species("Bi"),
    )

    assert in_standard.multiplicities() == (Spacegroup.standard(166).wyckoff_position("a").multiplicity,)
    assert in_rhombohedral.multiplicities() == (rhombohedral.wyckoff_position("a").multiplicity,)
    assert setting_local.multiplicities() == in_rhombohedral.multiplicities()
    assert setting_local.expand_sites() == in_rhombohedral.expand_sites()
    assert setting_local.transform.is_identity()
    assert in_standard.multiplicities() == (3,)
    assert in_rhombohedral.multiplicities() == (1,)


def test_a_supercell_transform_adds_the_missing_lattice_cosets() -> None:
    """A transform onto a larger cell needs points the standard orbit does not contain."""
    doubled = SettingTransform([["1/2", 0, 0], [0, 1, 0], [0, 0, 1]])
    assert len(doubled.lattice_cosets()) == 2

    asu = ASUStructure(
        [[11.28, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
        225,
        [WyckoffSite("a", NO_PARAMETERS, "Na")],
        _species("Na"),
        transform=doubled,
    )
    # Four sites in the standard cell become eight in the doubled one.
    assert asu.multiplicities() == (8,)
    coordinates = _fractions_of(asu)
    assert len(coordinates) == len(set(coordinates)) == 8
    assert (F(1, 4), F(0), F(1, 2)) in coordinates


def test_an_untabulated_setting_is_representable() -> None:
    """The point of storing a transform: a setting in no table works like any other."""
    shifted = SettingTransform(FracVector.eye((3, 3)), ["1/8", "1/8", "1/8"])
    asu = ASUStructure(CUBIC, 225, [WyckoffSite("a", NO_PARAMETERS, "Na")], _species("Na"), transform=shifted)

    assert asu.setting() is None
    assert not asu.is_standard_setting
    assert (F(1, 8), F(1, 8), F(1, 8)) in _fractions_of(asu)

    unshifted = ASUStructure(CUBIC, 225, [WyckoffSite("a", NO_PARAMETERS, "Na")], _species("Na"))
    # The same crystal, moved: same number of atoms, different coordinates.
    assert len(_fractions_of(asu)) == len(_fractions_of(unshifted))
    assert not same_crystal(asu, unshifted)


def test_a_tabulated_setting_reports_itself() -> None:
    asu = ASUStructure(
        [[5, 0, 0], [0, 6, 0], [0, 0, 7]],
        15,
        [WyckoffSite("e", FracVector(["1/3"]), "Si")],
        _species("Si"),
        transform=Spacegroup.from_setting("15:c1").transform_from_standard,
    )
    setting = asu.setting()
    assert setting is not None
    assert setting.setting == "15:c1"


# --- validation ---


def test_asu_structure_rejects_inconsistent_input() -> None:
    with pytest.raises(ValueError, match="unknown species"):
        ASUStructure(CUBIC, 225, [WyckoffSite("a", NO_PARAMETERS, "Xx")], _species("Na"))

    with pytest.raises(KeyError):
        ASUStructure(CUBIC, 225, [WyckoffSite("zz", NO_PARAMETERS, "Na")], _species("Na"))

    with pytest.raises(ValueError, match="free parameter"):
        # Wyckoff a of SG 225 is a fixed position and takes no parameters.
        ASUStructure(CUBIC, 225, [WyckoffSite("a", FracVector(["1/3"]), "Na")], _species("Na"))

    with pytest.raises(ValueError, match="unique"):
        ASUStructure(CUBIC, 225, [WyckoffSite("a", NO_PARAMETERS, "Na")], _species("Na") + _species("Na"))


def test_asu_structure_keeps_a_tabulated_setting_local() -> None:
    local = Spacegroup.from_setting("15:c1")
    asu = ASUStructure(CUBIC, local, [WyckoffSite("a", NO_PARAMETERS, "Na")], _species("Na"))

    assert asu.spacegroup == local
    assert asu.transform.is_identity()
    with pytest.raises(ValueError, match="nonidentity"):
        ASUStructure(
            CUBIC,
            local,
            [WyckoffSite("a", NO_PARAMETERS, "Na")],
            _species("Na"),
            transform=local.transform_from_standard,
        )


# --- structure-family integration ---


def test_backend_dispatch_and_kind_override() -> None:
    asu = _rocksalt()
    backend = asu
    assert isinstance(backend, FundamentalDomainStructure)
    assert isinstance(backend, StructureBackend)
    assert UnitcellStructureView(backend)._backend is backend


def test_view_rewrap_identity_shared_backend_and_unwrap() -> None:
    asu = _rocksalt()
    backend = asu
    first = UnitcellStructureView(backend)
    assert UnitcellStructureView(first) is first
    second = UnitcellStructureView(backend)
    assert first._backend is backend
    assert second._backend is backend
    assert unwrap(backend) is asu
    assert first.unwrap() is asu


def test_an_asu_structure_is_a_structure_everywhere() -> None:
    """Being in StructureLike is what lets it flow through the rest of the package."""
    view = UnitcellStructureView(_rocksalt())
    assert isinstance(view, UnitcellStructure)
    assert isinstance(view.cell, Cell)
    assert view.cartesian_sites().to_floats()[0] == [0.0, 0.0, 0.0]
    assert UnitcellStructureView(_rocksalt()) == view


def test_symmetry_reduced_expansion_rejects_ambiguous_molecular_placement() -> None:
    molecular = ASUStructure(
        CUBIC,
        225,
        [WyckoffSite("a", NO_PARAMETERS, "Na", FracVector([0, 0, 0]))],
        _species("Na"),
        molecular=True,
    )
    with pytest.raises(ValueError, match="molecular expansion"):
        _ = UnitcellStructureView(molecular).sites


def test_symmetry_reduced_expansion_retains_unambiguous_molecular_representative() -> None:
    point = FracVector([F(1, 7), F(2, 7), F(3, 7)])
    molecular = ASUStructure(
        CUBIC,
        1,
        [WyckoffSite("a", point, "C", point)],
        _species("C"),
        molecular=True,
    )
    expanded = UnitcellStructureView(molecular)
    assert expanded.sites.reduced_coords == FracVector([[F(1, 7), F(2, 7), F(3, 7)]])
    assert expanded.site_coordinate_span == "molecular_unit_cell"


def test_symmetry_reduced_expansion_rejects_ambiguous_assembly_correlations() -> None:
    correlated = ASUStructure(
        CUBIC,
        225,
        [WyckoffSite("a", NO_PARAMETERS, "Na")],
        _species("Na"),
        assemblies=(Assembly(((0,),), (1,)),),
    )
    with pytest.raises(ValueError) as error:
        _ = UnitcellStructureView(correlated).assemblies
    assert str(error.value) == (
        "symmetry-reduced expansion cannot map assembly correlations "
        "when a correlated domain site has multiple unit-cell images"
    )


def test_native_assemblies_are_servable_without_expansion() -> None:
    correlated = ASUStructure(
        CUBIC,
        225,
        [WyckoffSite("a", NO_PARAMETERS, "Na")],
        _species("Na"),
        assemblies=(Assembly(((0,),), (1,)),),
    )

    assert correlated.assemblies == (Assembly(((0,),), (1,)),)
    assert correlated.composition.amounts == (("Na", F(4)),)
    (entry,) = list(StructureEntryProvider({"correlated": correlated}).records("structures"))
    assert entry["assemblies"] == [{"sites_in_groups": [[0]], "group_probabilities": [1.0]}]

    pytest.importorskip("sqlalchemy")
    from httk.store import Backend, EntryIdScheme, SqlStore

    with Backend.sqlite() as database:
        SqlStore(database, entry_ids=EntryIdScheme("httk.test", "1"), entry_records={}).save(correlated)

    with pytest.raises(ValueError, match="assembly correlations"):
        _ = UnitcellStructureView(correlated).assemblies


def test_symmetry_reduced_expansion_remaps_one_to_one_assembly_sites() -> None:
    correlated = ASUStructure(
        CUBIC,
        221,
        [WyckoffSite("a", NO_PARAMETERS, "Na")],
        _species("Na"),
        assemblies=(Assembly(((0,),), (1,)),),
    )
    expanded = UnitcellStructureView(correlated)
    assert len(expanded.sites) == 1
    assert expanded.assemblies is not None
    assert expanded.assemblies[0].sites_in_groups == ((0,),)


# --- same_crystal ---


def test_same_crystal_ignores_order_and_lattice_translation() -> None:
    reference = UnitcellStructureView(_rocksalt())
    reordered = UnitcellStructure(
        reference.cell,
        list(reversed(reference.sites.reduced_coords.to_fractions())),
        reference.species,
        tuple(reversed(reference.species_at_sites)),
    )
    assert same_crystal(reference, reordered)

    # Shifting a site by a whole lattice vector names the same atom.
    translated_coords = [list(row) for row in reference.sites.reduced_coords.to_fractions()]
    translated_coords[0] = [value + 1 for value in translated_coords[0]]
    translated = UnitcellStructure(reference.cell, translated_coords, reference.species, reference.species_at_sites)
    assert same_crystal(reference, translated)


def test_same_crystal_detects_real_differences() -> None:
    reference = UnitcellStructureView(_rocksalt())

    different_cell = UnitcellStructure(
        [[5.65, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
        reference.sites.reduced_coords,
        reference.species,
        reference.species_at_sites,
    )
    assert not same_crystal(reference, different_cell)

    swapped = UnitcellStructure(
        reference.cell,
        reference.sites.reduced_coords,
        reference.species,
        tuple(reversed(reference.species_at_sites)),
    )
    assert not same_crystal(reference, swapped)

    # A doubled atom must not compare equal: the multiset counts repeats.
    coords = [list(row) for row in reference.sites.reduced_coords.to_fractions()]
    doubled = UnitcellStructure(
        reference.cell,
        coords + [coords[0]],
        reference.species,
        reference.species_at_sites + (reference.species_at_sites[0],),
    )
    assert not same_crystal(reference, doubled)
