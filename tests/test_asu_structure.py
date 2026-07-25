"""Tests for the asymmetric-unit structure representation and its expansion.

Expansion is exact: every assertion here compares whole coordinates as ``Fraction``s, and
none uses an approximate comparison, because there is no tolerance anywhere in the path
from an ASUStructure to a full cell.
"""

import fractions

import pytest
from httk.core import FracVector, unwrap

from httk.atomistic import (
    ASUSite,
    ASUStructure,
    Cell,
    SettingTransform,
    Spacegroup,
    Species,
    Structure,
    StructureASU,
    StructureBackend,
    StructureSimpleView,
    same_crystal,
)

F = fractions.Fraction

CUBIC = [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]
HEXAGONAL = [[4, 0, 0], [0, 4, 0], [0, 0, 12]]
NO_PARAMETERS = FracVector.create(())


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _rocksalt() -> ASUStructure:
    """NaCl in Fm-3m: sodium on Wyckoff a, chlorine on Wyckoff b."""
    return ASUStructure(
        cell=CUBIC,
        spacegroup=225,
        asu_sites=[ASUSite("a", NO_PARAMETERS, "Na"), ASUSite("b", NO_PARAMETERS, "Cl")],
        species=_species("Na", "Cl"),
    )


def _fractions_of(structure: object) -> list[tuple[F, ...]]:
    view = StructureSimpleView(structure)
    return [tuple(row) for row in view.sites.reduced_coords.to_fractions()]


# --- expansion ---


def test_rocksalt_expands_to_the_exact_face_centred_cell() -> None:
    structure = StructureSimpleView(_rocksalt())
    assert len(structure.sites) == 8
    assert structure.species_at_sites == ("Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl")
    assert structure.sites.reduced_coords == FracVector.create(
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


def test_a_free_parameter_places_a_whole_orbit() -> None:
    """One value of ``y`` on Wyckoff e of SG 15 places four atoms, exactly."""
    asu = ASUStructure(
        cell=[[5, 0, 0], [0, 6, 0], [0, 0, 7]],
        spacegroup=15,
        asu_sites=[ASUSite("e", FracVector.create(["1/3"]), "Si")],
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
    asu = ASUStructure(CUBIC, 225, [ASUSite("a", NO_PARAMETERS, "Na_half")], [half_sodium])
    structure = StructureSimpleView(asu)
    assert len(structure.sites) == 4
    assert structure.species[0].concentration == (0.5,)


# --- settings ---


def test_a_volume_changing_transform_collapses_the_orbit_exactly() -> None:
    """Rhombohedral axes hold a third of the standard hexagonal cell.

    The standard setting's orbit is three times too big for the smaller cell, and the
    surplus points coincide *exactly* after wrapping. Nothing here is within a tolerance.
    """
    rhombohedral = Spacegroup.for_setting("166:R")
    assert rhombohedral.transform_from_standard.determinant() == F(3)

    in_standard = ASUStructure(HEXAGONAL, 166, [ASUSite("a", NO_PARAMETERS, "Bi")], _species("Bi"))
    in_rhombohedral = ASUStructure(
        HEXAGONAL,
        166,
        [ASUSite("a", NO_PARAMETERS, "Bi")],
        _species("Bi"),
        transform=rhombohedral.transform_from_standard,
    )

    assert in_standard.multiplicities() == (Spacegroup.standard(166).wyckoff_position("a").multiplicity,)
    assert in_rhombohedral.multiplicities() == (rhombohedral.wyckoff_position("a").multiplicity,)
    assert in_standard.multiplicities() == (3,)
    assert in_rhombohedral.multiplicities() == (1,)


def test_a_supercell_transform_adds_the_missing_lattice_cosets() -> None:
    """A transform onto a larger cell needs points the standard orbit does not contain."""
    doubled = SettingTransform([["1/2", 0, 0], [0, 1, 0], [0, 0, 1]])
    assert len(doubled.lattice_cosets()) == 2

    asu = ASUStructure(
        [[11.28, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
        225,
        [ASUSite("a", NO_PARAMETERS, "Na")],
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
    asu = ASUStructure(CUBIC, 225, [ASUSite("a", NO_PARAMETERS, "Na")], _species("Na"), transform=shifted)

    assert asu.setting() is None
    assert not asu.is_standard_setting
    assert (F(1, 8), F(1, 8), F(1, 8)) in _fractions_of(asu)

    unshifted = ASUStructure(CUBIC, 225, [ASUSite("a", NO_PARAMETERS, "Na")], _species("Na"))
    # The same crystal, moved: same number of atoms, different coordinates.
    assert len(_fractions_of(asu)) == len(_fractions_of(unshifted))
    assert not same_crystal(asu, unshifted)


def test_a_tabulated_setting_reports_itself() -> None:
    asu = ASUStructure(
        [[5, 0, 0], [0, 6, 0], [0, 0, 7]],
        15,
        [ASUSite("e", FracVector.create(["1/3"]), "Si")],
        _species("Si"),
        transform=Spacegroup.for_setting("15:c1").transform_from_standard,
    )
    setting = asu.setting()
    assert setting is not None
    assert setting.setting == "15:c1"


# --- validation ---


def test_asu_structure_rejects_inconsistent_input() -> None:
    with pytest.raises(ValueError, match="unknown species"):
        ASUStructure(CUBIC, 225, [ASUSite("a", NO_PARAMETERS, "Xx")], _species("Na"))

    with pytest.raises(KeyError):
        ASUStructure(CUBIC, 225, [ASUSite("zz", NO_PARAMETERS, "Na")], _species("Na"))

    with pytest.raises(ValueError, match="free parameter"):
        # Wyckoff a of SG 225 is a fixed position and takes no parameters.
        ASUStructure(CUBIC, 225, [ASUSite("a", FracVector.create(["1/3"]), "Na")], _species("Na"))

    with pytest.raises(ValueError, match="unique"):
        ASUStructure(CUBIC, 225, [ASUSite("a", NO_PARAMETERS, "Na")], _species("Na") + _species("Na"))


def test_asu_structure_requires_the_standard_setting() -> None:
    """Wyckoff data is recorded against the standard setting; a difference is a transform."""
    with pytest.raises(ValueError, match="standard setting"):
        ASUStructure(
            CUBIC,
            Spacegroup.for_setting("15:c1"),
            [ASUSite("a", NO_PARAMETERS, "Na")],
            _species("Na"),
        )


# --- structure-family integration ---


def test_backend_dispatch_and_kind_override() -> None:
    asu = _rocksalt()
    backend = StructureBackend.create(asu)
    assert isinstance(backend, StructureASU)
    assert isinstance(StructureBackend.create(asu, kind="asu"), StructureASU)
    with pytest.raises(TypeError):
        StructureBackend.create(asu, kind="primitive")


def test_view_rewrap_identity_shared_backend_and_unwrap() -> None:
    asu = _rocksalt()
    backend = StructureBackend.create(asu)
    first = StructureSimpleView(backend)
    assert StructureSimpleView(first) is first
    second = StructureSimpleView(backend)
    assert first._backend is backend
    assert second._backend is backend
    assert unwrap(backend) is asu
    assert first.unwrap() is asu


def test_an_asu_structure_is_a_structure_everywhere() -> None:
    """Being in StructureLike is what lets it flow through the rest of the package."""
    view = StructureSimpleView(_rocksalt())
    assert isinstance(view, Structure)
    assert isinstance(view.cell, Cell)
    assert view.cartesian_sites().to_floats()[0] == [0.0, 0.0, 0.0]
    assert _rocksalt().to_structure() == view


# --- same_crystal ---


def test_same_crystal_ignores_order_and_lattice_translation() -> None:
    reference = StructureSimpleView(_rocksalt())
    reordered = Structure(
        reference.cell,
        list(reversed(reference.sites.reduced_coords.to_fractions())),
        reference.species,
        tuple(reversed(reference.species_at_sites)),
    )
    assert same_crystal(reference, reordered)

    # Shifting a site by a whole lattice vector names the same atom.
    translated_coords = [list(row) for row in reference.sites.reduced_coords.to_fractions()]
    translated_coords[0] = [value + 1 for value in translated_coords[0]]
    translated = Structure(reference.cell, translated_coords, reference.species, reference.species_at_sites)
    assert same_crystal(reference, translated)


def test_same_crystal_detects_real_differences() -> None:
    reference = StructureSimpleView(_rocksalt())

    different_cell = Structure(
        [[5.65, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
        reference.sites.reduced_coords,
        reference.species,
        reference.species_at_sites,
    )
    assert not same_crystal(reference, different_cell)

    swapped = Structure(
        reference.cell,
        reference.sites.reduced_coords,
        reference.species,
        tuple(reversed(reference.species_at_sites)),
    )
    assert not same_crystal(reference, swapped)

    # A doubled atom must not compare equal: the multiset counts repeats.
    coords = [list(row) for row in reference.sites.reduced_coords.to_fractions()]
    doubled = Structure(
        reference.cell,
        coords + [coords[0]],
        reference.species,
        reference.species_at_sites + (reference.species_at_sites[0],),
    )
    assert not same_crystal(reference, doubled)
