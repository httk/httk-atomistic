"""Tests for recognizing the asymmetric unit of a full structure.

The headline property is the round trip: ``ASUStructure -> UnitcellStructure -> ASUStructure ->
UnitcellStructure`` reproduces the same crystal, exactly, with no numerical drift. It is asserted
here for centred lattices, special and general positions, non-standard settings, and a
volume-changing transform.

The one asymmetry to keep in view: expansion is lossless, recognition is not. Recognition
snaps a measured structure onto an idealised symmetric one, so ``recognize -> expand``
returns the input only when the input was already exact.
"""

import fractions
import sys

import pytest
from httk.core import FracVector, unwrap

from httk.atomistic import (
    ASUStructure,
    ASUStructureView,
    Cell,
    SettingTransform,
    Sites,
    Spacegroup,
    Species,
    UnitcellStructure,
    UnitcellStructureView,
    WyckoffSite,
    recognize_asu,
    same_crystal,
)
from httk.atomistic.symmetry import recognition as recognition_module

F = fractions.Fraction

NO_PARAMETERS = FracVector(())
CUBIC = [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]
ORTHO = [[5, 0, 0], [0, 6, 0], [0, 0, 7]]
HEXAGONAL = [[4, 0, 0], [0, 4, 0], [0, 0, 12]]


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _rocksalt() -> ASUStructure:
    return ASUStructure(
        CUBIC, 225, [WyckoffSite("a", NO_PARAMETERS, "Na"), WyckoffSite("b", NO_PARAMETERS, "Cl")], _species("Na", "Cl")
    )


def _monoclinic(setting: str = "15:b1") -> ASUStructure:
    """One silicon on the one-parameter Wyckoff position e of SG 15."""
    spacegroup = Spacegroup.from_setting(setting)
    return ASUStructure(
        ORTHO,
        15,
        [WyckoffSite("e", FracVector(["1/3"]), "Si")],
        _species("Si"),
        transform=spacegroup.transform_from_standard,
    )


# --- the round trip ---


@pytest.mark.parametrize(
    "build",
    [_rocksalt, _monoclinic, lambda: _monoclinic("15:c1"), lambda: _monoclinic("15:-b2")],
    ids=["rocksalt-Fm-3m", "SG15-standard", "SG15-c1", "SG15-origin-shifted"],
)
def test_round_trip_reproduces_the_crystal_exactly(build: object) -> None:
    """The guarantee, for a centred lattice and for three settings of one group."""
    original = build()  # type: ignore[operator]
    expanded = UnitcellStructureView(original)

    recovered = recognize_asu(expanded, setting=original.setting())
    rebuilt = UnitcellStructureView(recovered)

    assert same_crystal(expanded, rebuilt)
    # Stronger than same_crystal: the coordinates are identical, not merely equivalent.
    assert expanded.sites.reduced_coords == rebuilt.sites.reduced_coords
    assert expanded.species_at_sites == rebuilt.species_at_sites


def test_round_trip_through_a_volume_changing_transform() -> None:
    """Rhombohedral axes: the standard orbit is three times too large for this cell."""
    rhombohedral = Spacegroup.from_setting("166:R")
    original = ASUStructure(
        HEXAGONAL,
        166,
        [WyckoffSite("a", NO_PARAMETERS, "Bi")],
        _species("Bi"),
        transform=rhombohedral.transform_from_standard,
    )
    expanded = UnitcellStructureView(original)
    assert len(expanded.sites) == 1

    recovered = recognize_asu(expanded, setting=rhombohedral)
    assert recovered.multiplicities() == (1,)
    assert expanded.sites.reduced_coords == UnitcellStructureView(recovered).sites.reduced_coords


def test_recognition_places_every_image_in_a_supercell_coset() -> None:
    """A supercell image maps back to its standard Wyckoff site."""
    transform = SettingTransform([["1/2", 0, 0], [0, 1, 0], [0, 0, 1]])
    original = ASUStructure(
        [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        1,
        [WyckoffSite("a", FracVector((0, 0, 0)), "H")],
        _species("H"),
        transform=transform,
    )
    expanded = UnitcellStructureView(original)
    assert expanded.sites.reduced_coords.to_fractions() == [[F(0), F(0), F(0)], [F(1, 2), F(0), F(0)]]

    recovered = recognize_asu(expanded, standard=Spacegroup.standard(1), transform=transform)

    assert [(site.wyckoff, site.species) for site in recovered.wyckoff_sites] == [("a", "H")]
    assert recovered.multiplicities() == (2,)
    assert expanded.sites.reduced_coords == UnitcellStructureView(recovered).sites.reduced_coords


def test_recognition_rejects_duplicate_supercell_cosets() -> None:
    """Two copies in one coset cannot stand in for one site in each coset."""
    transform = SettingTransform([["1/2", 0, 0], [0, 1, 0], [0, 0, 1]])
    duplicated = UnitcellStructure(
        [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 0, 0]],
        _species("H"),
        ("H", "H"),
    )

    with pytest.raises(ValueError, match="does not occupy each generated position exactly once"):
        recognize_asu(duplicated, standard=Spacegroup.standard(1), transform=transform)


def test_tolerance_cap_keeps_split_sites_off_the_same_special_position() -> None:
    """The half-separation cap must be strict after Cartesian float arithmetic."""
    split = UnitcellStructure(
        Cell(((5, 0, 0), (0, F(820817, 25000), 0), (0, 0, 7))),
        Sites(
            ((F(123, 1000), F(493, 100000), F(234, 1000)), (F(123, 1000), F(99507, 100000), F(234, 1000))),
            F(1, 100),
        ),
        _species("H"),
        ("H", "H"),
    )

    recovered = recognize_asu(split, standard=Spacegroup.standard(6), transform=SettingTransform.identity())

    assert recovered.multiplicities() == (2,)
    assert len(UnitcellStructureView(recovered).sites) == 2


def test_round_trip_through_an_untabulated_setting() -> None:
    """A setting in no table round-trips like any other, given its transform."""
    shifted = SettingTransform(FracVector.eye((3, 3)), ["1/8", "1/8", "1/8"])
    original = ASUStructure(CUBIC, 225, [WyckoffSite("a", NO_PARAMETERS, "Na")], _species("Na"), transform=shifted)
    expanded = UnitcellStructureView(original)

    recovered = recognize_asu(expanded, standard=Spacegroup.standard(225), transform=shifted)
    assert recovered.setting() is None
    assert expanded.sites.reduced_coords == UnitcellStructureView(recovered).sites.reduced_coords


def test_recognition_recovers_the_original_wyckoff_description() -> None:
    recovered = recognize_asu(UnitcellStructureView(_rocksalt()), setting=Spacegroup.standard(225))
    assert [(site.wyckoff, site.species) for site in recovered.wyckoff_sites] == [("a", "Na"), ("b", "Cl")]
    assert recovered.multiplicities() == (4, 4)

    monoclinic = recognize_asu(UnitcellStructureView(_monoclinic()), setting=Spacegroup.standard(15))
    assert monoclinic.wyckoff_sites[0].wyckoff == "e"
    assert monoclinic.wyckoff_sites[0].free_params == FracVector([F(1, 3)])


# --- tolerance, and the lossy direction ---


def _perturbed(structure: UnitcellStructure, amount: F) -> UnitcellStructure:
    """The same structure with every coordinate nudged, as measured data would be."""
    coords = [
        [value + amount * F(index * 3 + axis + 1) for axis, value in enumerate(row)]
        for index, row in enumerate(structure.sites.reduced_coords.to_fractions())
    ]
    return UnitcellStructure(structure.cell, coords, structure.species, structure.species_at_sites)


def test_a_measured_structure_is_snapped_onto_its_symmetry() -> None:
    """Every orbit member carries its own rounding, and they must still group as one orbit.

    Exact orbit membership would put each atom in an orbit of its own and report the
    structure as having no symmetry, so membership is tested within the tolerance.
    """
    exact = UnitcellStructureView(_monoclinic())
    noisy = _perturbed(exact, F(1, 100000))

    recovered = recognize_asu(noisy, setting=Spacegroup.standard(15))
    assert len(recovered.wyckoff_sites) == 1
    assert recovered.multiplicities() == (4,)

    rebuilt = UnitcellStructureView(recovered)
    assert len(rebuilt.sites) == len(noisy.sites)
    # The fixed components of the position are restored exactly; only the free parameter
    # keeps the measured value.
    for row in rebuilt.sites.reduced_coords.to_fractions():
        assert row[0] in (F(0), F(1, 2))
        assert row[2] in (F(1, 4), F(3, 4))


def test_recognition_is_lossy_and_expansion_is_not() -> None:
    """The documented asymmetry, asserted rather than just described."""
    exact = UnitcellStructureView(_monoclinic())
    noisy = _perturbed(exact, F(1, 100000))

    once = UnitcellStructureView(recognize_asu(noisy, setting=Spacegroup.standard(15)))
    # Recognition moved the atoms, so the result is not the noisy input.
    assert not same_crystal(noisy, once)

    # But it is now exact, so recognizing and expanding again changes nothing at all.
    twice = UnitcellStructureView(recognize_asu(once, setting=Spacegroup.standard(15)))
    assert once.sites.reduced_coords == twice.sites.reduced_coords
    assert same_crystal(once, twice)


def test_limit_denominator_idealises_free_parameters_on_request() -> None:
    """Turning 0.3333 into 1/3 is a claim about the data, so it is opt-in."""
    exact = UnitcellStructureView(_monoclinic())
    noisy = _perturbed(exact, F(1, 100000))

    faithful = recognize_asu(noisy, setting=Spacegroup.standard(15))
    assert faithful.wyckoff_sites[0].free_params.to_fractions()[0] != F(1, 3)

    idealised = recognize_asu(noisy, setting=Spacegroup.standard(15), limit_denominator=12)
    assert idealised.wyckoff_sites[0].free_params == FracVector([F(1, 3)])


def test_a_structure_without_the_claimed_symmetry_is_rejected() -> None:
    """Silence here would mean quietly dropping atoms."""
    structure = UnitcellStructureView(_rocksalt())
    coords = [list(row) for row in structure.sites.reduced_coords.to_fractions()]
    # Move a single atom well off its position, breaking the orbit.
    coords[1] = [F(3, 7), F(1, 5), F(2, 9)]
    broken = UnitcellStructure(structure.cell, coords, structure.species, structure.species_at_sites)

    with pytest.raises(ValueError, match="not symmetric|does not lie"):
        recognize_asu(broken, setting=Spacegroup.standard(225))


def test_recognize_asu_rejects_contradictory_arguments() -> None:
    structure = UnitcellStructureView(_rocksalt())
    with pytest.raises(TypeError):
        recognize_asu(structure, setting=Spacegroup.standard(225), transform=SettingTransform.identity())
    with pytest.raises(TypeError):
        recognize_asu(structure, standard=Spacegroup.standard(225))
    with pytest.raises(ValueError, match="standard setting"):
        recognize_asu(structure, standard=Spacegroup.from_setting("15:c1"), transform=SettingTransform.identity())


# --- the view ---


def test_view_adopts_an_existing_asu_without_recognizing_anything() -> None:
    """A backend that already holds an ASU is passed through, exactly and with no tolerance."""
    original = _monoclinic()
    view = ASUStructureView(original)
    assert view.wyckoff_sites == original.wyckoff_sites
    assert view.transform == original.transform
    assert unwrap(view) is original
    assert view.unview() is original


def test_view_rewrap_identity_and_unwrap() -> None:
    view = ASUStructureView(_rocksalt())
    assert ASUStructureView(view) is view
    configured = ASUStructureView(view, tolerance=0.01)
    assert configured is not view
    assert configured._tolerance == 0.01
    assert isinstance(view, ASUStructure)


def test_view_recognizes_a_plain_structure() -> None:
    expanded = UnitcellStructureView(_rocksalt())
    view = ASUStructureView(expanded, setting=Spacegroup.standard(225))
    assert [site.wyckoff for site in view.wyckoff_sites] == ["a", "b"]
    assert same_crystal(expanded, UnitcellStructureView(view))


# --- spglib ---


def test_spglib_transform_keeps_a_data_derived_origin_shift() -> None:
    """An arbitrary origin is not rounded onto the small crystallographic-fraction grid."""
    operation = recognition_module._exact_operation(
        [[1.0, 0.0, 0.0], [0.0, 1.0 / 3.0, 0.0], [0.0, 0.0, 1.0]],
        [1.0 / 3.0, 0.96857143, 0.0],
    )

    # True setting constants retain their clean exact value, while an atom-selected origin keeps
    # the measured decimal rather than being displaced to the closest denominator <= 48.
    assert operation.matrix[1][1] == F(1, 3)
    assert operation.vector[0] == F(1, 3)
    assert operation.vector[1] == F(96857143, 100000000)


def test_asu_view_no_arguments_uses_spglib() -> None:
    pytest.importorskip("spglib")
    expanded = UnitcellStructureView(_rocksalt())
    structure = UnitcellStructure(
        expanded.cell, expanded.sites.reduced_coords, expanded.species, expanded.species_at_sites
    )

    view = ASUStructureView(structure)

    assert view.spacegroup.it_number == 225
    assert same_crystal(structure, UnitcellStructureView(view))


def test_asu_view_without_spglib_raises_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    expanded = UnitcellStructureView(_rocksalt())
    structure = UnitcellStructure(
        expanded.cell, expanded.sites.reduced_coords, expanded.species, expanded.species_at_sites
    )
    monkeypatch.setitem(sys.modules, "spglib", None)

    with pytest.raises(ImportError, match=r"spglib.*httk-atomistic\[default\]"):
        deferred = ASUStructureView(structure)
        _ = deferred.wyckoff_sites


def test_spglib_finds_the_symmetry_of_a_structure_that_carries_none() -> None:
    """The only path that needs spglib: a bare cell and site list."""
    pytest.importorskip("spglib")
    expanded = UnitcellStructureView(_rocksalt())

    recovered = recognize_asu(expanded)
    assert recovered.spacegroup.it_number == 225
    assert [site.wyckoff for site in recovered.wyckoff_sites] == ["a", "b"]
    assert same_crystal(expanded, UnitcellStructureView(recovered))


@pytest.mark.parametrize("it_number", [227, 70, 141, 225])
def test_spglib_bridges_the_two_origin_groups_correctly(it_number: int) -> None:
    """spglib's default setting is not the IT standard one for the two-origin groups.

    Assuming they coincide would displace the structure by a fraction of a cell while still
    passing a symmetry check, so the bridge is asserted on the groups where it matters.
    """
    pytest.importorskip("spglib")
    standard = Spacegroup.standard(it_number)
    letter = next(position.letter for position in standard.wyckoff if position.free_count == 0)
    original = ASUStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]] if standard.crystal_system == "cubic" else ORTHO,
        it_number,
        [WyckoffSite(letter, NO_PARAMETERS, "C")],
        _species("C"),
    )
    expanded = UnitcellStructureView(original)

    recovered = recognize_asu(expanded)
    assert same_crystal(expanded, UnitcellStructureView(recovered))


def test_spglib_result_prefers_the_standard_setting_when_the_structure_is_in_it() -> None:
    """spglib fixes the group but not the frame; the tidier of the valid answers is chosen."""
    pytest.importorskip("spglib")
    original = ASUStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]], 227, [WyckoffSite("a", NO_PARAMETERS, "C")], _species("C")
    )
    recovered = recognize_asu(UnitcellStructureView(original))
    assert recovered.spacegroup.it_number == 227
    assert recovered.is_standard_setting
    assert recovered.wyckoff_sites[0].wyckoff == "a"
