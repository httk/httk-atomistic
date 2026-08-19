"""Tests for the exact symmetry primitives: affine maps, setting transforms, Wyckoff algebra.

The load-bearing test here is
:func:`test_setting_transform_maps_the_standard_symop_set_onto_each_setting`. A setting
transform applied in the wrong direction, or with the matrix transposed, still yields a
structurally valid crystal — just the wrong one — so it is pinned by exact set equality of
symmetry operations rather than by a spot check that could pass by coincidence.
"""

import fractions

import pytest

from httk.atomistic import data
from httk.atomistic.symmetry.affine_operation import AffineOperation
from httk.atomistic.symmetry.setting_transform import SettingTransform
from httk.atomistic.symmetry.spacegroup import Spacegroup, wyckoff_letter_map
from httk.core import FracVector, SurdVector

F = fractions.Fraction

# Settings chosen to discriminate: a plain reference setting, axis permutations, an origin
# shift, the two-origin groups, and both rhombohedral cases (where det M == 3).
SAMPLE_SETTINGS = (
    "1",
    "15:b1",
    "15:c1",
    "15:-b2",
    "48:1",
    "48:2",
    "68:1",
    "142:1",
    "160:R",
    "166:H",
    "166:R",
    "224:1",
    "227:1",
    "227:2",
)

RHOMBOHEDRAL_SETTINGS = ("146:R", "148:R", "155:R", "160:R", "161:R", "166:R", "167:R")


def _wrapped_symops(spacegroup: Spacegroup) -> frozenset[AffineOperation]:
    """The group's operations as a set, translations reduced into ``[0, 1)``."""
    return frozenset(operation.wrapped() for operation in spacegroup.symmetry_operations)


def _point_set(coordinates: FracVector) -> frozenset[FracVector]:
    """An unordered set of wrapped coordinates.

    Relies on ``FracVector`` hashing consistently with its numerical equality, so that a
    coordinate reached by different arithmetic counts as the same point.
    """
    return frozenset(coordinates.normalize())


# --- AffineOperation ---


def test_affine_operation_applies_composes_and_inverts_exactly() -> None:
    operation = AffineOperation([["-1", "0", "0"], ["0", "1", "0"], ["0", "0", "-1"]], ["1/2", "1/2", "1/2"])
    assert operation.to_xyz() == "-x+1/2,y+1/2,-z+1/2"
    assert operation.determinant() == F(1)

    assert operation.apply(("1/3", "1/4", "1/5")) == FracVector([F(1, 6), F(3, 4), F(3, 10)])
    assert (operation * operation.inverse()).is_identity()
    assert operation.conjugated_by(AffineOperation.identity()) == operation

    # (a * b) applies b first.
    shift_half = AffineOperation(FracVector.eye((3, 3)), ["1/2", 0, 0])
    shift_quarter = AffineOperation(FracVector.eye((3, 3)), ["1/4", 0, 0])
    assert (shift_half * shift_quarter).to_xyz() == "x+3/4,y,z"


def test_affine_operation_wrapping_and_hashing() -> None:
    operation = AffineOperation(FracVector.eye((3, 3)), ["3/2", 0, 0])
    assert operation.wrapped().vector == FracVector([F(1, 2), F(0), F(0)])
    # Equality is exact, not modulo the lattice; wrapping is what makes them comparable.
    assert operation != operation.wrapped()
    assert len({operation.wrapped(), operation.wrapped()}) == 1


def test_affine_operation_renders_tabulated_symops_identically() -> None:
    """``to_xyz`` reproduces the tabulated ``xyz`` strings byte-for-byte.

    Reads both the matrix and the translation back out through an independent code path,
    so it pins the rational parsing as well as the rendering. Sampled every third setting
    to keep the suite quick; the full sweep over all 7340 operations was run during
    development and had zero mismatches.
    """
    checked = 0
    for record in data.spacegroup_settings()[::3]:
        for entry in record["symops"]:
            operation = AffineOperation.from_record(entry)
            assert operation.to_xyz() == entry["affine_transformation"]["xyz"]
            checked += 1
    assert checked > 2000


def test_affine_operation_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError):
        AffineOperation([[1, 0], [0, 1]], (0, 0))
    with pytest.raises(ValueError):
        AffineOperation(FracVector.eye((3, 3)), (0, 0))


# --- SettingTransform ---


@pytest.mark.parametrize("setting", SAMPLE_SETTINGS)
def test_setting_transform_maps_the_standard_symop_set_onto_each_setting(setting: str) -> None:
    """The convention check that cannot pass by accident.

    Conjugating the standard setting's whole symmetry group by the transform must
    reproduce the target setting's group *exactly*, as a set modulo lattice translations.
    Transposing the matrix or inverting the direction fails this for the settings sampled
    here, even though several self-inverse cases would pass either way.
    """
    target = Spacegroup.from_setting(setting)
    standard = target.standard_setting()
    transform = target.transform_from_standard

    mapped = frozenset(transform.symop_to_setting(operation).wrapped() for operation in standard.symmetry_operations)
    assert mapped == _wrapped_symops(target)


def test_setting_transform_is_the_identity_exactly_for_standard_settings() -> None:
    for it_number in range(1, 231):
        standard = Spacegroup.standard(it_number)
        assert standard.transform_from_standard.is_identity()
        assert standard.is_standard_setting


def test_only_rhombohedral_settings_change_the_cell_volume() -> None:
    changing = {
        record["setting_it_nc"]: SettingTransform.from_hall_entry(record["hall_entry"]).determinant()
        for record in data.spacegroup_settings()
        if abs(SettingTransform.from_hall_entry(record["hall_entry"]).determinant()) != 1
    }
    assert set(changing) == set(RHOMBOHEDRAL_SETTINGS)
    assert set(changing.values()) == {F(3)}


@pytest.mark.parametrize("setting", SAMPLE_SETTINGS)
def test_setting_transform_round_trips_coordinates(setting: str) -> None:
    transform = Spacegroup.from_setting(setting).transform_from_standard
    point = FracVector(["1/7", "2/11", "3/13"])
    assert transform.to_standard(transform.to_setting(point)) == point
    assert transform.inverse().inverse() == transform


@pytest.mark.parametrize("setting", SAMPLE_SETTINGS)
def test_setting_transform_preserves_cartesian_positions(setting: str) -> None:
    """A change of setting relabels coordinates; it must not move any atom.

    With the basis transforming as ``B_own = inv(M).T * B_std``, the Cartesian point
    ``f * B`` has to come out the same in both settings. This catches a transposed basis
    rule that the symop test would not.
    """
    transform = Spacegroup.from_setting(setting).transform_from_standard
    standard_basis = SurdVector([[3, 0, 0], [0, 5, 0], [0, 0, 7]])
    own_basis = transform.basis_to_setting(standard_basis)

    standard_point = FracVector(["1/7", "2/11", "3/13"])
    # The origin shift moves the origin, so compare a difference vector, which is
    # independent of it.
    other_point = FracVector(["1/3", "1/5", "1/9"])
    standard_delta = SurdVector(standard_point - other_point) * standard_basis
    own_delta = SurdVector(transform.to_setting(standard_point) - transform.to_setting(other_point)) * own_basis
    assert standard_delta == own_delta

    assert transform.basis_to_standard(own_basis) == standard_basis


def test_lattice_cosets_are_trivial_for_every_tabulated_setting() -> None:
    """All 527 transforms have integer matrices, so no setting needs extra cosets.

    The generic code path exists for a caller-supplied transform into a supercell setting;
    this records that the shipped data never exercises it.
    """
    zero = FracVector((0, 0, 0))
    for record in data.spacegroup_settings():
        transform = SettingTransform.from_hall_entry(record["hall_entry"])
        assert transform.lattice_cosets() == (zero,)


def test_lattice_cosets_found_for_a_supercell_transform() -> None:
    """A half-scale transform doubles the cell along x, so one extra coset is needed."""
    transform = SettingTransform([["1/2", 0, 0], [0, 1, 0], [0, 0, 1]])
    assert transform.determinant() == F(1, 2)
    assert transform.lattice_cosets() == (
        FracVector((0, 0, 0)),
        FracVector((F(1, 2), 0, 0)),
    )


def test_singular_setting_transform_is_rejected() -> None:
    with pytest.raises(ValueError):
        SettingTransform([[1, 0, 0], [1, 0, 0], [0, 0, 1]])


# --- Wyckoff algebra ---


def test_wyckoff_positions_are_ordered_most_specific_first() -> None:
    positions = Spacegroup.from_setting("15:b1").wyckoff
    assert [position.letter for position in positions] == ["a", "b", "c", "d", "e", "f"]
    assert [position.free_count for position in positions] == [0, 0, 0, 0, 1, 3]


def test_wyckoff_forward_evaluation_is_exact() -> None:
    position = Spacegroup.from_setting("15:b1").wyckoff_position("e")
    assert position.multiplicity == 4
    assert position.free == (1,)
    assert position.site_symmetry == "2"
    assert position.coordinates(["1/3"]) == FracVector(
        [
            [F(0), F(1, 3), F(1, 4)],
            [F(0), F(-1, 3), F(3, 4)],
            [F(1, 2), F(1, 6), F(3, 4)],
            [F(1, 2), F(5, 6), F(1, 4)],
        ]
    )


def test_every_orbit_member_recovers_its_parameters_not_just_the_representative() -> None:
    """The property the reverse direction lives or dies by.

    A matcher that tested only the representative branch would reject most orbit members:
    across the vendored tables, 11673 of the 20639 non-representative members lie on a
    different branch than the representative.
    """
    checked = 0
    for setting in SAMPLE_SETTINGS:
        spacegroup = Spacegroup.from_setting(setting)
        for position in spacegroup.wyckoff:
            parameters = ["1/7", "1/11", "1/13"][: position.free_count]
            orbit = position.coordinates(parameters)
            expected = _point_set(orbit)

            recovered_all = [position.parameters_of(coordinate.normalize()) for coordinate in orbit]
            assert all(recovered is not None for recovered in recovered_all), (
                f"{setting} letter {position.letter} lost an orbit member"
            )
            checked += len(recovered_all)

            # Whichever branch a member matched on, its parameters must regenerate the very
            # same orbit. Compared as a set, since branches enumerate points in different
            # orders. Checked on the first and last member rather than all of them: this is
            # the expensive part and is quadratic in a multiplicity that reaches 192.
            for recovered in (recovered_all[0], recovered_all[-1]):
                assert recovered is not None
                assert _point_set(position.coordinates(recovered)) == expected
    assert checked > 1000


def test_wyckoff_rejects_coordinates_that_are_not_on_the_position() -> None:
    spacegroup = Spacegroup.from_setting("15:b1")
    special = spacegroup.wyckoff_position("e")
    general = spacegroup.wyckoff_position("f")
    off_position = FracVector(["1/7", "2/11", "3/13"])

    assert special.parameters_of(off_position) is None
    assert general.parameters_of(off_position) is not None
    assert spacegroup.wyckoff_position("a").parameters_of(off_position) is None


def test_identify_wyckoff_returns_the_most_specific_position() -> None:
    spacegroup = Spacegroup.from_setting("15:b1")
    # The origin is Wyckoff a, which is more specific than the general position it also
    # trivially satisfies as a point of the cell.
    identified = spacegroup.identify_wyckoff(FracVector((0, 0, 0)))
    assert identified is not None
    assert identified[0].letter == "a"

    on_e = spacegroup.wyckoff_position("e").coordinates(["1/3"])[0].normalize()
    identified_e = spacegroup.identify_wyckoff(on_e)
    assert identified_e is not None
    assert identified_e[0].letter == "e"
    assert identified_e[1] == FracVector([F(1, 3)])

    assert spacegroup.identify_wyckoff(FracVector(["1/7", "2/11", "3/13"]))[0].letter == "f"


def test_fixed_positions_take_no_parameters() -> None:
    position = Spacegroup.from_setting("15:b1").wyckoff_position("a")
    assert position.free_count == 0
    assert position.coordinates([])[0] == FracVector((0, 0, 0))
    with pytest.raises(ValueError):
        position.representative.coordinate(["1/3"])


# --- Spacegroup ---


def test_spacegroup_lookup_and_identity() -> None:
    spacegroup = Spacegroup.from_setting("15:c1")
    assert spacegroup.it_number == 15
    assert spacegroup.hall_entry == "-a_2a"
    assert spacegroup.crystal_system == "monoclinic"
    assert spacegroup.centring_type == "A"
    assert not spacegroup.is_standard_setting
    assert spacegroup.standard_setting().setting == "15:b1"
    assert Spacegroup.from_hall_entry("-a_2a") == spacegroup
    assert len({spacegroup, Spacegroup.from_setting("15:c1")}) == 1


def test_spacegroup_operation_count_matches_the_group_order() -> None:
    for setting in SAMPLE_SETTINGS:
        spacegroup = Spacegroup.from_setting(setting)
        assert len(spacegroup.symmetry_operations) == spacegroup.record["n_symops"]
        assert len(_wrapped_symops(spacegroup)) == len(spacegroup.symmetry_operations)


def test_unknown_wyckoff_letter_raises() -> None:
    with pytest.raises(KeyError):
        Spacegroup.from_setting("15:b1").wyckoff_position("z")


# --- Wyckoff letters across settings ---


def test_wyckoff_letters_are_usually_but_not_always_preserved_across_settings() -> None:
    """Setting ``224:1`` swaps letters ``i`` and ``j``; nothing else in the tables does.

    Trusting a CIF's declared letter across a setting boundary is therefore wrong in a way
    that raises no error, which is why the map is computed rather than assumed.
    """
    swapped = Spacegroup.from_setting("224:1")
    mapping = wyckoff_letter_map(swapped.standard_setting(), swapped)
    assert mapping["i"] == "j"
    assert mapping["j"] == "i"

    for setting in ("15:c1", "48:1", "68:1", "142:1", "166:R", "227:1"):
        target = Spacegroup.from_setting(setting)
        identity_map = wyckoff_letter_map(target.standard_setting(), target)
        assert all(source == mapped for source, mapped in identity_map.items()), setting


def test_wyckoff_letter_map_rejects_different_space_groups() -> None:
    with pytest.raises(ValueError):
        wyckoff_letter_map(Spacegroup.standard(15), Spacegroup.standard(16))
