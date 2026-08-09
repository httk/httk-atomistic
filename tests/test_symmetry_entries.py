"""Tests for serving symmetry over OPTIMADE.

Two things are being checked. That the standard OPTIMADE symmetry properties carry the
right values — which for the symbols and Wyckoff letters means the values *for the setting
the structure is written in*, not for the standard setting the ASU stores them against.
And that the provider-specific properties keep the published `schemas.httk.org`
definitions they were taken from, rather than a local paraphrase.
"""

import fractions

import pytest
from httk.core import FracVector

from httk.atomistic import (
    WyckoffSite,
    ASUStructure,
    SettingTransform,
    Spacegroup,
    Species,
    UnitcellStructure,
    StructureEntryProvider,
)
from httk.atomistic.entries.symmetry import (
    SETTING_PROPERTY_KEYS,
    SYMMETRY_PROPERTY_KEYS,
    setting_definitions,
)

F = fractions.Fraction

NO_PARAMETERS = FracVector(())
CUBIC = [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]]


def _species(*names: str) -> list[Species]:
    return [Species(name=name, chemical_symbols=(name,), concentration=(1.0,)) for name in names]


def _record(structure: object) -> dict:
    provider = StructureEntryProvider({"x": structure})
    return next(iter(provider.records("structures")))


def _rocksalt() -> ASUStructure:
    return ASUStructure(
        CUBIC, 225, [WyckoffSite("a", NO_PARAMETERS, "Na"), WyckoffSite("b", NO_PARAMETERS, "Cl")], _species("Na", "Cl")
    )


# --- standard OPTIMADE properties ---


def test_standard_symmetry_properties_are_served_from_an_asu() -> None:
    record = _record(_rocksalt())

    assert record["space_group_it_number"] == 225
    assert record["space_group_symbol_hall"] == "-F 4 2 3"
    assert record["space_group_symbol_hermann_mauguin"] == "F m -3 m"
    assert record["space_group_symbol_hermann_mauguin_extended"].startswith("F 4/m -3 2/m")
    assert len(record["space_group_symmetry_operations_xyz"]) == 192
    assert record["space_group_symmetry_operations_xyz"][0].count(",") == 2
    assert record["wyckoff_positions"] == ["a", "b"]
    assert record["site_coordinate_span"] == "asymmetric_unit"
    assert len(record["fractional_site_positions"]) == 2


def test_the_extended_symbol_is_a_single_line() -> None:
    """International Tables prints it over several lines; a symbol property is one string."""
    record = _record(_rocksalt())
    assert "\n" not in record["space_group_symbol_hermann_mauguin_extended"]


def test_a_plain_structure_serves_null_symmetry() -> None:
    """A UnitcellStructure carries no symmetry, and inferring some would mean a hidden search."""
    structure = UnitcellStructure(CUBIC, [[0, 0, 0]], _species("Na"), ["Na"])
    record = _record(structure)

    for name in SYMMETRY_PROPERTY_KEYS:
        if name in ("fractional_site_positions", "site_coordinate_span", "space_group_symmetry_operations_xyz"):
            continue
        assert record[name] is None, name
    assert record["space_group_symmetry_operations_xyz"] == ["x,y,z"]
    for name in SETTING_PROPERTY_KEYS:
        assert record[name] is None, name

    # These two need no symmetry, so they are served regardless.
    assert record["fractional_site_positions"] == [[0.0, 0.0, 0.0]]
    assert record["site_coordinate_span"] == "unit_cell"


def test_an_entry_without_a_structure_serves_null_symmetry() -> None:
    record = _record(None)
    for name in (*SYMMETRY_PROPERTY_KEYS, *SETTING_PROPERTY_KEYS):
        assert record[name] is None, name


# --- the setting is what the symbols describe ---


def test_symbols_and_letters_describe_the_setting_the_structure_is_in() -> None:
    """A structure in ``15:c1`` is ``A 2/a``, not the standard setting's ``C 2/c``."""
    setting = Spacegroup.for_setting("15:c1")
    asu = ASUStructure(
        [[5, 0, 0], [0, 6, 0], [0, 0, 7]],
        15,
        [WyckoffSite("e", FracVector(["1/3"]), "Si")],
        _species("Si"),
        transform=setting.transform_from_standard,
    )
    record = _record(asu)

    assert record["space_group_it_number"] == 15
    assert record["space_group_symbol_hermann_mauguin"] == "A 2/a"
    assert record["space_group_symbol_hall"] == "-A 2a"
    assert record["_httk_setting_it_nc"] == "15:c1"
    assert record["_httk_is_reference_setting"] is False


def test_wyckoff_letters_are_translated_across_the_setting_boundary() -> None:
    """Setting ``224:1`` permutes letters ``i`` and ``j``; nothing else in the tables does.

    Reporting the stored standard-setting letter for a structure written in ``224:1`` would
    name a different position, and nothing would complain — which is exactly why this is
    tested rather than assumed.
    """
    setting = Spacegroup.for_setting("224:1")
    standard = Spacegroup.standard(224)
    asu = ASUStructure(
        [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
        224,
        [WyckoffSite("j", FracVector(["1/7"]), "O")],
        _species("O"),
        transform=setting.transform_from_standard,
    )
    record = _record(asu)

    reported = {symbol[-1] for symbol in record["wyckoff_positions"]}
    assert reported == {"i"}, "standard-setting j is setting 224:1's i"
    assert standard.wyckoff_position("j").free_count == setting.wyckoff_position("i").free_count


def test_wyckoff_multiplicity_follows_the_setting_too() -> None:
    """A 3a of the standard hexagonal cell is a 1a of the smaller rhombohedral one."""
    setting = Spacegroup.for_setting("166:R")
    asu = ASUStructure(
        [[4, 0, 0], [0, 4, 0], [0, 0, 12]],
        166,
        [WyckoffSite("a", NO_PARAMETERS, "Bi")],
        _species("Bi"),
        transform=setting.transform_from_standard,
    )
    record = _record(asu)

    assert Spacegroup.standard(166).wyckoff_position("a").multiplicity == 3
    assert record["wyckoff_positions"] == ["a"]
    assert len(record["fractional_site_positions"]) == 1


def test_an_untabulated_setting_serves_the_number_but_not_a_symbol() -> None:
    """Symbols and Wyckoff letters name a setting; this one has no name, so they are null.

    The space-group number and the change of basis are still meaningful and are served, so
    a client has everything it needs to reconstruct the structure.
    """
    shifted = SettingTransform(FracVector.eye((3, 3)), ["1/8", "1/8", "1/8"])
    asu = ASUStructure(CUBIC, 225, [WyckoffSite("a", NO_PARAMETERS, "Na")], _species("Na"), transform=shifted)
    record = _record(asu)

    assert record["space_group_it_number"] == 225
    assert record["space_group_symbol_hall"] is None
    assert record["space_group_symbol_hermann_mauguin"] is None
    assert record["wyckoff_positions"] is None
    assert record["space_group_symmetry_operations_xyz"] == [
        shifted.symop_to_setting(value).wrapped().to_xyz() for value in Spacegroup.standard(225).symmetry_operations
    ]
    assert record["_httk_setting_it_nc"] is None
    assert record["_httk_setting_transform"]["vector"] == ["1/8", "1/8", "1/8"]


def test_the_setting_transform_is_served_as_exact_rationals() -> None:
    """Rendered as strings, so the value survives JSON without becoming a float."""
    setting = Spacegroup.for_setting("15:c1")
    asu = ASUStructure(
        [[5, 0, 0], [0, 6, 0], [0, 0, 7]],
        15,
        [WyckoffSite("e", FracVector(["1/3"]), "Si")],
        _species("Si"),
        transform=setting.transform_from_standard,
    )
    transform = _record(asu)["_httk_setting_transform"]

    assert transform["matrix"] == [["0", "0", "1"], ["1", "0", "0"], ["0", "1", "0"]]
    assert transform["vector"] == ["0", "0", "0"]
    assert transform["xyz"] == "z,x,y"


# --- the provider-specific definitions ---


def test_provider_specific_properties_keep_their_published_definitions() -> None:
    """The whole point of reusing schemas.httk.org rather than paraphrasing it."""
    definitions = setting_definitions()
    assert set(definitions) == set(SETTING_PROPERTY_KEYS)

    for name, definition in definitions.items():
        document = definition.as_optimade()
        assert document["$id"].startswith("https://schemas.httk.org/defs/v0.1/properties/")
        assert document["description"]
        # The served name carries the prefix OPTIMADE requires of a database-specific
        # property; the definition it points at is the published one.
        assert name.startswith("_httk_")


def test_the_served_definition_describes_every_served_property() -> None:
    """A property served but not described is what OPTIMADE forbids, so it is checked."""
    provider = StructureEntryProvider({"x": _rocksalt()})
    described = provider.entry_types()["structures"].properties
    record = next(iter(provider.records("structures")))

    for name in (*SYMMETRY_PROPERTY_KEYS, *SETTING_PROPERTY_KEYS):
        assert name in described, name
        assert name in provider.property_keys("structures")
        assert name in record


def test_the_definition_shape_does_not_depend_on_the_contents() -> None:
    """A database of plain structures describes the same properties as one of ASUs."""
    plain = StructureEntryProvider({"x": UnitcellStructure(CUBIC, [[0, 0, 0]], _species("Na"), ["Na"])})
    symmetric = StructureEntryProvider({"x": _rocksalt()})
    assert set(plain.entry_types()["structures"].properties) == set(symmetric.entry_types()["structures"].properties)


def test_custom_extra_definitions_still_work_alongside() -> None:
    from httk.core import PropertyDefinition

    energy = PropertyDefinition.from_simple("_httk_total_energy", description="E", fulltype="float")
    provider = StructureEntryProvider(
        {"x": _rocksalt()},
        extra_definitions={"_httk_total_energy": energy},
        properties={"x": {"_httk_total_energy": -1.5}},
    )
    described = provider.entry_types()["structures"].properties
    assert "_httk_total_energy" in described
    assert "_httk_setting_it_nc" in described


def test_a_collision_with_a_served_symmetry_property_is_rejected() -> None:
    """Redefining one of these would silently change what the server promises."""
    from httk.core import PropertyDefinition

    clashing = PropertyDefinition.from_simple("_httk_setting_it_nc", description="x", fulltype="string")
    with pytest.raises(ValueError, match="already defined"):
        StructureEntryProvider({"x": _rocksalt()}, extra_definitions={"_httk_setting_it_nc": clashing})


# --- precision ---


def test_precision_is_served_from_the_structure() -> None:
    """So a client can derive its own tolerance instead of guessing one."""
    from httk.atomistic import Cell

    asu = ASUStructure(
        Cell(CUBIC, 1, F(1, 1000)),
        225,
        [WyckoffSite("a", NO_PARAMETERS, "Na")],
        _species("Na"),
        coordinate_precision=F(1, 10000),
    )
    record = _record(asu)
    assert record["_httk_coordinate_precision"] == pytest.approx(1e-4)
    assert record["_httk_basis_precision"] == pytest.approx(1e-3)


def test_a_structure_that_states_no_precision_serves_null() -> None:
    """Distinguishable from a claim of exactness, which is the point of using null."""
    record = _record(UnitcellStructure(CUBIC, [[0, 0, 0]], _species("Na"), ["Na"]))
    assert record["_httk_coordinate_precision"] is None
    assert record["_httk_basis_precision"] is None


def test_the_precision_definitions_are_the_published_ones() -> None:
    from httk.atomistic.entries.precision import precision_definitions

    definitions = precision_definitions()
    assert set(definitions) == {"_httk_coordinate_precision", "_httk_basis_precision"}

    coordinate = definitions["_httk_coordinate_precision"].as_optimade()
    basis = definitions["_httk_basis_precision"].as_optimade()
    assert coordinate["$id"].endswith("/core/fractional_coordinate_precision")
    assert basis["$id"].endswith("/core/length_precision")

    # A fractional coordinate is dimensionless; a cell edge is a length. That is why these
    # are two definitions rather than one.
    assert coordinate["x-optimade-unit"] == "inapplicable"
    assert basis["x-optimade-unit"] == "angstrom"
