"""The symmetry properties an asymmetric-unit structure can serve over OPTIMADE.

Two families, kept apart on purpose:

**Standard OPTIMADE properties.** ``space_group_it_number``, the three space-group
symbols, ``space_group_symmetry_operations_xyz``, ``wyckoff_positions``,
``fractional_site_positions``, and ``site_coordinate_span`` are all part of the OPTIMADE
standard and are already described in the vendored ``structures.json``. They
were simply never served.

**Provider-specific properties**, for the things OPTIMADE does not standardise — chiefly
*which setting* a structure is written in, and the change of basis to the standard one.
Rather than inventing descriptions, these reuse the published property definitions from
`schemas.httk.org <https://schemas.httk.org>`_, vendored verbatim in the registry schema package. The served *name* carries the ``_httk_`` prefix that OPTIMADE requires
of a database-specific property, but the definition keeps its own ``$id``, so a client
following the link reaches the authoritative published schema instead of a local
paraphrase.

Everything here serves ``null`` for a plain :class:`~httk.atomistic.UnitcellStructure`, which
carries no symmetry — with the sole exception of ``fractional_site_positions``, which is
just the reduced coordinates and is always available.
"""

from functools import cache
from typing import Any

from httk.core import PropertyDefinition, unwrap

from httk.atomistic.entries.definitions import load_httk_definitions
from httk.atomistic.models.structure.asu import FundamentalDomainStructure
from httk.atomistic.symmetry.spacegroup import wyckoff_letter_map

__all__ = [
    "SETTING_PROPERTY_KEYS",
    "SYMMETRY_PROPERTY_KEYS",
    "setting_definitions",
    "symmetry_properties",
]

#: Standard OPTIMADE symmetry properties, served from an ASU when there is one. Already
#: described by the vendored ``structures.json``, so they need no definition of their own.
SYMMETRY_PROPERTY_KEYS: tuple[str, ...] = (
    "space_group_it_number",
    "space_group_symbol_hall",
    "space_group_symbol_hermann_mauguin",
    "space_group_symbol_hermann_mauguin_extended",
    "space_group_symmetry_operations_xyz",
    "wyckoff_positions",
    "fractional_site_positions",
    "site_coordinate_span",
    "site_coordinate_span_description",
)

#: Provider-specific properties describing *which setting* a structure is written in,
#: mapped to the vendored definition file that describes each. The served name carries the
#: ``_httk_`` prefix; the definition keeps its ``schemas.httk.org`` identity.
SETTING_PROPERTY_KEYS: dict[str, str] = {
    "_httk_setting_it_nc": "setting_it_nc",
    "_httk_hall_entry": "hall_entry",
    "_httk_is_reference_setting": "is_reference_setting",
    "_httk_crystal_system": "crystal_system",
    "_httk_centring_type": "centring_type",
    "_httk_setting_transform": "affine_transformation",
}


@cache
def setting_definitions() -> dict[str, PropertyDefinition]:
    """Load the vendored symmetry-setting definitions.

    Loaded verbatim, so each keeps the ``$id`` under ``schemas.httk.org`` that makes it
    resolvable. Cached, since the documents are immutable.

    :return: Definitions keyed by their served names.
    """
    return load_httk_definitions(SETTING_PROPERTY_KEYS)


def symmetry_properties(structure: Any) -> dict[str, Any]:
    """Project standard and provider-specific symmetry properties.

    A structure that is not an :class:`~httk.atomistic.ASUStructure` carries no symmetry, so
    everything but ``fractional_site_positions`` serves null. That is the honest answer:
    inferring a space group here would mean running a symmetry search behind the caller's
    back, with a tolerance nobody chose, on every record served.

    :param structure: The structure to project, or ``None`` for an empty entry.
    :return: All supported symmetry properties, with unavailable values as ``None``.
    """
    values: dict[str, Any] = {name: None for name in SYMMETRY_PROPERTY_KEYS}
    values.update({name: None for name in SETTING_PROPERTY_KEYS})

    if structure is None:
        return values

    values["fractional_site_positions"] = (
        structure.fractional_site_positions
        if not isinstance(structure, FundamentalDomainStructure)
        else structure._representative_sites().reduced_coords.to_floats()
    )
    values["site_coordinate_span"] = structure.site_coordinate_span
    values["site_coordinate_span_description"] = structure.site_coordinate_span_description

    # A structure built from an asymmetric unit unwraps back to it, so the symmetry is
    # recovered rather than re-derived; anything else genuinely has none to report.
    asu = unwrap(structure)
    if not isinstance(asu, FundamentalDomainStructure):
        return values

    setting = asu.setting()
    values["space_group_it_number"] = asu.spacegroup.it_number

    if setting is None:
        # The structure is in a setting that appears in no table. Its space-group *number*
        # is still meaningful, but the symbols and the Wyckoff letters name a setting, and
        # this one has no name — so reporting the standard setting's would misdescribe it.
        values["_httk_setting_transform"] = _transform_record(asu)
        values["_httk_is_reference_setting"] = False
        values["_httk_crystal_system"] = asu.spacegroup.crystal_system
        return values

    values["space_group_symbol_hall"] = setting.hall_symbol
    values["space_group_symbol_hermann_mauguin"] = setting.hermann_mauguin
    values["space_group_symbol_hermann_mauguin_extended"] = _extended_symbol(setting)
    values["space_group_symmetry_operations_xyz"] = [
        operation.wrapped().to_xyz() for operation in setting.symmetry_operations
    ]
    values["wyckoff_positions"] = _wyckoff_symbols(asu, setting)

    values["_httk_setting_it_nc"] = setting.setting
    values["_httk_hall_entry"] = setting.hall_entry
    values["_httk_is_reference_setting"] = setting.is_standard_setting
    values["_httk_crystal_system"] = setting.crystal_system
    values["_httk_centring_type"] = setting.centring_type
    values["_httk_setting_transform"] = _transform_record(asu)
    return values


def _extended_symbol(setting: Any) -> str | None:
    """The extended Hermann-Mauguin symbol as a single line.

    International Tables prints it over several lines, listing the additional symmetry
    elements beneath the generators, and the tables keep that layout — so the continuation
    lines are folded onto one, since this property is a symbol string.

    Deliberately *not* the "universal" cctbx symbol, which is otherwise the more informative
    string: it appends the origin choice (``P n n n :2 (a+1/4,b+1/4,c+1/4)``), and this
    property states that the symbol SHOULD NOT be amended to convey that. The extended
    symbol is already written in the setting's own axes (``A 1 1 2/a`` for ``15:c1``), so
    the axis choice is conveyed without a change-of-basis suffix.
    """
    extended = setting.record.get("hm_extended")
    if not extended:
        return None
    return " ".join(part.strip() for part in str(extended).split("\n") if part.strip())


def _wyckoff_symbols(asu: FundamentalDomainStructure, setting: Any) -> list[str]:
    """One Wyckoff symbol per directly represented domain site.

    The symbols name positions of the setting the structure is *written in*, as OPTIMADE
    requires, not of the standard setting the ASU stores them against. One difference bites
    silently: the letters themselves are permuted for setting ``224:1``
    (standard ``j`` is that setting's ``i`` and vice versa). The multiplicity is used only
    to repeat each letter for the expanded sites; OPTIMADE receives the bare letter.
    """
    letters = wyckoff_letter_map(asu.spacegroup, setting)
    symbols: list[str] = []
    for site in asu.wyckoff_sites:
        position = setting.wyckoff_position(letters[site.wyckoff])
        symbols.append(position.letter)
    return symbols


def _transform_record(asu: FundamentalDomainStructure) -> dict[str, Any]:
    """The change of basis from the IT standard setting into this structure's own.

    Rendered as exact rational strings, matching how the vendored symmetry tables write
    them, so the value survives a JSON round trip without becoming a float.
    """
    transform = asu.transform
    return {
        "matrix": [[str(value) for value in row] for row in transform.matrix.to_fractions()],
        "vector": [str(value) for value in transform.vector.to_fractions()],
        "xyz": transform.operation.to_xyz(),
    }
