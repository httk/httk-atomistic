"""Loading the vendored property definitions published by schemas.httk.org.

OPTIMADE standardises a good deal about a structure but nothing about *which setting* it is
written in, nor about how precisely its numbers were stated. Both are served here as
database-specific properties, and rather than describing them locally the descriptions are
taken from the published definitions at `schemas.httk.org <https://schemas.httk.org>`_,
vendored verbatim in ``httk.registry.schemas.atomistic``.

The served *name* carries the ``_httk_`` prefix that OPTIMADE requires of a
database-specific property; the definition keeps its own ``$id``, so a client following the
link reaches the authoritative schema rather than a local paraphrase.
"""

from collections.abc import Mapping

from httk.core import PropertyDefinition, load_property_definition

__all__ = ["load_httk_definitions"]

_DEFINITION_IDS = {
    "affine_transformation": "https://schemas.httk.org/defs/v0.1/properties/symmetry/affine_transformation",
    "centring_type": "https://schemas.httk.org/defs/v0.1/properties/spacegroups/centring_type",
    "crystal_system": "https://schemas.httk.org/defs/v0.1/properties/pointgroups/crystal_system",
    "fractional_coordinate_precision": "https://schemas.httk.org/defs/v0.1/properties/core/fractional_coordinate_precision",
    "hall_entry": "https://schemas.httk.org/defs/v0.1/properties/spacegroups/hall_entry",
    "is_reference_setting": "https://schemas.httk.org/defs/v0.1/properties/spacegroups/is_reference_setting",
    "length_precision": "https://schemas.httk.org/defs/v0.1/properties/core/length_precision",
    "species_charges": "https://schemas.httk.org/defs/v0.1/properties/chemistry/species_charges",
    "species_labels": "https://schemas.httk.org/defs/v0.1/properties/chemistry/species_labels",
    "species_spins": "https://schemas.httk.org/defs/v0.1/properties/chemistry/species_spins",
    "setting_it_nc": "https://schemas.httk.org/defs/v0.1/properties/spacegroups/setting_it_nc",
    "site_moments": "https://schemas.httk.org/defs/v0.1/properties/magnetism/site_moments",
    "structure_charge": "https://schemas.httk.org/defs/v0.1/properties/chemistry/structure_charge",
    "frame_stresses": "https://schemas.httk.org/defs/v0.1/properties/trajectories/frame_stresses",
    "frame_temperatures": "https://schemas.httk.org/defs/v0.1/properties/trajectories/frame_temperatures",
    "frame_total_energies": "https://schemas.httk.org/defs/v0.1/properties/trajectories/frame_total_energies",
    "time_step": "https://schemas.httk.org/defs/v0.1/properties/trajectories/time_step",
}


def load_httk_definitions(names: Mapping[str, str]) -> dict[str, PropertyDefinition]:
    """Load vendored property definitions by served name.

    Each document is loaded verbatim, ``$id`` included. Note that a definition's own
    ``x-optimade-definition.name`` stays the unprefixed name it was published under, which
    will differ from the prefixed name httk serves it as — that is correct, and rewriting a
    published document to match a local naming choice would defeat the point of pointing at
    it.

    :param names: A map from served property name to vendored definition stem.
    :return: The loaded definitions keyed by served property name.
    """
    return {
        served_name: PropertyDefinition.from_optimade(
            served_name, load_property_definition(_DEFINITION_IDS[definition_name]).as_optimade()
        )
        for served_name, definition_name in names.items()
    }
