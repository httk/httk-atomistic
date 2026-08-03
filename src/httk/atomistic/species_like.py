"""
The accepted-input union for species functions in httk-atomistic.
"""

from typing import Any

from . import species, species_backend, species_view_base

type SpeciesLike = (
    species_backend.SpeciesBackend | species_view_base.SpeciesViewBase | species.Species | dict[str, Any] | str | int
)
