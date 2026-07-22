"""
The accepted-input union for species functions in httk-atomistic.
"""

from typing import Any

from . import species, species_backend, species_view

type SpeciesLike = (species_backend.SpeciesBackend | species_view.SpeciesView | species.Species | dict[str, Any])
