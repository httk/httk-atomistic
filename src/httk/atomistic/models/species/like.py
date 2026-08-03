"""
The accepted-input union for species functions in httk-atomistic.
"""

from typing import Any

import httk.atomistic.models.species.backend
import httk.atomistic.models.species.species
import httk.atomistic.models.species.view_base

type SpeciesLike = (
    httk.atomistic.models.species.backend.SpeciesBackend
    | httk.atomistic.models.species.view_base.SpeciesViewBase
    | httk.atomistic.models.species.species.Species
    | dict[str, Any]
    | str
    | int
)
