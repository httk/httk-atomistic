"""
The abstract base class for all species backends in httk-atomistic.
"""

from typing import Any, ClassVar

from httk.core import Backend

from .species_api import SpeciesAPI


class SpeciesBackend(Backend["SpeciesBackend"], SpeciesAPI):
    """
    Abstract base class for all backends of single-species data.

    Concrete backends carry a native representation and produce the canonical OPTIMADE
    species accessors declared by ``SpeciesAPI`` from it.
    """

    backend_classes: ClassVar[list[type[Backend[Any]]]]
