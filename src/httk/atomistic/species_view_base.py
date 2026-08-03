"""
The abstract base class for all species views in httk-atomistic.
"""

from typing import ClassVar, Self

from httk.core import View

from .species_backend import SpeciesBackend


class SpeciesViewBase(View[SpeciesBackend]):
    """
    Abstract base class for all views of single-species data.
    """

    _backend_base_cls: ClassVar[type[SpeciesBackend]] = SpeciesBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


SpeciesViewBase._view_base_cls = SpeciesViewBase
