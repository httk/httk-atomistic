from typing import TYPE_CHECKING

from .api import SpeciesAPI
from .backend import SpeciesBackend
from .like import SpeciesLike
from .plain import PlainSpecies
from .plain_view import PlainSpeciesView
from .species import Species
from .view import SpeciesView
from .view_base import SpeciesViewBase

__all__ = [
    "PlainSpecies",
    "PlainSpeciesView",
    "RecordSpecies",
    "Species",
    "SpeciesAPI",
    "SpeciesBackend",
    "SpeciesLike",
    "SpeciesView",
    "SpeciesViewBase",
]

if TYPE_CHECKING:
    from .record import RecordSpecies


def __getattr__(name: str) -> object:
    if name == "RecordSpecies":
        from .record import RecordSpecies

        globals()[name] = RecordSpecies
        return RecordSpecies
    raise AttributeError(name)
