"""The assigned geometrical-class crystallotype family."""

from typing import TYPE_CHECKING

from .api import CrystallotypeAPI
from .backend import CrystallotypeBackend
from .crystallotype import Crystallotype
from .like import CrystallotypeLike
from .view_base import CrystallotypeViewBase

if TYPE_CHECKING:
    from .recognized import RecognizedCrystallotype
    from .view import CrystallotypeView

__all__ = [
    "Crystallotype",
    "CrystallotypeAPI",
    "CrystallotypeBackend",
    "CrystallotypeLike",
    "CrystallotypeView",
    "CrystallotypeViewBase",
    "RecognizedCrystallotype",
]


def __getattr__(name: str) -> object:
    if name == "RecognizedCrystallotype":
        from .recognized import RecognizedCrystallotype

        globals()[name] = RecognizedCrystallotype
        return RecognizedCrystallotype
    if name == "CrystallotypeView":
        from .view import CrystallotypeView

        globals()[name] = CrystallotypeView
        return CrystallotypeView
    raise AttributeError(name)
