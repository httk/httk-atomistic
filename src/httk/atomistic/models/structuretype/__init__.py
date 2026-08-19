"""The assigned geometrical-class structuretype family."""

from typing import TYPE_CHECKING

from .api import StructuretypeAPI
from .backend import StructuretypeBackend
from .like import StructuretypeLike
from .structuretype import Structuretype
from .view_base import StructuretypeViewBase

if TYPE_CHECKING:
    from .recognized import RecognizedStructuretype
    from .view import StructuretypeView

__all__ = [
    "RecognizedStructuretype",
    "Structuretype",
    "StructuretypeAPI",
    "StructuretypeBackend",
    "StructuretypeLike",
    "StructuretypeView",
    "StructuretypeViewBase",
]


def __getattr__(name: str) -> object:
    if name == "RecognizedStructuretype":
        from .recognized import RecognizedStructuretype

        globals()[name] = RecognizedStructuretype
        return RecognizedStructuretype
    if name == "StructuretypeView":
        from .view import StructuretypeView

        globals()[name] = StructuretypeView
        return StructuretypeView
    raise AttributeError(name)
