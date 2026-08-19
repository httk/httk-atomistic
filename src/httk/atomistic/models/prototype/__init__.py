"""The anonymous geometrical-class prototype family."""

from typing import TYPE_CHECKING

from .api import PrototypeAPI
from .backend import PrototypeBackend
from .like import PrototypeLike
from .prototype import Prototype
from .view_base import PrototypeViewBase

if TYPE_CHECKING:
    from .recognized import RecognizedPrototype
    from .view import PrototypeView

__all__ = [
    "Prototype",
    "PrototypeAPI",
    "PrototypeBackend",
    "PrototypeLike",
    "PrototypeView",
    "PrototypeViewBase",
    "RecognizedPrototype",
]


def __getattr__(name: str) -> object:
    if name == "RecognizedPrototype":
        from .recognized import RecognizedPrototype

        globals()[name] = RecognizedPrototype
        return RecognizedPrototype
    if name == "PrototypeView":
        from .view import PrototypeView

        globals()[name] = PrototypeView
        return PrototypeView
    raise AttributeError(name)
