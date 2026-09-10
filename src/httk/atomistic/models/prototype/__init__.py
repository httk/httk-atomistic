"""The anonymous geometrical-class prototype family."""

from typing import TYPE_CHECKING

from .api import PrototypeAPI
from .backend import PrototypeBackend
from .like import PrototypeLike
from .occupation import PrototypeOccupation
from .prototype import Prototype
from .view_base import PrototypeViewBase

if TYPE_CHECKING:
    from .derived import DerivedPrototype
    from .label import PrototypeLabel
    from .view import PrototypeView

__all__ = [
    "DerivedPrototype",
    "Prototype",
    "PrototypeAPI",
    "PrototypeBackend",
    "PrototypeLabel",
    "PrototypeLike",
    "PrototypeOccupation",
    "PrototypeView",
    "PrototypeViewBase",
]


def __getattr__(name: str) -> object:
    if name == "PrototypeView":
        from .view import PrototypeView

        globals()[name] = PrototypeView
        return PrototypeView
    if name == "PrototypeLabel":
        from .label import PrototypeLabel

        globals()[name] = PrototypeLabel
        return PrototypeLabel
    if name == "DerivedPrototype":
        from .derived import DerivedPrototype

        globals()[name] = DerivedPrototype
        return DerivedPrototype
    raise AttributeError(name)
