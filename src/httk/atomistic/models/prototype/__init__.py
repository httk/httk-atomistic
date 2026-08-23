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
    from .label_string import PrototypeLabelString
    from .recognized import RecognizedPrototype
    from .view import PrototypeView

__all__ = [
    "DerivedPrototype",
    "Prototype",
    "PrototypeAPI",
    "PrototypeBackend",
    "PrototypeLabel",
    "PrototypeLabelString",
    "PrototypeLike",
    "PrototypeOccupation",
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
    if name == "PrototypeLabel":
        from .label import PrototypeLabel

        globals()[name] = PrototypeLabel
        return PrototypeLabel
    if name == "PrototypeLabelString":
        from .label_string import PrototypeLabelString

        globals()[name] = PrototypeLabelString
        return PrototypeLabelString
    if name == "DerivedPrototype":
        from .derived import DerivedPrototype

        globals()[name] = DerivedPrototype
        return DerivedPrototype
    raise AttributeError(name)
