"""The geometry-free, element-free protochroma family."""

from typing import TYPE_CHECKING

from .api import ProtochromaAPI
from .backend import ProtochromaBackend
from .like import ProtochromaLike
from .occupation import ProtochromaOccupation
from .protochroma import Protochroma
from .view_base import ProtochromaViewBase

if TYPE_CHECKING:
    from .derived import DerivedProtochroma
    from .label import ProtochromaLabel
    from .label_string import ProtochromaLabelString
    from .view import ProtochromaView

__all__ = [
    "DerivedProtochroma",
    "Protochroma",
    "ProtochromaAPI",
    "ProtochromaBackend",
    "ProtochromaLabel",
    "ProtochromaLabelString",
    "ProtochromaLike",
    "ProtochromaOccupation",
    "ProtochromaView",
    "ProtochromaViewBase",
]


def __getattr__(name: str) -> object:
    if name == "DerivedProtochroma":
        from .derived import DerivedProtochroma

        globals()[name] = DerivedProtochroma
        return DerivedProtochroma
    if name == "ProtochromaLabel":
        from .label import ProtochromaLabel

        globals()[name] = ProtochromaLabel
        return ProtochromaLabel
    if name == "ProtochromaLabelString":
        from .label_string import ProtochromaLabelString

        globals()[name] = ProtochromaLabelString
        return ProtochromaLabelString
    if name == "ProtochromaView":
        from .view import ProtochromaView

        globals()[name] = ProtochromaView
        return ProtochromaView
    raise AttributeError(name)
