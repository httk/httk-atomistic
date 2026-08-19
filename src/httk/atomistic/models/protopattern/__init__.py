"""The geometry-free, element-free protopattern family."""

from typing import TYPE_CHECKING

from .api import ProtopatternAPI
from .backend import ProtopatternBackend
from .like import ProtopatternLike
from .occupation import ProtopatternOccupation
from .protopattern import Protopattern
from .view_base import ProtopatternViewBase

if TYPE_CHECKING:
    from .derived import DerivedProtopattern
    from .label import ProtopatternLabel
    from .label_string import ProtopatternLabelString
    from .view import ProtopatternView

__all__ = [
    "DerivedProtopattern",
    "Protopattern",
    "ProtopatternAPI",
    "ProtopatternBackend",
    "ProtopatternLabel",
    "ProtopatternLabelString",
    "ProtopatternLike",
    "ProtopatternOccupation",
    "ProtopatternView",
    "ProtopatternViewBase",
]


def __getattr__(name: str) -> object:
    if name == "DerivedProtopattern":
        from .derived import DerivedProtopattern

        globals()[name] = DerivedProtopattern
        return DerivedProtopattern
    if name == "ProtopatternLabel":
        from .label import ProtopatternLabel

        globals()[name] = ProtopatternLabel
        return ProtopatternLabel
    if name == "ProtopatternLabelString":
        from .label_string import ProtopatternLabelString

        globals()[name] = ProtopatternLabelString
        return ProtopatternLabelString
    if name == "ProtopatternView":
        from .view import ProtopatternView

        globals()[name] = ProtopatternView
        return ProtopatternView
    raise AttributeError(name)
