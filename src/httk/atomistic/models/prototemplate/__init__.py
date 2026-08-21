"""The geometry-free, element-free prototemplate family."""

from typing import TYPE_CHECKING

from .api import PrototemplateAPI
from .backend import PrototemplateBackend
from .like import PrototemplateLike
from .occupation import PrototemplateOccupation
from .prototemplate import Prototemplate
from .view_base import PrototemplateViewBase

if TYPE_CHECKING:
    from .derived import DerivedPrototemplate
    from .label import PrototemplateLabel
    from .label_string import PrototemplateLabelString
    from .view import PrototemplateView

__all__ = [
    "DerivedPrototemplate",
    "Prototemplate",
    "PrototemplateAPI",
    "PrototemplateBackend",
    "PrototemplateLabel",
    "PrototemplateLabelString",
    "PrototemplateLike",
    "PrototemplateOccupation",
    "PrototemplateView",
    "PrototemplateViewBase",
]


def __getattr__(name: str) -> object:
    if name == "DerivedPrototemplate":
        from .derived import DerivedPrototemplate

        globals()[name] = DerivedPrototemplate
        return DerivedPrototemplate
    if name == "PrototemplateLabel":
        from .label import PrototemplateLabel

        globals()[name] = PrototemplateLabel
        return PrototemplateLabel
    if name == "PrototemplateLabelString":
        from .label_string import PrototemplateLabelString

        globals()[name] = PrototemplateLabelString
        return PrototemplateLabelString
    if name == "PrototemplateView":
        from .view import PrototemplateView

        globals()[name] = PrototemplateView
        return PrototemplateView
    raise AttributeError(name)
