"""The geometry-free protostructure family."""

from typing import TYPE_CHECKING

from .api import ProtostructureAPI
from .backend import ProtostructureBackend
from .like import ProtostructureLike
from .occupation import WyckoffOccupation
from .protostructure import Protostructure
from .view_base import ProtostructureViewBase

if TYPE_CHECKING:
    from .recognized import RecognizedProtostructure
    from .view import ProtostructureView

__all__ = [
    "Protostructure",
    "ProtostructureAPI",
    "ProtostructureBackend",
    "ProtostructureLike",
    "ProtostructureView",
    "ProtostructureViewBase",
    "RecognizedProtostructure",
    "WyckoffOccupation",
]


def __getattr__(name: str) -> object:
    if name == "RecognizedProtostructure":
        from .recognized import RecognizedProtostructure

        globals()[name] = RecognizedProtostructure
        return RecognizedProtostructure
    if name == "ProtostructureView":
        from .view import ProtostructureView

        globals()[name] = ProtostructureView
        return ProtostructureView
    raise AttributeError(name)
