"""The assigned-species geometrical-classification family."""

from typing import TYPE_CHECKING

from .api import ProtostructureAPI
from .backend import ProtostructureBackend
from .like import ProtostructureLike
from .occupation import WyckoffOccupation
from .protostructure import Protostructure
from .view_base import ProtostructureViewBase

if TYPE_CHECKING:
    from .label import ProtostructureLabel
    from .view import ProtostructureView

__all__ = [
    "Protostructure",
    "ProtostructureAPI",
    "ProtostructureBackend",
    "ProtostructureLabel",
    "ProtostructureLike",
    "ProtostructureView",
    "ProtostructureViewBase",
    "WyckoffOccupation",
]


def __getattr__(name: str) -> object:
    if name == "ProtostructureView":
        from .view import ProtostructureView

        globals()[name] = ProtostructureView
        return ProtostructureView
    if name == "ProtostructureLabel":
        from .label import ProtostructureLabel

        globals()[name] = ProtostructureLabel
        return ProtostructureLabel
    raise AttributeError(name)
