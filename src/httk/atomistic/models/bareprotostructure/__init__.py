"""The assigned-species Wyckoff-only classification family."""

from typing import TYPE_CHECKING

from httk.atomistic.models.protostructure.occupation import WyckoffOccupation

from .api import BareProtostructureAPI
from .backend import BareProtostructureBackend
from .bareprotostructure import BareProtostructure
from .like import BareProtostructureLike
from .view_base import BareProtostructureViewBase

if TYPE_CHECKING:
    from .label import BareProtostructureLabel
    from .label_string import BareProtostructureLabelString
    from .recognized import RecognizedBareProtostructure
    from .view import BareProtostructureView

__all__ = [
    "BareProtostructure",
    "BareProtostructureAPI",
    "BareProtostructureBackend",
    "BareProtostructureLabel",
    "BareProtostructureLabelString",
    "BareProtostructureLike",
    "BareProtostructureView",
    "BareProtostructureViewBase",
    "RecognizedBareProtostructure",
    "WyckoffOccupation",
]


def __getattr__(name: str) -> object:
    if name == "RecognizedBareProtostructure":
        from .recognized import RecognizedBareProtostructure

        globals()[name] = RecognizedBareProtostructure
        return RecognizedBareProtostructure
    if name == "BareProtostructureView":
        from .view import BareProtostructureView

        globals()[name] = BareProtostructureView
        return BareProtostructureView
    if name == "BareProtostructureLabel":
        from .label import BareProtostructureLabel

        globals()[name] = BareProtostructureLabel
        return BareProtostructureLabel
    if name == "BareProtostructureLabelString":
        from .label_string import BareProtostructureLabelString

        globals()[name] = BareProtostructureLabelString
        return BareProtostructureLabelString
    raise AttributeError(name)
