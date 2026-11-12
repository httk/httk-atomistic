"""The assigned-species protonetstructure family: a bare protostructure plus a covalent bonding net."""

from typing import TYPE_CHECKING

from .api import ProtonetstructureAPI
from .backend import ProtonetstructureBackend
from .like import ProtonetstructureLike
from .protonetstructure import Protonetstructure
from .view_base import ProtonetstructureViewBase

if TYPE_CHECKING:
    from .bonded import BondedProtonetstructure
    from .label import ProtonetstructureLabel
    from .label_string import ProtonetstructureLabelString
    from .view import ProtonetstructureView

__all__ = [
    "BondedProtonetstructure",
    "Protonetstructure",
    "ProtonetstructureAPI",
    "ProtonetstructureBackend",
    "ProtonetstructureLabel",
    "ProtonetstructureLabelString",
    "ProtonetstructureLike",
    "ProtonetstructureView",
    "ProtonetstructureViewBase",
]


def __getattr__(name: str) -> object:
    if name == "ProtonetstructureView":
        from .view import ProtonetstructureView

        globals()[name] = ProtonetstructureView
        return ProtonetstructureView
    if name == "ProtonetstructureLabel":
        from .label import ProtonetstructureLabel

        globals()[name] = ProtonetstructureLabel
        return ProtonetstructureLabel
    if name == "ProtonetstructureLabelString":
        from .label_string import ProtonetstructureLabelString

        globals()[name] = ProtonetstructureLabelString
        return ProtonetstructureLabelString
    if name == "BondedProtonetstructure":
        from .bonded import BondedProtonetstructure

        globals()[name] = BondedProtonetstructure
        return BondedProtonetstructure
    raise AttributeError(name)
