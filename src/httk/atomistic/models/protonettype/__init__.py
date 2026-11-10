"""The anonymous protonettype family: a bare prototype plus a covalent bonding net."""

from typing import TYPE_CHECKING

from .api import ProtonettypeAPI
from .backend import ProtonettypeBackend
from .like import ProtonettypeLike
from .protonettype import Protonettype
from .view_base import ProtonettypeViewBase

if TYPE_CHECKING:
    from .derived import DerivedProtonettype
    from .label import ProtonettypeLabel
    from .label_string import ProtonettypeLabelString
    from .view import ProtonettypeView

__all__ = [
    "DerivedProtonettype",
    "Protonettype",
    "ProtonettypeAPI",
    "ProtonettypeBackend",
    "ProtonettypeLabel",
    "ProtonettypeLabelString",
    "ProtonettypeLike",
    "ProtonettypeView",
    "ProtonettypeViewBase",
]


def __getattr__(name: str) -> object:
    if name == "ProtonettypeView":
        from .view import ProtonettypeView

        globals()[name] = ProtonettypeView
        return ProtonettypeView
    if name == "ProtonettypeLabel":
        from .label import ProtonettypeLabel

        globals()[name] = ProtonettypeLabel
        return ProtonettypeLabel
    if name == "ProtonettypeLabelString":
        from .label_string import ProtonettypeLabelString

        globals()[name] = ProtonettypeLabelString
        return ProtonettypeLabelString
    if name == "DerivedProtonettype":
        from .derived import DerivedProtonettype

        globals()[name] = DerivedProtonettype
        return DerivedProtonettype
    raise AttributeError(name)
