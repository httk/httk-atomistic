"""The anonymous Wyckoff-only prototype family."""

from typing import TYPE_CHECKING

from httk.atomistic.models.prototype.occupation import PrototypeOccupation

from .api import BarePrototypeAPI
from .backend import BarePrototypeBackend
from .bareprototype import BarePrototype
from .like import BarePrototypeLike
from .view_base import BarePrototypeViewBase

if TYPE_CHECKING:
    from .derived import DerivedBarePrototype
    from .label import BarePrototypeLabel
    from .label_string import BarePrototypeLabelString
    from .recognized import RecognizedBarePrototype
    from .view import BarePrototypeView

__all__ = [
    "BarePrototype",
    "BarePrototypeAPI",
    "BarePrototypeBackend",
    "BarePrototypeLabel",
    "BarePrototypeLabelString",
    "BarePrototypeLike",
    "BarePrototypeView",
    "BarePrototypeViewBase",
    "DerivedBarePrototype",
    "PrototypeOccupation",
    "RecognizedBarePrototype",
]


def __getattr__(name: str) -> object:
    if name == "RecognizedBarePrototype":
        from .recognized import RecognizedBarePrototype

        globals()[name] = RecognizedBarePrototype
        return RecognizedBarePrototype
    if name == "BarePrototypeView":
        from .view import BarePrototypeView

        globals()[name] = BarePrototypeView
        return BarePrototypeView
    if name == "BarePrototypeLabel":
        from .label import BarePrototypeLabel

        globals()[name] = BarePrototypeLabel
        return BarePrototypeLabel
    if name == "BarePrototypeLabelString":
        from .label_string import BarePrototypeLabelString

        globals()[name] = BarePrototypeLabelString
        return BarePrototypeLabelString
    if name == "DerivedBarePrototype":
        from .derived import DerivedBarePrototype

        globals()[name] = DerivedBarePrototype
        return DerivedBarePrototype
    raise AttributeError(name)
