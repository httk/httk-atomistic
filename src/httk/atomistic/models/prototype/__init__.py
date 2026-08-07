"""The anonymous-structure and prototype family."""

from typing import TYPE_CHECKING

from .anonymize import canonical_dummy_assignment, dummy_species, is_dummy_species
from .anonymous import AnonymousStructure
from .api import AnonymousStructureAPI
from .backend import AnonymousStructureBackend
from .like import AnonymousStructureLike
from .prototype import Prototype
from .view_base import AnonymousStructureViewBase

if TYPE_CHECKING:
    from .anonymous_view import AnonymousStructureView
    from .prototype_view import PrototypeView

__all__ = [
    "AnonymousStructure",
    "AnonymousStructureAPI",
    "AnonymousStructureBackend",
    "AnonymousStructureLike",
    "AnonymousStructureView",
    "AnonymousStructureViewBase",
    "Prototype",
    "PrototypeView",
    "canonical_dummy_assignment",
    "dummy_species",
    "is_dummy_species",
]


def __getattr__(name: str) -> object:
    if name == "AnonymousStructureView":
        from .anonymous_view import AnonymousStructureView

        globals()[name] = AnonymousStructureView
        return AnonymousStructureView
    if name == "PrototypeView":
        from .prototype_view import PrototypeView

        globals()[name] = PrototypeView
        return PrototypeView
    if name == "AnonymizedStructure":
        from .anonymized import AnonymizedStructure

        globals()[name] = AnonymizedStructure
        return AnonymizedStructure
    raise AttributeError(name)
