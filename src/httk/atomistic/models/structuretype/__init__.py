"""The structuretype family: dummy-species structuretypes and fundamental domains."""

from typing import TYPE_CHECKING

from .anonymize import canonical_dummy_assignment, dummy_species, is_dummy_species
from .api import StructuretypeAPI
from .backend import StructuretypeBackend
from .fundamental import ASUTemplate, FundamentalDomainTemplate
from .like import StructuretypeLike
from .structuretype import Structuretype
from .view_base import StructuretypeViewBase

if TYPE_CHECKING:
    from .fundamental_view import FundamentalDomainTemplateView
    from .view import StructuretypeView

__all__ = [
    "ASUTemplate",
    "FundamentalDomainTemplate",
    "FundamentalDomainTemplateView",
    "Structuretype",
    "StructuretypeAPI",
    "StructuretypeBackend",
    "StructuretypeLike",
    "StructuretypeView",
    "StructuretypeViewBase",
    "canonical_dummy_assignment",
    "dummy_species",
    "is_dummy_species",
]


def __getattr__(name: str) -> object:
    if name == "StructuretypeView":
        from .view import StructuretypeView

        globals()[name] = StructuretypeView
        return StructuretypeView
    if name == "FundamentalDomainTemplateView":
        from .fundamental_view import FundamentalDomainTemplateView

        globals()[name] = FundamentalDomainTemplateView
        return FundamentalDomainTemplateView
    if name == "AnonymizedStructure":
        from .anonymized import AnonymizedStructure

        globals()[name] = AnonymizedStructure
        return AnonymizedStructure
    raise AttributeError(name)
