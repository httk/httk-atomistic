"""The crystal-template family: dummy-species crystal templates and fundamental domains."""

from typing import TYPE_CHECKING

from .anonymize import canonical_dummy_assignment, dummy_species, is_dummy_species
from .api import CrystalTemplateAPI
from .backend import CrystalTemplateBackend
from .crystaltemplate import CrystalTemplate
from .fundamental import ASUTemplate, FundamentalDomainTemplate
from .like import CrystalTemplateLike
from .view_base import CrystalTemplateViewBase

if TYPE_CHECKING:
    from .fundamental_view import FundamentalDomainTemplateView
    from .view import CrystalTemplateView

__all__ = [
    "ASUTemplate",
    "CrystalTemplate",
    "CrystalTemplateAPI",
    "CrystalTemplateBackend",
    "CrystalTemplateLike",
    "CrystalTemplateView",
    "CrystalTemplateViewBase",
    "FundamentalDomainTemplate",
    "FundamentalDomainTemplateView",
    "canonical_dummy_assignment",
    "dummy_species",
    "is_dummy_species",
]


def __getattr__(name: str) -> object:
    if name == "CrystalTemplateView":
        from .view import CrystalTemplateView

        globals()[name] = CrystalTemplateView
        return CrystalTemplateView
    if name == "FundamentalDomainTemplateView":
        from .fundamental_view import FundamentalDomainTemplateView

        globals()[name] = FundamentalDomainTemplateView
        return FundamentalDomainTemplateView
    if name == "AnonymizedStructure":
        from .anonymized import AnonymizedStructure

        globals()[name] = AnonymizedStructure
        return AnonymizedStructure
    raise AttributeError(name)
