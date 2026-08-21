"""The chromastructure family: dummy-species chromastructures and fundamental domains."""

from typing import TYPE_CHECKING

from .anonymize import canonical_dummy_assignment, dummy_species, is_dummy_species
from .api import ChromastructureAPI
from .backend import ChromastructureBackend
from .chromastructure import Chromastructure
from .fundamental import ASUPattern, FundamentalDomainPattern
from .like import ChromastructureLike
from .view_base import ChromastructureViewBase

if TYPE_CHECKING:
    from .fundamental_view import FundamentalDomainPatternView
    from .view import ChromastructureView

__all__ = [
    "ASUPattern",
    "Chromastructure",
    "ChromastructureAPI",
    "ChromastructureBackend",
    "ChromastructureLike",
    "ChromastructureView",
    "ChromastructureViewBase",
    "FundamentalDomainPattern",
    "FundamentalDomainPatternView",
    "canonical_dummy_assignment",
    "dummy_species",
    "is_dummy_species",
]


def __getattr__(name: str) -> object:
    if name == "ChromastructureView":
        from .view import ChromastructureView

        globals()[name] = ChromastructureView
        return ChromastructureView
    if name == "FundamentalDomainPatternView":
        from .fundamental_view import FundamentalDomainPatternView

        globals()[name] = FundamentalDomainPatternView
        return FundamentalDomainPatternView
    if name == "AnonymizedStructure":
        from .anonymized import AnonymizedStructure

        globals()[name] = AnonymizedStructure
        return AnonymizedStructure
    raise AttributeError(name)
