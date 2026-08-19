"""The crystal-pattern family: dummy-species crystal patterns and fundamental domains."""

from typing import TYPE_CHECKING

from .anonymize import canonical_dummy_assignment, dummy_species, is_dummy_species
from .api import CrystalPatternAPI
from .backend import CrystalPatternBackend
from .crystalpattern import CrystalPattern
from .fundamental import ASUPattern, FundamentalDomainPattern
from .like import CrystalPatternLike
from .view_base import CrystalPatternViewBase

if TYPE_CHECKING:
    from .fundamental_view import FundamentalDomainPatternView
    from .view import CrystalPatternView

__all__ = [
    "ASUPattern",
    "CrystalPattern",
    "CrystalPatternAPI",
    "CrystalPatternBackend",
    "CrystalPatternLike",
    "CrystalPatternView",
    "CrystalPatternViewBase",
    "FundamentalDomainPattern",
    "FundamentalDomainPatternView",
    "canonical_dummy_assignment",
    "dummy_species",
    "is_dummy_species",
]


def __getattr__(name: str) -> object:
    if name == "CrystalPatternView":
        from .view import CrystalPatternView

        globals()[name] = CrystalPatternView
        return CrystalPatternView
    if name == "FundamentalDomainPatternView":
        from .fundamental_view import FundamentalDomainPatternView

        globals()[name] = FundamentalDomainPatternView
        return FundamentalDomainPatternView
    if name == "AnonymizedStructure":
        from .anonymized import AnonymizedStructure

        globals()[name] = AnonymizedStructure
        return AnonymizedStructure
    raise AttributeError(name)
