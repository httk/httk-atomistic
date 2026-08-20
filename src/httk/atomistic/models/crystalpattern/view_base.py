"""The abstract view base for crystal patterns."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.crystalpattern.backend import CrystalPatternBackend


class CrystalPatternViewBase(View[CrystalPatternBackend]):
    """Base class for views presenting crystal-pattern backends."""

    _backend_base_cls: ClassVar[type[CrystalPatternBackend]] = CrystalPatternBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


CrystalPatternViewBase._view_base_cls = CrystalPatternViewBase
