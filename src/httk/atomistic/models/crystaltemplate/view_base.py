"""The abstract view base for crystal templates."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.crystaltemplate.backend import CrystalTemplateBackend


class CrystalTemplateViewBase(View[CrystalTemplateBackend]):
    """Base class for views presenting crystal-template backends."""

    _backend_base_cls: ClassVar[type[CrystalTemplateBackend]] = CrystalTemplateBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


CrystalTemplateViewBase._view_base_cls = CrystalTemplateViewBase
