"""The abstract structuretype view base."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.structuretype.backend import StructuretypeBackend


class StructuretypeViewBase(View[StructuretypeBackend]):
    """Base class for views presenting structuretype backends."""

    _backend_base_cls: ClassVar[type[StructuretypeBackend]] = StructuretypeBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


StructuretypeViewBase._view_base_cls = StructuretypeViewBase
