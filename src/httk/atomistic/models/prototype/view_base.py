"""The abstract prototype view base."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.prototype.backend import PrototypeBackend


class PrototypeViewBase(View[PrototypeBackend]):
    """Base class for views presenting prototype backends."""

    _backend_base_cls: ClassVar[type[PrototypeBackend]] = PrototypeBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


PrototypeViewBase._view_base_cls = PrototypeViewBase
