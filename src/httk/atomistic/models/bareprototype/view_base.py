"""The abstract prototype view base."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.bareprototype.backend import BarePrototypeBackend


class BarePrototypeViewBase(View[BarePrototypeBackend]):
    """Base class for views presenting prototype backends."""

    _backend_base_cls: ClassVar[type[BarePrototypeBackend]] = BarePrototypeBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


BarePrototypeViewBase._view_base_cls = BarePrototypeViewBase
