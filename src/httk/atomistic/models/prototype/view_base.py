"""The abstract view base for anonymous structures."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.prototype.backend import AnonymousStructureBackend


class AnonymousStructureViewBase(View[AnonymousStructureBackend]):
    """Base class for views presenting anonymous-structure backends."""

    _backend_base_cls: ClassVar[type[AnonymousStructureBackend]] = AnonymousStructureBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


AnonymousStructureViewBase._view_base_cls = AnonymousStructureViewBase
