"""The abstract protochroma view base."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.protochroma.backend import ProtochromaBackend


class ProtochromaViewBase(View[ProtochromaBackend]):
    """Base class for views presenting protochroma backends."""

    _backend_base_cls: ClassVar[type[ProtochromaBackend]] = ProtochromaBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


ProtochromaViewBase._view_base_cls = ProtochromaViewBase
