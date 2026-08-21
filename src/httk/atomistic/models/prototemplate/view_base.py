"""The abstract prototemplate view base."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.prototemplate.backend import PrototemplateBackend


class PrototemplateViewBase(View[PrototemplateBackend]):
    """Base class for views presenting prototemplate backends."""

    _backend_base_cls: ClassVar[type[PrototemplateBackend]] = PrototemplateBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


PrototemplateViewBase._view_base_cls = PrototemplateViewBase
