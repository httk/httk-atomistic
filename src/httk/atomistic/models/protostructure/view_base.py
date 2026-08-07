"""The abstract protostructure view base."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.protostructure.backend import ProtostructureBackend


class ProtostructureViewBase(View[ProtostructureBackend]):
    """Base class for views presenting protostructure backends."""

    _backend_base_cls: ClassVar[type[ProtostructureBackend]] = ProtostructureBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


ProtostructureViewBase._view_base_cls = ProtostructureViewBase
