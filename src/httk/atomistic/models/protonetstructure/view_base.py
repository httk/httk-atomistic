"""The abstract protonetstructure view base."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.protonetstructure.backend import ProtonetstructureBackend


class ProtonetstructureViewBase(View[ProtonetstructureBackend]):
    """Base class for views presenting protonetstructure backends."""

    _backend_base_cls: ClassVar[type[ProtonetstructureBackend]] = ProtonetstructureBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


ProtonetstructureViewBase._view_base_cls = ProtonetstructureViewBase
