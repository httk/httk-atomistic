"""The abstract protopattern view base."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.protopattern.backend import ProtopatternBackend


class ProtopatternViewBase(View[ProtopatternBackend]):
    """Base class for views presenting protopattern backends."""

    _backend_base_cls: ClassVar[type[ProtopatternBackend]] = ProtopatternBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


ProtopatternViewBase._view_base_cls = ProtopatternViewBase
