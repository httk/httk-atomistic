"""The abstract protostructure view base."""

from typing import Any, ClassVar, Self

from httk.core import View, unwrap

from httk.atomistic.models.bareprotostructure.backend import BareProtostructureBackend


class BareProtostructureViewBase(View[BareProtostructureBackend]):
    """Base class for views presenting protostructure backends."""

    @classmethod
    def _prepare_backend(cls, obj: Any, hints: dict[str, Any]) -> BareProtostructureBackend:
        from httk.atomistic.models.protostructure.backend import ProtostructureBackend

        if isinstance(obj, View):
            source = unwrap(obj)
            if isinstance(source, (BareProtostructureBackend, ProtostructureBackend)):
                obj = source
        return super()._prepare_backend(obj, hints)

    _backend_base_cls: ClassVar[type[BareProtostructureBackend]] = BareProtostructureBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


BareProtostructureViewBase._view_base_cls = BareProtostructureViewBase
