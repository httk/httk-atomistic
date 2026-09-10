"""The abstract protostructure view base."""

from typing import Any, ClassVar, Self

from httk.core import View, unwrap

from httk.atomistic.models.protostructure.backend import ProtostructureBackend


class ProtostructureViewBase(View[ProtostructureBackend]):
    """Base class for views presenting protostructure backends."""

    @classmethod
    def _prepare_backend(cls, obj: Any, hints: dict[str, Any]) -> ProtostructureBackend:
        if isinstance(obj, View):
            source = unwrap(obj)
            if isinstance(source, cls._backend_base_cls):
                obj = source
        return super()._prepare_backend(obj, hints)

    _backend_base_cls: ClassVar[type[ProtostructureBackend]] = ProtostructureBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


ProtostructureViewBase._view_base_cls = ProtostructureViewBase
