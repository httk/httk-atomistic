"""The abstract prototype view base."""

from typing import Any, ClassVar, Self

from httk.core import View, unwrap

from httk.atomistic.models.prototype.backend import PrototypeBackend


class PrototypeViewBase(View[PrototypeBackend]):
    """Base class for views presenting prototype backends."""

    @classmethod
    def _prepare_backend(cls, obj: Any, hints: dict[str, Any]) -> PrototypeBackend:
        if isinstance(obj, View):
            source = unwrap(obj)
            if isinstance(source, cls._backend_base_cls):
                obj = source
            else:
                from httk.atomistic.models.protostructure.backend import ProtostructureBackend

                if isinstance(source, ProtostructureBackend):
                    obj = source
        return super()._prepare_backend(obj, hints)

    _backend_base_cls: ClassVar[type[PrototypeBackend]] = PrototypeBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


PrototypeViewBase._view_base_cls = PrototypeViewBase
