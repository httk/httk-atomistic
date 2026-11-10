"""The abstract protonettype view base."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.protonettype.backend import ProtonettypeBackend


class ProtonettypeViewBase(View[ProtonettypeBackend]):
    """Base class for views presenting protonettype backends."""

    _backend_base_cls: ClassVar[type[ProtonettypeBackend]] = ProtonettypeBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


ProtonettypeViewBase._view_base_cls = ProtonettypeViewBase
