"""The abstract base class for all site-moments views."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.moments.backend import SiteMomentsBackend


class SiteMomentsViewBase(View[SiteMomentsBackend]):
    """Abstract base class for all views of site-moments data."""

    _backend_base_cls: ClassVar[type[SiteMomentsBackend]] = SiteMomentsBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


SiteMomentsViewBase._view_base_cls = SiteMomentsViewBase
