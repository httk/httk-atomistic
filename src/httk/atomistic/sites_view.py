"""
The abstract base class for all sites views in httk-atomistic.
"""

from typing import ClassVar, Self

from httk.core import View

from .sites_backend import SitesBackend


class SitesView(View[SitesBackend]):
    """
    Abstract base class for all views of sites data.
    """

    _backend_base_cls: ClassVar[type[SitesBackend]] = SitesBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


SitesView._view_base_cls = SitesView
