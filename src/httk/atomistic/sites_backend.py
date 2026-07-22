"""
The abstract base class for all sites backends in httk-atomistic.
"""

from typing import Any, ClassVar

from httk.core import Backend

from .sites_api import SitesAPI


class SitesBackend(Backend["SitesBackend"], SitesAPI):
    """
    Abstract base class for all backends of sites data.

    Concrete backends carry a native representation and produce the canonical Nx3
    ``reduced_coords`` declared by ``SitesAPI`` from it.
    """

    backend_classes: ClassVar[list[type[Backend[Any]]]]
