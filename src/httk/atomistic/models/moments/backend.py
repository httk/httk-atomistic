"""
The abstract base class for all site-moments backends in httk-atomistic.
"""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.moments.api import SiteMomentsAPI


class SiteMomentsBackend(Backend["SiteMomentsBackend"], SiteMomentsAPI):
    """
    Abstract base class for all backends of site-moments data.

    Concrete backends carry a native representation and produce the canonical Nx3
    ``cartesian_moments`` declared by ``SiteMomentsAPI`` from it.
    """

    backend_classes: ClassVar[list[type[Backend[Any]]]]
