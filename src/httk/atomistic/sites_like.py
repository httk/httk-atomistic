"""
The accepted-input union for sites functions in httk-atomistic.
"""

from typing import Any

from . import sites, sites_backend, sites_view

type SitesLike = (sites_backend.SitesBackend | sites_view.SitesView | sites.Sites | tuple[Any, ...] | list[Any])
