"""
The accepted-input union for sites functions in httk-atomistic.
"""

import httk.core

from . import sites, sites_backend, sites_view

# Sites are any sites backend/view, a Sites, or any Nx3 vector-like (nested numbers, FracVector,
# SurdVector, numpy array, ...).
type SitesLike = sites_backend.SitesBackend | sites_view.SitesView | sites.Sites | httk.core.VectorLike
