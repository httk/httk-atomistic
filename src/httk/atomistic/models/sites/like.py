"""
The accepted-input union for sites functions in httk-atomistic.
"""

import httk.core

import httk.atomistic.models.sites.backend
import httk.atomistic.models.sites.sites
import httk.atomistic.models.sites.view_base

# Sites are any sites backend/view, a Sites, or any Nx3 vector-like (nested numbers, FracVector,
# SurdVector, numpy array, ...).
type SitesLike = (
    httk.atomistic.models.sites.backend.SitesBackend
    | httk.atomistic.models.sites.view_base.SitesViewBase
    | httk.atomistic.models.sites.sites.Sites
    | httk.core.VectorLike
)
