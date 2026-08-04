"""The accepted backend/view union for site-moments functions.

There is intentionally no ``VectorLike`` arm: a bare Nx3 array is frame-ambiguous, and no
lossless conversion from collinear scalars to a Cartesian or crystal-axis representation exists.
There is consequently no ``CollinearSiteMomentsView`` either.
"""

import httk.atomistic.models.moments.backend
import httk.atomistic.models.moments.view_base

type SiteMomentsLike = (
    httk.atomistic.models.moments.backend.SiteMomentsBackend
    | httk.atomistic.models.moments.view_base.SiteMomentsViewBase
)
