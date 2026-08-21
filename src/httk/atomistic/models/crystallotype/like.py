"""The accepted-input union for crystallotypes."""

import httk.atomistic.models.crystallotype.backend
import httk.atomistic.models.crystallotype.crystallotype
import httk.atomistic.models.crystallotype.view_base

type CrystallotypeLike = (
    httk.atomistic.models.crystallotype.backend.CrystallotypeBackend
    | httk.atomistic.models.crystallotype.view_base.CrystallotypeViewBase
    | httk.atomistic.models.crystallotype.crystallotype.Crystallotype
)
