"""The accepted-input union for crystal patterns."""

import httk.atomistic.models.crystalpattern.backend
import httk.atomistic.models.crystalpattern.crystalpattern
import httk.atomistic.models.crystalpattern.fundamental
import httk.atomistic.models.crystalpattern.view_base
import httk.atomistic.models.protostructure.like

type CrystalPatternLike = (
    httk.atomistic.models.crystalpattern.backend.CrystalPatternBackend
    | httk.atomistic.models.crystalpattern.view_base.CrystalPatternViewBase
    | httk.atomistic.models.crystalpattern.crystalpattern.CrystalPattern
    | httk.atomistic.models.crystalpattern.fundamental.FundamentalDomainPattern
)

# Transitional alias, removed in the taxonomy phase 4.
type PrototypeLike = CrystalPatternLike | httk.atomistic.models.protostructure.like.ProtostructureLike
