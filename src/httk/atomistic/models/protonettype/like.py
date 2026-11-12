"""The accepted-input union for protonettypes."""

from typing import TYPE_CHECKING

import httk.atomistic.models.protonettype.backend
import httk.atomistic.models.protonettype.label
import httk.atomistic.models.protonettype.protonettype
import httk.atomistic.models.protonettype.view_base

if TYPE_CHECKING:
    import httk.atomistic.models.protonetstructure.backend
    import httk.atomistic.models.protonetstructure.view_base
    import httk.atomistic.models.structure.asu

type ProtonettypeLike = (
    httk.atomistic.models.protonettype.backend.ProtonettypeBackend
    | httk.atomistic.models.protonettype.view_base.ProtonettypeViewBase
    | httk.atomistic.models.protonettype.protonettype.Protonettype
    | httk.atomistic.models.protonettype.label.ProtonettypeLabel
    | str
    | httk.atomistic.models.protonetstructure.backend.ProtonetstructureBackend
    | httk.atomistic.models.protonetstructure.view_base.ProtonetstructureViewBase
    | httk.atomistic.models.structure.asu.FundamentalDomainStructure
)
