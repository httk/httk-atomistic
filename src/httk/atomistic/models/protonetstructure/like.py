"""The accepted-input union for protonetstructures."""

from typing import TYPE_CHECKING

import httk.atomistic.models.protonetstructure.backend
import httk.atomistic.models.protonetstructure.label
import httk.atomistic.models.protonetstructure.protonetstructure
import httk.atomistic.models.protonetstructure.view_base

if TYPE_CHECKING:
    import httk.atomistic.models.structure.asu

type ProtonetstructureLike = (
    httk.atomistic.models.protonetstructure.backend.ProtonetstructureBackend
    | httk.atomistic.models.protonetstructure.view_base.ProtonetstructureViewBase
    | httk.atomistic.models.protonetstructure.protonetstructure.Protonetstructure
    | httk.atomistic.models.protonetstructure.label.ProtonetstructureLabel
    | str
    | httk.atomistic.models.structure.asu.FundamentalDomainStructure
)
