"""The accepted-input union for chromastructures."""

import httk.atomistic.models.chromastructure.backend
import httk.atomistic.models.chromastructure.chromastructure
import httk.atomistic.models.chromastructure.fundamental
import httk.atomistic.models.chromastructure.view_base

type ChromastructureLike = (
    httk.atomistic.models.chromastructure.backend.ChromastructureBackend
    | httk.atomistic.models.chromastructure.view_base.ChromastructureViewBase
    | httk.atomistic.models.chromastructure.chromastructure.Chromastructure
    | httk.atomistic.models.chromastructure.fundamental.FundamentalDomainPattern
)
