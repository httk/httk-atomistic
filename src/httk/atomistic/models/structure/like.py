"""
The accepted-input union for structure functions in httk-atomistic.
"""

import io
import os
import urllib.request
from typing import Any

import httk.core
import httk.core.datastream
import httk.core.optimade

import httk.atomistic.models.structure.ase
import httk.atomistic.models.structure.asu
import httk.atomistic.models.structure.backend
import httk.atomistic.models.structure.unitcell
import httk.atomistic.models.structure.view
import httk.atomistic.storage.records

# A structure is any structure backend/view, a UnitcellStructure, an ASUStructure (a structure
# held as its asymmetric unit, expanded on demand), or an spglib-like
# (lattice, positions, numbers) triple whose lattice and positions are vector-like.
type StructureLike = (
    httk.atomistic.models.structure.backend.StructureBackend
    | httk.atomistic.models.structure.view.StructureView
    | httk.atomistic.models.structure.unitcell.UnitcellStructure
    | httk.atomistic.storage.records.UnitcellStructureRecord
    | httk.atomistic.storage.records.FundamentalDomainStructureRecord
    | httk.atomistic.storage.records.ASUStructureRecord
    | httk.atomistic.models.structure.asu.FundamentalDomainStructure
    | httk.core.optimade.OptimadeResource
    | str
    | os.PathLike[str]
    | httk.core.DatastreamURL
    | urllib.request.Request
    | io.IOBase
    | httk.core.datastream.TextstreamBackend
    | httk.core.datastream.TextstreamView
    | httk.core.datastream.BytestreamBackend
    | httk.core.datastream.BytestreamView
    | httk.atomistic.models.structure.ase.ASEAtomsProtocol
    | tuple[httk.core.VectorLike, httk.core.VectorLike, Any]
    | list[Any]
)
