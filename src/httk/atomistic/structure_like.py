"""
The accepted-input union for structure functions in httk-atomistic.
"""

import io
import os
import urllib.request
from typing import Any

import httk.core

from . import (
    asu_structure,
    compat,
    structure_backend,
    structure_record,
    structure_view,
    unitcell_structure,
)

# A structure is any structure backend/view, a UnitcellStructure, an ASUStructure (a structure
# held as its asymmetric unit, expanded on demand), or an spglib-like
# (lattice, positions, numbers) triple whose lattice and positions are vector-like.
type StructureLike = (
    structure_backend.StructureBackend
    | structure_view.StructureView
    | unitcell_structure.UnitcellStructure
    | structure_record.UnitcellStructureRecord
    | structure_record.FundamentalDomainStructureRecord
    | structure_record.ASUStructureRecord
    | asu_structure.FundamentalDomainStructure
    | httk.core.OptimadeResource
    | str
    | os.PathLike[str]
    | httk.core.DatastreamURL
    | urllib.request.Request
    | io.IOBase
    | httk.core.TextstreamBackend
    | httk.core.TextstreamView
    | httk.core.BytestreamBackend
    | httk.core.BytestreamView
    | compat.ASEAtomsProtocol
    | tuple[httk.core.VectorLike, httk.core.VectorLike, Any]
    | list[Any]
)
