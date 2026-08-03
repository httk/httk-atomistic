"""
The accepted-input union for cell functions in httk-atomistic.
"""

import httk.core

import httk.atomistic.models.cell.backend
import httk.atomistic.models.cell.cell
import httk.atomistic.models.cell.view_base

# A cell is any cell backend/view, a Cell, or any vector-like: a 3x3 basis matrix (nested numbers,
# FracVector, SurdVector, numpy array, ...) or a flat 6-sequence of cell parameters.
type CellLike = (
    httk.atomistic.models.cell.backend.CellBackend
    | httk.atomistic.models.cell.view_base.CellViewBase
    | httk.atomistic.models.cell.cell.Cell
    | httk.core.VectorLike
)
