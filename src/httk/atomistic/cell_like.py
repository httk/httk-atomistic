"""
The accepted-input union for cell functions in httk-atomistic.
"""

import httk.core

from . import cell, cell_backend, cell_view

# A cell is any cell backend/view, a Cell, or any vector-like: a 3x3 basis matrix (nested numbers,
# FracVector, SurdVector, numpy array, ...) or a flat 6-sequence of cell parameters.
type CellLike = cell_backend.CellBackend | cell_view.CellView | cell.Cell | httk.core.VectorLike
