"""
The accepted-input union for cell functions in httk-atomistic.
"""

from typing import Any

from . import cell, cell_backend, cell_view

type CellLike = (
    cell_backend.CellBackend
    | cell_view.CellView
    | cell.Cell
    | tuple[Any, Any, Any]
    | tuple[Any, Any, Any, Any, Any, Any]
    | list[Any]
)
