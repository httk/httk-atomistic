"""
The abstract base class for all cell views in httk-atomistic.
"""

from typing import ClassVar, Self

from httk.core import View

from .cell_backend import CellBackend


class CellView(View[CellBackend]):
    """
    Abstract base class for all views of cell data.
    """

    _backend_base_cls: ClassVar[type[CellBackend]] = CellBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


CellView._view_base_cls = CellView
