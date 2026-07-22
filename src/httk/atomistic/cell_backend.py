"""
The abstract base class for all cell backends in httk-atomistic.
"""

from typing import Any, ClassVar

from httk.core import Backend

from .cell_api import CellAPI


class CellBackend(Backend["CellBackend"], CellAPI):
    """
    Abstract base class for all backends of cell data.

    Concrete backends carry a native representation and produce the canonical 3x3
    ``matrix`` declared by ``CellAPI`` from it.
    """

    backend_classes: ClassVar[list[type[Backend[Any]]]]
