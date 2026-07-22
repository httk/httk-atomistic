"""
The abstract base class for all structure backends in httk-atomistic.
"""

from typing import Any, ClassVar

from httk.core import Backend

from .structure_api import StructureAPI


class StructureBackend(Backend["StructureBackend"], StructureAPI):
    """
    Abstract base class for all backends of crystal structure data.

    Concrete backends carry a native representation and produce the canonical
    Simple quartet declared by ``StructureAPI`` from it.
    """

    backend_classes: ClassVar[list[type[Backend[Any]]]]
