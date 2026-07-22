"""
The abstract base class for all structure views in httk-atomistic.
"""

from typing import ClassVar, Self

from httk.core import View

from .structure_backend import StructureBackend


class StructureView(View[StructureBackend]):
    """
    Abstract base class for all views of crystal structure data.
    """

    _backend_base_cls: ClassVar[type[StructureBackend]] = StructureBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


StructureView._view_base_cls = StructureView
