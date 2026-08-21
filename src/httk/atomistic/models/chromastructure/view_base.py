"""The abstract view base for chromastructures."""

from typing import ClassVar, Self

from httk.core import View

from httk.atomistic.models.chromastructure.backend import ChromastructureBackend


class ChromastructureViewBase(View[ChromastructureBackend]):
    """Base class for views presenting chromastructure backends."""

    _backend_base_cls: ClassVar[type[ChromastructureBackend]] = ChromastructureBackend  # type: ignore[type-abstract]
    _view_base_cls: ClassVar[type[Self]]


ChromastructureViewBase._view_base_cls = ChromastructureViewBase
