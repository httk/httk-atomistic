"""The abstract backend for chromastructures and prototypes."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.chromastructure.api import ChromastructureAPI


class ChromastructureBackend(Backend["ChromastructureBackend"], ChromastructureAPI):
    """Backend root for the dummy-species chromastructure family."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
    __httk_storage_record__: ClassVar[type[Any]]
