"""The abstract backend for structuretypes and prototypes."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.structuretype.api import StructuretypeAPI


class StructuretypeBackend(Backend["StructuretypeBackend"], StructuretypeAPI):
    """Backend root for the dummy-species structuretype family."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
    __httk_storage_record__: ClassVar[type[Any]]
