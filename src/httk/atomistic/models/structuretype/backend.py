"""The abstract structuretype backend."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.structuretype.api import StructuretypeAPI


class StructuretypeBackend(Backend["StructuretypeBackend"], StructuretypeAPI):
    """Backend root for assigned geometrical-class structuretypes."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
    __httk_storage_record__: ClassVar[type[Any]]
