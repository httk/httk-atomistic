"""The abstract prototype backend."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.prototype.api import PrototypeAPI


class PrototypeBackend(Backend["PrototypeBackend"], PrototypeAPI):
    """Backend root for anonymous geometrical-class prototypes."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
    __httk_storage_record__: ClassVar[type[Any]]
