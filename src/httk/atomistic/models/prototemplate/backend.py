"""The abstract prototemplate backend."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.prototemplate.api import PrototemplateAPI


class PrototemplateBackend(Backend["PrototemplateBackend"], PrototemplateAPI):
    """Backend root for standard-setting anonymous crystal keys."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
    __httk_storage_record__: ClassVar[type[Any]]
