"""The abstract protochroma backend."""

from typing import Any, ClassVar

from httk.core import Backend

from httk.atomistic.models.protochroma.api import ProtochromaAPI


class ProtochromaBackend(Backend["ProtochromaBackend"], ProtochromaAPI):
    """Backend root for standard-setting anonymous crystal keys."""

    backend_classes: ClassVar[list[type[Backend[Any]]]]
    __httk_storage_record__: ClassVar[type[Any]]
